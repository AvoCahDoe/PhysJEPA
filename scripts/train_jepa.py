#!/usr/bin/env python3
"""Train JEPA on procedural physics rollouts."""

from __future__ import annotations

import argparse
import sys
from datetime import datetime, timezone
from pathlib import Path

import torch
import yaml
from torch.utils.data import DataLoader

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from physjepa.data import JEPAWindowDataset, RolloutDataset, split_entries_by_seed
from physjepa.models import JEPAModule
from physjepa.train import MetricsWriter, jepa_collate, train_jepa


def resolve_device(name: str) -> torch.device:
    if name == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(name)


def main() -> None:
    parser = argparse.ArgumentParser(description="Train JEPA")
    parser.add_argument("--config", type=Path, default=ROOT / "configs" / "jepa_default.yaml")
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--device", type=str, default=None)
    parser.add_argument("--run-id", type=str, default=None)
    parser.add_argument("--predictor-type", type=str, default=None, help="mlp | gru | transformer")
    parser.add_argument("--context-len", type=int, default=None)
    args = parser.parse_args()

    with args.config.open("r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    if args.epochs is not None:
        cfg["optim"]["epochs"] = args.epochs
    if args.batch_size is not None:
        cfg["optim"]["batch_size"] = args.batch_size
    if args.device is not None:
        cfg["train"]["device"] = args.device
    if args.predictor_type is not None:
        cfg.setdefault("model", {})["predictor_type"] = args.predictor_type
    if args.context_len is not None:
        cfg.setdefault("model", {})["context_len"] = args.context_len

    torch.manual_seed(int(cfg["train"]["seed"]))

    data_cfg = cfg["data"]
    model_cfg = cfg["model"]
    optim_cfg = cfg["optim"]
    train_cfg = cfg["train"]

    index_path = ROOT / data_cfg["train_index"]
    base = RolloutDataset(
        index_path,
        seed_max_exclusive=int(data_cfg.get("seed_max_exclusive", 9000)),
        cache_frames=bool(data_cfg.get("cache_frames", True)),
    )
    train_entries, val_entries = split_entries_by_seed(
        base.entries, float(data_cfg.get("val_fraction", 0.1))
    )

    train_rollout = RolloutDataset(
        root=base.root,
        entries=train_entries,
        cache_frames=True,
    )
    val_rollout = RolloutDataset(
        root=base.root,
        entries=val_entries,
        cache_frames=True,
    )

    ctx = int(model_cfg["context_len"])
    hor = int(model_cfg["pred_horizon"])
    wpe = int(data_cfg.get("windows_per_episode", 4))

    train_ds = JEPAWindowDataset(
        train_rollout,
        context_len=ctx,
        pred_horizon=hor,
        windows_per_episode=wpe,
        seed=int(train_cfg["seed"]),
    )
    val_ds = JEPAWindowDataset(
        val_rollout,
        context_len=ctx,
        pred_horizon=hor,
        windows_per_episode=max(1, wpe // 2),
        seed=int(train_cfg["seed"]) + 1,
    )

    train_loader = DataLoader(
        train_ds,
        batch_size=int(optim_cfg["batch_size"]),
        shuffle=True,
        num_workers=int(data_cfg.get("num_workers", 0)),
        collate_fn=jepa_collate,
        drop_last=len(train_ds) >= int(optim_cfg["batch_size"]),
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=int(optim_cfg["batch_size"]),
        shuffle=False,
        num_workers=int(data_cfg.get("num_workers", 0)),
        collate_fn=jepa_collate,
    )

    device = resolve_device(str(train_cfg.get("device", "auto")))
    model = JEPAModule(
        latent_dim=int(model_cfg["latent_dim"]),
        base_channels=int(model_cfg["base_channels"]),
        context_len=ctx,
        pred_horizon=hor,
        predictor_hidden=int(model_cfg["predictor_hidden"]),
        predictor_layers=int(model_cfg["predictor_layers"]),
        ema_momentum=float(model_cfg["ema_momentum"]),
        predictor_type=str(model_cfg.get("predictor_type", "mlp")),
    )
    print("params", model.count_parameters(), "predictor", model.predictor_type)
    print(f"device={device} train_windows={len(train_ds)} val_windows={len(val_ds)}")

    optimizer = torch.optim.AdamW(
        list(model.encoder.parameters()) + list(model.predictor.parameters()),
        lr=float(optim_cfg["lr"]),
        weight_decay=float(optim_cfg["weight_decay"]),
    )

    run_id = args.run_id or datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    run_dir = ROOT / train_cfg["run_root"] / run_id
    metrics = MetricsWriter(run_dir=run_dir, run_id=run_id, config=cfg)

    result = train_jepa(
        model,
        train_loader,
        val_loader if len(val_ds) > 0 else None,
        optimizer,
        device=device,
        epochs=int(optim_cfg["epochs"]),
        grad_clip=float(optim_cfg["grad_clip"]),
        ema_base=float(model_cfg["ema_momentum"]),
        ema_final=float(model_cfg.get("ema_final", 1.0)),
        log_every=int(train_cfg.get("log_every", 10)),
        metrics=metrics,
        ckpt_dir=run_dir,
    )
    print(f"Done. run_dir={run_dir} best_val={result['best_val_loss']}")


if __name__ == "__main__":
    main()
