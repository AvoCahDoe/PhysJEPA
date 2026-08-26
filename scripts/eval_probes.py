#!/usr/bin/env python3
"""Train frozen-encoder linear probes on physics targets."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from physjepa.eval import load_model_checkpoint, resolve_device, run_linear_probes, write_json


def main() -> None:
    parser = argparse.ArgumentParser(description="Eval linear probes")
    parser.add_argument("--config", type=Path, default=ROOT / "configs" / "eval_default.yaml")
    parser.add_argument("--ckpt", type=Path, default=None)
    parser.add_argument("--model", type=str, default="auto", choices=["auto", "jepa", "pixel"])
    parser.add_argument("--out", type=Path, default=None)
    parser.add_argument("--device", type=str, default=None)
    parser.add_argument("--epochs", type=int, default=None)
    args = parser.parse_args()

    with args.config.open("r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    ckpt = Path(args.ckpt or cfg["checkpoint"]["path"])
    if not ckpt.is_absolute():
        ckpt = ROOT / ckpt
    mt = None if args.model == "auto" else args.model
    config_path = ROOT / cfg["checkpoint"].get("config", "configs/jepa_default.yaml")
    if mt == "pixel" or "pixel" in ckpt.parts:
        config_path = ROOT / "configs" / "pixel_default.yaml"
    device = resolve_device(args.device or cfg["eval"].get("device", "auto"))

    model, train_cfg, model_type = load_model_checkpoint(
        ckpt, config_path=config_path, model_type=mt, device=device
    )
    probes_cfg = cfg["probes"]
    data_cfg = cfg["data"]

    result = run_linear_probes(
        model,
        ROOT / data_cfg["train_index"],
        val_fraction=float(data_cfg.get("val_fraction", 0.1)),
        seed_max_exclusive=int(data_cfg.get("seed_max_exclusive", 9000)),
        body_id=int(probes_cfg.get("body_id", 0)),
        targets=list(probes_cfg.get("targets", ["xy", "vxvy", "mass", "visible"])),
        epochs=int(args.epochs or probes_cfg.get("epochs", 30)),
        lr=float(probes_cfg.get("lr", 1e-3)),
        batch_size=int(probes_cfg.get("batch_size", 64)),
        device=device,
    )

    run_id = ckpt.parent.name
    out_dir = args.out or (ckpt.parent / cfg["eval"].get("out_subdir", "eval"))
    if not Path(out_dir).is_absolute():
        out_dir = ROOT / out_dir if args.out else out_dir

    payload = {
        "schema_version": 1,
        "model_type": model_type,
        "run_id": run_id,
        "ckpt": str(ckpt.name),
        "ckpt_path": str(ckpt),
        "body_id": result["body_id"],
        "n_train_episodes": result["n_train_episodes"],
        "n_val_episodes": result["n_val_episodes"],
        "latent_dim": result["latent_dim"],
        "targets": result["targets"],
        "jepa_config": train_cfg.get("model", {}),
    }
    path = write_json(Path(out_dir) / "probes.json", payload)
    print(f"Wrote {path}")
    for name, metrics in result["targets"].items():
        print(f"  {name}: {metrics}")


if __name__ == "__main__":
    main()
