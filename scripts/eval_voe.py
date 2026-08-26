#!/usr/bin/env python3
"""Compute VoE surprise curves for a JEPA checkpoint."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from physjepa.eval import load_model_checkpoint, resolve_device, run_voe_surprise, write_json


def main() -> None:
    parser = argparse.ArgumentParser(description="Eval VoE surprise")
    parser.add_argument("--config", type=Path, default=ROOT / "configs" / "eval_default.yaml")
    parser.add_argument("--ckpt", type=Path, default=None)
    parser.add_argument("--model", type=str, default="auto", choices=["auto", "jepa", "pixel"])
    parser.add_argument("--out", type=Path, default=None)
    parser.add_argument("--device", type=str, default=None)
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
    voe_index = ROOT / cfg["data"]["voe_index"]

    result = run_voe_surprise(
        model,
        voe_index,
        device=device,
        spike_window=int(cfg["voe"].get("spike_window", 3)),
        model_type=model_type,
    )

    run_id = ckpt.parent.name
    out_dir = args.out or (ckpt.parent / cfg["eval"].get("out_subdir", "eval"))
    if args.out and not Path(out_dir).is_absolute():
        out_dir = ROOT / out_dir

    payload = {
        "schema_version": 1,
        "model_type": model_type,
        "surprise_metric": result["surprise_metric"],
        "run_id": run_id,
        "ckpt": str(ckpt.name),
        "ckpt_path": str(ckpt),
        "context_len": result["context_len"],
        "pred_horizon": result["pred_horizon"],
        "spike_window": result["spike_window"],
        "by_type": result["by_type"],
        "pairs": result["pairs"],
        "jepa_config": train_cfg.get("model", {}),
    }
    path = write_json(Path(out_dir) / "voe_surprise.json", payload)
    print(f"Wrote {path}")
    for vtype, stats in result["by_type"].items():
        print(
            f"  {vtype}: spike={stats['spike_score']:.4f} "
            f"pre_gap={stats['pre_tstar_abs_gap']:.6f} n={stats['n_pairs']}"
        )


if __name__ == "__main__":
    main()
