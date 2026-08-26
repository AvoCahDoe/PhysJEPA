#!/usr/bin/env python3
"""Compare JEPA vs pixel baseline on probes + VoE (React-ready comparison.json)."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from physjepa.eval import (
    load_model_checkpoint,
    resolve_device,
    run_linear_probes,
    run_voe_surprise,
    write_json,
)


def _run_one(
    ckpt: Path,
    model_type: str,
    cfg: dict,
    device,
) -> dict:
    config_path = ROOT / (
        "configs/pixel_default.yaml" if model_type == "pixel" else "configs/jepa_default.yaml"
    )
    model, train_cfg, mt = load_model_checkpoint(
        ckpt, config_path=config_path, model_type=model_type, device=device
    )
    probes = run_linear_probes(
        model,
        ROOT / cfg["data"]["train_index"],
        val_fraction=float(cfg["data"].get("val_fraction", 0.1)),
        seed_max_exclusive=int(cfg["data"].get("seed_max_exclusive", 9000)),
        body_id=int(cfg["probes"].get("body_id", 0)),
        targets=list(cfg["probes"].get("targets", ["xy", "vxvy", "mass", "visible"])),
        epochs=int(cfg["probes"].get("epochs", 30)),
        lr=float(cfg["probes"].get("lr", 1e-3)),
        batch_size=int(cfg["probes"].get("batch_size", 64)),
        device=device,
    )
    voe = run_voe_surprise(
        model,
        ROOT / cfg["data"]["voe_index"],
        device=device,
        spike_window=int(cfg["voe"].get("spike_window", 3)),
        model_type=mt,
    )
    return {
        "model_type": mt,
        "ckpt": str(ckpt),
        "run_id": ckpt.parent.name,
        "probes": probes["targets"],
        "voe_by_type": {
            k: {
                "spike_score": v["spike_score"],
                "pre_tstar_abs_gap": v["pre_tstar_abs_gap"],
                "t_star": v["t_star"],
            }
            for k, v in voe["by_type"].items()
        },
        "voe_surprise_metric": voe["surprise_metric"],
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Compare JEPA vs pixel baseline")
    parser.add_argument("--config", type=Path, default=ROOT / "configs" / "eval_default.yaml")
    parser.add_argument("--jepa-ckpt", type=Path, required=True)
    parser.add_argument("--pixel-ckpt", type=Path, required=True)
    parser.add_argument("--out", type=Path, default=ROOT / "runs" / "comparison.json")
    parser.add_argument("--device", type=str, default="auto")
    args = parser.parse_args()

    with args.config.open("r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    device = resolve_device(args.device)
    jepa_ckpt = args.jepa_ckpt if args.jepa_ckpt.is_absolute() else ROOT / args.jepa_ckpt
    pixel_ckpt = args.pixel_ckpt if args.pixel_ckpt.is_absolute() else ROOT / args.pixel_ckpt

    jepa = _run_one(jepa_ckpt, "jepa", cfg, device)
    pixel = _run_one(pixel_ckpt, "pixel", cfg, device)

    # Headline deltas: JEPA spike - pixel spike per violation type
    delta_spike = {}
    for vtype in jepa["voe_by_type"]:
        delta_spike[vtype] = (
            jepa["voe_by_type"][vtype]["spike_score"]
            - pixel["voe_by_type"][vtype]["spike_score"]
        )

    payload = {
        "schema_version": 1,
        "jepa": jepa,
        "pixel": pixel,
        "delta_voe_spike_jepa_minus_pixel": delta_spike,
    }
    out = args.out if args.out.is_absolute() else ROOT / args.out
    write_json(out, payload)
    print(f"Wrote {out}")
    for vtype, d in delta_spike.items():
        print(f"  {vtype}: delta_spike={d:.5f}")


if __name__ == "__main__":
    main()
