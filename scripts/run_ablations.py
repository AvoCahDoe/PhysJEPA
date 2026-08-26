#!/usr/bin/env python3
"""
Phase 6 ablations:
  1) Occlusion-duration VoE sets (short / medium / long)
  2) Predictor architecture (mlp / gru / transformer)

Writes React-ready runs/ablations.json (+ per-run VoE JSON under runs/jepa/*/eval/).
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from physjepa.eval import (  # noqa: E402
    load_model_checkpoint,
    resolve_device,
    run_linear_probes,
    run_voe_surprise,
    write_json,
)


def _run(cmd: list[str]) -> None:
    print("+", " ".join(cmd))
    subprocess.run(cmd, cwd=ROOT, check=True)


def _mean_occ_frames(voe_dir: Path) -> float | None:
    vals = []
    idx = voe_dir / "index.json"
    if not idx.exists():
        return None
    data = json.loads(idx.read_text(encoding="utf-8"))
    for entry in data.get("pairs", []):
        if "occlusion_frames" in entry:
            vals.append(float(entry["occlusion_frames"]))
            continue
        meta = voe_dir / entry["pair_id"] / "pair_meta.json"
        if meta.exists():
            m = json.loads(meta.read_text(encoding="utf-8"))
            if m.get("occlusion_frames") is not None:
                vals.append(float(m["occlusion_frames"]))
    return sum(vals) / len(vals) if vals else None


def generate_occ_sets(cfg: dict) -> dict[str, Path]:
    outs: dict[str, Path] = {}
    occ = cfg["occlusion"]
    for label in occ["labels"]:
        voe_cfg = ROOT / occ["configs"][label]
        out_dir = ROOT / occ["out_dirs"][label]
        _run(
            [
                sys.executable,
                "scripts/generate_voe.py",
                "--voe-config",
                str(voe_cfg),
                "--out",
                str(out_dir),
            ]
        )
        outs[label] = out_dir
    return outs


def train_arch(cfg: dict, predictor: str, epochs: int, batch_size: int, device: str) -> Path:
    arch = cfg["architecture"]
    config_path = ROOT / arch["configs"][predictor]
    run_id = f"ablate_{predictor}"
    reuse = ROOT / arch.get("reuse_mlp_ckpt", "")
    if predictor == "mlp" and reuse.exists():
        print(f"Reusing existing MLP checkpoint: {reuse}")
        return reuse

    _run(
        [
            sys.executable,
            "scripts/train_jepa.py",
            "--config",
            str(config_path),
            "--epochs",
            str(epochs),
            "--batch-size",
            str(batch_size),
            "--device",
            device,
            "--run-id",
            run_id,
            "--predictor-type",
            predictor,
        ]
    )
    return ROOT / "runs" / "jepa" / run_id / "ckpt_last.pt"


def eval_voe_on(
    ckpt: Path,
    voe_index: Path,
    device,
    spike_window: int,
    model_type: str = "jepa",
) -> dict:
    model, train_cfg, mt = load_model_checkpoint(
        ckpt,
        config_path=None,
        model_type=model_type,
        device=device,
    )
    result = run_voe_surprise(
        model,
        voe_index,
        device=device,
        spike_window=spike_window,
        model_type=mt,
    )
    # Summarize spikes
    spikes = {
        k: {
            "spike_score": v["spike_score"],
            "pre_tstar_abs_gap": v["pre_tstar_abs_gap"],
            "t_star": v["t_star"],
            "n_pairs": v.get("n_pairs"),
        }
        for k, v in result["by_type"].items()
    }
    return {
        "ckpt": str(ckpt),
        "run_id": ckpt.parent.name,
        "predictor_type": train_cfg.get("model", {}).get("predictor_type", "mlp"),
        "context_len": result["context_len"],
        "by_type": spikes,
        "mean_spike": (
            sum(s["spike_score"] for s in spikes.values()) / max(len(spikes), 1)
        ),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Run Phase 6 ablations")
    parser.add_argument("--config", type=Path, default=ROOT / "configs" / "ablations.yaml")
    parser.add_argument("--skip-generate", action="store_true")
    parser.add_argument("--skip-train", action="store_true")
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--device", type=str, default=None)
    parser.add_argument("--out", type=Path, default=None)
    args = parser.parse_args()

    with args.config.open("r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    device_name = args.device or cfg["train"].get("device", "auto")
    device = resolve_device(device_name)
    epochs = int(args.epochs if args.epochs is not None else cfg["train"]["epochs"])
    batch_size = int(cfg["train"]["batch_size"])
    spike_window = int(cfg["eval"]["spike_window"])

    # 1) Occlusion VoE generation
    if args.skip_generate:
        occ_dirs = {
            label: ROOT / cfg["occlusion"]["out_dirs"][label]
            for label in cfg["occlusion"]["labels"]
        }
    else:
        occ_dirs = generate_occ_sets(cfg)

    # 2) Architecture training
    ckpts: dict[str, Path] = {}
    for predictor in cfg["architecture"]["predictors"]:
        if args.skip_train and predictor != "mlp":
            p = ROOT / "runs" / "jepa" / f"ablate_{predictor}" / "ckpt_last.pt"
            if not p.exists():
                raise SystemExit(f"Missing checkpoint for {predictor}: {p}")
            ckpts[predictor] = p
        else:
            ckpts[predictor] = train_arch(cfg, predictor, epochs, batch_size, device_name)

    # Prefer last over missing best
    for k, p in list(ckpts.items()):
        if not p.exists():
            alt = p.parent / "ckpt_best.pt"
            if alt.exists():
                ckpts[k] = alt

    # 3) Occlusion ablation: fixed MLP (or first available) across occ lengths
    mlp_ckpt = ckpts.get("mlp") or next(iter(ckpts.values()))
    occlusion_results = []
    for label, out_dir in occ_dirs.items():
        idx = out_dir / "index.json"
        voe = eval_voe_on(mlp_ckpt, idx, device, spike_window)
        mean_occ = _mean_occ_frames(out_dir)
        # teleport spike is the relevant metric for this ablation
        tele = None
        for key, stats in voe["by_type"].items():
            if "teleport" in key or "occ" in key:
                tele = stats
                break
        if tele is None and voe["by_type"]:
            tele = next(iter(voe["by_type"].values()))
        occlusion_results.append(
            {
                "label": label,
                "voe_dir": str(out_dir),
                "mean_occlusion_frames": mean_occ,
                "spike_score": tele["spike_score"] if tele else None,
                "pre_tstar_abs_gap": tele["pre_tstar_abs_gap"] if tele else None,
                "t_star": tele["t_star"] if tele else None,
                "ckpt": str(mlp_ckpt),
            }
        )

    # 4) Architecture ablation: each predictor on default VoE
    default_voe = ROOT / cfg["data"]["voe_index"]
    architecture_results = []
    for predictor, ckpt in ckpts.items():
        voe = eval_voe_on(ckpt, default_voe, device, spike_window)
        probes = run_linear_probes(
            load_model_checkpoint(ckpt, model_type="jepa", device=device)[0],
            ROOT / cfg["data"]["train_index"],
            val_fraction=float(cfg["data"].get("val_fraction", 0.1)),
            seed_max_exclusive=int(cfg["data"].get("seed_max_exclusive", 9000)),
            epochs=int(cfg["eval"].get("probe_epochs", 10)),
            device=device,
        )
        architecture_results.append(
            {
                "predictor_type": predictor,
                "ckpt": str(ckpt),
                "mean_voe_spike": voe["mean_spike"],
                "voe_by_type": voe["by_type"],
                "probes": {
                    k: {
                        "val_r2": v.get("val_r2"),
                        "val_acc": v.get("val_acc"),
                        "val_mse": v.get("val_mse"),
                    }
                    for k, v in probes["targets"].items()
                },
            }
        )

    payload = {
        "schema_version": 1,
        "occlusion_duration": {
            "description": "JEPA VoE spike vs occlusion length (teleport under occluder)",
            "results": occlusion_results,
        },
        "architecture": {
            "description": "Predictor type vs VoE spike + linear probes (same data)",
            "results": architecture_results,
        },
    }
    out = args.out or ROOT / cfg.get("out", "runs/ablations.json")
    if not Path(out).is_absolute():
        out = ROOT / out
    write_json(out, payload)
    print(f"Wrote {out}")
    print("Occlusion:")
    for r in occlusion_results:
        print(
            f"  {r['label']}: occ_frames={r['mean_occlusion_frames']} "
            f"spike={r['spike_score']}"
        )
    print("Architecture:")
    for r in architecture_results:
        print(f"  {r['predictor_type']}: mean_spike={r['mean_voe_spike']:.5f}")


if __name__ == "__main__":
    main()
