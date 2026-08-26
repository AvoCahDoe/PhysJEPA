#!/usr/bin/env python3
"""
Mid-scale paper pipeline: generate data → train JEPA/pixel → eval → ablations → sync/figures.

Resume with --skip-* flags after a partial run.
"""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def _run(cmd: list[str]) -> None:
    print("+", " ".join(cmd))
    subprocess.run(cmd, cwd=ROOT, check=True)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run mid-scale paper_mid training + eval + demo fixture sync"
    )
    parser.add_argument("--n-train", type=int, default=1000, help="Train episode count")
    parser.add_argument("--jepa-run", type=str, default="paper_mid")
    parser.add_argument("--pixel-run", type=str, default="paper_mid")
    parser.add_argument("--epochs", type=int, default=None, help="Override train/ablation epochs")
    parser.add_argument("--device", type=str, default="auto")
    parser.add_argument("--skip-generate", action="store_true")
    parser.add_argument("--skip-train", action="store_true")
    parser.add_argument("--skip-eval", action="store_true")
    parser.add_argument("--skip-ablations", action="store_true")
    parser.add_argument("--skip-sync", action="store_true")
    parser.add_argument("--skip-figures", action="store_true")
    args = parser.parse_args()

    py = sys.executable
    jepa_ckpt = ROOT / "runs" / "jepa" / args.jepa_run / "ckpt_last.pt"
    pixel_ckpt = ROOT / "runs" / "pixel" / args.pixel_run / "ckpt_last.pt"

    if not args.skip_generate:
        _run(
            [
                py,
                "scripts/generate_train.py",
                "--n",
                str(args.n_train),
                "--out",
                "data/train",
            ]
        )
        _run([py, "scripts/generate_voe.py", "--out", "data/voe"])

    if not args.skip_train:
        jepa_cmd = [
            py,
            "scripts/train_jepa.py",
            "--config",
            "configs/jepa_paper.yaml",
            "--run-id",
            args.jepa_run,
            "--device",
            args.device,
        ]
        pixel_cmd = [
            py,
            "scripts/train_pixel.py",
            "--config",
            "configs/pixel_paper.yaml",
            "--run-id",
            args.pixel_run,
            "--device",
            args.device,
        ]
        if args.epochs is not None:
            jepa_cmd.extend(["--epochs", str(args.epochs)])
            pixel_cmd.extend(["--epochs", str(args.epochs)])
        _run(jepa_cmd)
        _run(pixel_cmd)

    if not args.skip_eval:
        if not jepa_ckpt.exists() or not pixel_ckpt.exists():
            raise SystemExit(f"Missing ckpts: {jepa_ckpt} / {pixel_ckpt}")
        _run([py, "scripts/eval_probes.py", "--ckpt", str(jepa_ckpt)])
        _run([py, "scripts/eval_voe.py", "--ckpt", str(jepa_ckpt)])
        _run(
            [
                py,
                "scripts/eval_probes.py",
                "--ckpt",
                str(pixel_ckpt),
                "--model",
                "pixel",
            ]
        )
        _run(
            [
                py,
                "scripts/eval_voe.py",
                "--ckpt",
                str(pixel_ckpt),
                "--model",
                "pixel",
            ]
        )
        _run(
            [
                py,
                "scripts/eval_compare.py",
                "--jepa-ckpt",
                str(jepa_ckpt),
                "--pixel-ckpt",
                str(pixel_ckpt),
                "--out",
                "runs/comparison.json",
            ]
        )

    if not args.skip_ablations:
        abl_cmd = [
            py,
            "scripts/run_ablations.py",
            "--config",
            "configs/ablations_paper.yaml",
            "--device",
            args.device,
        ]
        if args.epochs is not None:
            abl_cmd.extend(["--epochs", str(args.epochs)])
        _run(abl_cmd)

    if not args.skip_sync:
        _run(
            [
                py,
                "scripts/sync_viz_fixtures.py",
                "--jepa-run",
                args.jepa_run,
                "--pixel-run",
                args.pixel_run,
            ]
        )

    if not args.skip_figures:
        _run([py, "scripts/export_report_figures.py"])

    print("paper_mid pipeline complete.")


if __name__ == "__main__":
    main()
