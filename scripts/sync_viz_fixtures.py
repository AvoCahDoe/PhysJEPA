#!/usr/bin/env python3
"""Copy viz fixtures + run summaries into dashboard public/fixtures."""

from __future__ import annotations

import argparse
import shutil
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
VIZ = ROOT / "viz" / "fixtures"
OUT = ROOT / "viz" / "dashboard" / "public" / "fixtures"


def copy_tree(src: Path, dst: Path) -> None:
    if dst.exists():
        shutil.rmtree(dst)
    shutil.copytree(src, dst)


def _copy_both(src: Path, name: str) -> None:
    if not src.exists():
        print(f"skip missing {src}")
        return
    VIZ.mkdir(parents=True, exist_ok=True)
    OUT.mkdir(parents=True, exist_ok=True)
    shutil.copy2(src, VIZ / name)
    shutil.copy2(src, OUT / name)
    print(f"copied {src} -> {name}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Sync dashboard fixtures from runs")
    parser.add_argument("--jepa-run", type=str, default="paper_mid")
    parser.add_argument("--pixel-run", type=str, default="paper_mid")
    parser.add_argument(
        "--comparison",
        type=Path,
        default=ROOT / "runs" / "comparison.json",
    )
    parser.add_argument(
        "--ablations",
        type=Path,
        default=ROOT / "runs" / "ablations.json",
    )
    args = parser.parse_args()

    OUT.mkdir(parents=True, exist_ok=True)
    VIZ.mkdir(parents=True, exist_ok=True)

    # Prefer live run artifacts; fall back to existing viz/fixtures copies.
    comparison = args.comparison if args.comparison.is_absolute() else ROOT / args.comparison
    if comparison.exists():
        _copy_both(comparison, "sample_comparison.json")
    elif (VIZ / "sample_comparison.json").exists():
        shutil.copy2(VIZ / "sample_comparison.json", OUT / "sample_comparison.json")
        print("copied sample_comparison.json from viz/fixtures")
    else:
        print("skip missing comparison.json")

    jepa_voe = ROOT / "runs" / "jepa" / args.jepa_run / "eval" / "voe_surprise.json"
    if jepa_voe.exists():
        _copy_both(jepa_voe, "sample_eval.json")
    elif (VIZ / "sample_eval.json").exists():
        shutil.copy2(VIZ / "sample_eval.json", OUT / "sample_eval.json")
        print("copied sample_eval.json from viz/fixtures")
    else:
        print(f"skip missing {jepa_voe}")

    abl = args.ablations if args.ablations.is_absolute() else ROOT / args.ablations
    if abl.exists():
        _copy_both(abl, "sample_ablations.json")
    elif (VIZ / "sample_ablations.json").exists():
        shutil.copy2(VIZ / "sample_ablations.json", OUT / "sample_ablations.json")
        print("copied sample_ablations.json from viz/fixtures")
    else:
        print("skip missing ablations.json")

    ep = VIZ / "sample_episode"
    if ep.exists():
        copy_tree(ep, OUT / "sample_episode")
    else:
        print(f"skip missing {ep}")

    for model, run_id in (("jepa", args.jepa_run), ("pixel", args.pixel_run)):
        src = ROOT / "runs" / model / run_id / "summary.json"
        if src.exists():
            OUT.mkdir(parents=True, exist_ok=True)
            shutil.copy2(src, OUT / f"{model}_summary.json")
            shutil.copy2(src, VIZ / f"{model}_summary.json")
            print(f"copied {src} -> {model}_summary.json")
        else:
            print(f"skip missing {src}")

    print(f"Synced fixtures -> {OUT} (jepa={args.jepa_run}, pixel={args.pixel_run})")


if __name__ == "__main__":
    main()
