#!/usr/bin/env python3
"""Export report figures from fixture / run JSON (no training)."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[1]


def _load(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def _style() -> None:
    plt.rcParams.update(
        {
            "figure.facecolor": "white",
            "axes.facecolor": "white",
            "axes.grid": True,
            "grid.alpha": 0.3,
            "font.size": 11,
        }
    )


def _summary_path(model: str) -> Path:
    for candidate in (
        ROOT / "viz" / "fixtures" / f"{model}_summary.json",
        ROOT / "viz" / "dashboard" / "public" / "fixtures" / f"{model}_summary.json",
    ):
        if candidate.exists():
            return candidate
    raise FileNotFoundError(f"No {model}_summary.json under viz/fixtures or dashboard public")


def fig_train_loss(out: Path) -> None:
    jepa = _load(_summary_path("jepa"))
    pixel = _load(_summary_path("pixel"))
    fig, ax = plt.subplots(figsize=(6.5, 3.8))
    ax.plot(jepa["curves"]["step"], jepa["curves"]["loss"], label="JEPA", color="#2c7bb6")
    ax.plot(pixel["curves"]["step"], pixel["curves"]["loss"], label="Pixel", color="#d7191c")
    ax.set_xlabel("step")
    ax.set_ylabel("train loss")
    ax.set_title("Training loss (paper_mid, 80 epochs)")
    ax.legend()
    fig.tight_layout()
    fig.savefig(out, dpi=150)
    plt.close(fig)


def _fig_voe_type(out: Path, violation_type: str, filename: str) -> None:
    voe = _load(ROOT / "viz/fixtures/sample_eval.json")
    block = voe["by_type"][violation_type]
    t = np.array(block["t"])
    poss = np.array([np.nan if v is None else v for v in block["possible_mean"]], dtype=float)
    imposs = np.array([np.nan if v is None else v for v in block["impossible_mean"]], dtype=float)
    fig, ax = plt.subplots(figsize=(6.5, 3.8))
    ax.plot(t, poss, label="possible", color="#2c7bb6")
    ax.plot(t, imposs, label="impossible", color="#d7191c")
    ax.axvline(block["t_star"], color="#fdae61", linestyle="--", label=f"t*={block['t_star']}")
    ax.set_xlabel("timestep t")
    ax.set_ylabel("surprise (smooth L1)")
    label = violation_type.replace("_", " ")
    ax.set_title(f"VoE {label} (spike={block['spike_score']:.4f})")
    ax.legend()
    fig.tight_layout()
    fig.savefig(out, dpi=150)
    plt.close(fig)


def fig_voe_teleport(out: Path) -> None:
    _fig_voe_type(out, "teleport_occlusion", "fig_voe_teleport.png")


def fig_voe_bounce(out: Path) -> None:
    _fig_voe_type(out, "impossible_bounce", "fig_voe_bounce.png")


def fig_comparison_delta(out: Path) -> None:
    cmp_ = _load(ROOT / "viz/fixtures/sample_comparison.json")
    deltas = cmp_["delta_voe_spike_jepa_minus_pixel"]
    labels = [k.replace("_", "\n") for k in deltas]
    vals = list(deltas.values())
    colors = ["#2c7bb6" if v >= 0 else "#d7191c" for v in vals]
    fig, ax = plt.subplots(figsize=(7.0, 3.8))
    ax.bar(range(len(vals)), vals, color=colors)
    ax.set_xticks(range(len(vals)))
    ax.set_xticklabels(labels, fontsize=9)
    ax.axhline(0, color="#333", linewidth=0.8)
    ax.set_ylabel("Δ spike (JEPA − pixel)")
    ax.set_title("VoE spike delta by violation type")
    fig.tight_layout()
    fig.savefig(out, dpi=150)
    plt.close(fig)


def fig_ablation_occlusion(out: Path) -> None:
    abl = _load(ROOT / "viz/fixtures/sample_ablations.json")
    rows = abl["occlusion_duration"]["results"]
    labels = [r["label"] for r in rows]
    occ = [r["mean_occlusion_frames"] for r in rows]
    spike = [r["spike_score"] for r in rows]
    fig, ax1 = plt.subplots(figsize=(6.5, 3.8))
    ax2 = ax1.twinx()
    ax1.plot(labels, occ, "o-", color="#fdae61", label="occluded frames")
    ax2.plot(labels, spike, "s-", color="#2c7bb6", label="VoE spike")
    ax1.set_ylabel("mean occluded frames")
    ax2.set_ylabel("VoE spike")
    ax1.set_title("Occlusion-duration ablation")
    lines1, lab1 = ax1.get_legend_handles_labels()
    lines2, lab2 = ax2.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, lab1 + lab2, loc="best")
    fig.tight_layout()
    fig.savefig(out, dpi=150)
    plt.close(fig)


def fig_ablation_arch(out: Path) -> None:
    abl = _load(ROOT / "viz/fixtures/sample_ablations.json")
    rows = abl["architecture"]["results"]
    labels = [r["predictor_type"] for r in rows]
    vals = [r["mean_voe_spike"] for r in rows]
    fig, ax = plt.subplots(figsize=(5.5, 3.8))
    ax.bar(labels, vals, color="#2c7bb6")
    ax.axhline(0, color="#333", linewidth=0.8)
    ax.set_ylabel("mean VoE spike")
    ax.set_title("Predictor architecture ablation")
    fig.tight_layout()
    fig.savefig(out, dpi=150)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description="Export report figures")
    parser.add_argument("--out", type=Path, default=ROOT / "docs" / "figures")
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    _style()
    fig_train_loss(args.out / "fig_train_loss.png")
    fig_voe_teleport(args.out / "fig_voe_teleport.png")
    fig_voe_bounce(args.out / "fig_voe_bounce.png")
    fig_comparison_delta(args.out / "fig_comparison_delta.png")
    fig_ablation_occlusion(args.out / "fig_ablation_occlusion.png")
    fig_ablation_arch(args.out / "fig_ablation_arch.png")
    print(f"Wrote figures to {args.out}")


if __name__ == "__main__":
    main()
