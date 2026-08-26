"""VoE surprise: per-timestep JEPA prediction error on matched pairs."""

from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F

from physjepa.data import load_episode_frames
from physjepa.models import JEPAModule, PixelVideoModule


def frames_to_tensor(frames_nhwc: np.ndarray) -> torch.Tensor:
    """(T,H,W,3) uint8/float -> (T,3,H,W) float [0,1]."""
    x = frames_nhwc.astype(np.float32) / 255.0
    return torch.from_numpy(x).permute(0, 3, 1, 2).contiguous()


@torch.no_grad()
def surprise_curve_jepa(
    model: JEPAModule,
    frames: torch.Tensor,
    *,
    device: torch.device,
) -> list[float | None]:
    """
    frames: (T, 3, H, W)
    Returns length-T list; None where context/future window invalid.
    Surprise at t uses context [t-c+1, t] predicting [t+1, t+k].
    """
    model.eval()
    T = frames.shape[0]
    c = model.context_len
    k = model.pred_horizon
    out: list[float | None] = [None] * T
    frames = frames.to(device)

    for t in range(T):
        start = t - c + 1
        fut_end = t + 1 + k
        if start < 0 or fut_end > T:
            continue
        context = frames[start : t + 1].unsqueeze(0)  # (1,c,3,H,W)
        future = frames[t + 1 : fut_end].unsqueeze(0)
        context_z = model.encode(context)
        pred_z = model.predictor(context_z)
        target_z = model.encode_target(future)
        loss = F.smooth_l1_loss(pred_z, target_z, reduction="mean")
        out[t] = float(loss.item())
    return out


@torch.no_grad()
def surprise_curve_pixel(
    model: PixelVideoModule,
    frames: torch.Tensor,
    *,
    device: torch.device,
) -> list[float | None]:
    """Pixel-space prediction error (smooth L1) at each valid timestep."""
    model.eval()
    T = frames.shape[0]
    c = model.context_len
    k = model.pred_horizon
    out: list[float | None] = [None] * T
    frames = frames.to(device)

    for t in range(T):
        start = t - c + 1
        fut_end = t + 1 + k
        if start < 0 or fut_end > T:
            continue
        context = frames[start : t + 1].unsqueeze(0)
        future = frames[t + 1 : fut_end].unsqueeze(0)
        pred = model.predict_pixels(context)
        loss = F.smooth_l1_loss(pred, future, reduction="mean")
        out[t] = float(loss.item())
    return out


def surprise_curve(
    model: JEPAModule | PixelVideoModule,
    frames: torch.Tensor,
    *,
    device: torch.device,
) -> list[float | None]:
    if isinstance(model, PixelVideoModule):
        return surprise_curve_pixel(model, frames, device=device)
    return surprise_curve_jepa(model, frames, device=device)


def _nanmean_stack(curves: list[list[float | None]]) -> tuple[list[float | None], list[float | None]]:
    if not curves:
        return [], []
    width = max(len(c) for c in curves)
    mean_list: list[float | None] = []
    std_list: list[float | None] = []
    arr = np.array(
        [[(np.nan if v is None else v) for v in c] + [np.nan] * (width - len(c)) for c in curves],
        dtype=np.float64,
    )
    for col in range(width):
        vals = arr[:, col]
        valid = vals[~np.isnan(vals)]
        if valid.size == 0:
            mean_list.append(None)
            std_list.append(None)
        else:
            mean_list.append(float(valid.mean()))
            std_list.append(float(valid.std(ddof=0)))
    return mean_list, std_list

def spike_score(
    possible: list[float | None],
    impossible: list[float | None],
    t_star: int,
    window: int = 3,
) -> float:
    def _win(curve: list[float | None]) -> float:
        vals = []
        for t in range(t_star, min(t_star + window, len(curve))):
            if curve[t] is not None:
                vals.append(curve[t])
        return float(np.mean(vals)) if vals else float("nan")

    return _win(impossible) - _win(possible)


def pre_tstar_gap(
    possible: list[float | None],
    impossible: list[float | None],
    t_star: int,
) -> float:
    diffs = []
    for t in range(t_star):
        if possible[t] is not None and impossible[t] is not None:
            diffs.append(abs(possible[t] - impossible[t]))
    return float(np.mean(diffs)) if diffs else float("nan")


def run_voe_surprise(
    model: JEPAModule | PixelVideoModule,
    voe_index: str | Path,
    *,
    device: torch.device | None = None,
    spike_window: int = 3,
    model_type: str = "jepa",
) -> dict[str, Any]:
    device = device or torch.device("cpu")
    voe_index = Path(voe_index)
    root = voe_index.parent
    with voe_index.open("r", encoding="utf-8") as f:
        index = json.load(f)

    pairs_out: list[dict[str, Any]] = []
    by_type_curves: dict[str, dict[str, list]] = defaultdict(
        lambda: {"possible": [], "impossible": [], "t_star": None}
    )

    for entry in index["pairs"]:
        pair_id = entry["pair_id"]
        vtype = entry["violation_type"]
        t_star = int(entry["t_star"])
        poss_dir = root / entry["possible"]
        imposs_dir = root / entry["impossible"]

        poss_frames = frames_to_tensor(load_episode_frames(poss_dir))
        imposs_frames = frames_to_tensor(load_episode_frames(imposs_dir))

        s_poss = surprise_curve(model, poss_frames, device=device)
        s_imposs = surprise_curve(model, imposs_frames, device=device)

        sp = spike_score(s_poss, s_imposs, t_star, spike_window)
        pre_gap = pre_tstar_gap(s_poss, s_imposs, t_star)

        pairs_out.append(
            {
                "pair_id": pair_id,
                "violation_type": vtype,
                "t_star": t_star,
                "matched_seed": entry.get("matched_seed"),
                "spike_score": sp,
                "pre_tstar_abs_gap": pre_gap,
                "curves": {"possible": s_poss, "impossible": s_imposs},
            }
        )
        by_type_curves[vtype]["possible"].append(s_poss)
        by_type_curves[vtype]["impossible"].append(s_imposs)
        by_type_curves[vtype]["t_star"] = t_star

    by_type: dict[str, Any] = {}
    for vtype, blob in by_type_curves.items():
        t_star = int(blob["t_star"])
        T = len(blob["possible"][0])
        p_mean, p_std = _nanmean_stack(blob["possible"])
        i_mean, i_std = _nanmean_stack(blob["impossible"])
        # Aggregate spike from mean curves
        sp = spike_score(p_mean, i_mean, t_star, spike_window)
        pre = pre_tstar_gap(p_mean, i_mean, t_star)
        by_type[vtype] = {
            "t_star": t_star,
            "t": list(range(T)),
            "possible_mean": p_mean,
            "impossible_mean": i_mean,
            "possible_std": p_std,
            "impossible_std": i_std,
            "spike_score": sp,
            "pre_tstar_abs_gap": pre,
            "n_pairs": len(blob["possible"]),
        }

    return {
        "model_type": model_type,
        "surprise_metric": "pixel_smooth_l1" if model_type == "pixel" else "latent_smooth_l1",
        "context_len": model.context_len,
        "pred_horizon": model.pred_horizon,
        "spike_window": spike_window,
        "by_type": by_type,
        "pairs": pairs_out,
    }
