"""Frozen-encoder linear probes for physics variables (body 0)."""

from __future__ import annotations

from typing import Any

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset

from physjepa.data import RolloutDataset, split_entries_by_seed


def _body0(frame_traj: dict[str, Any], body_id: int = 0) -> dict[str, Any] | None:
    for b in frame_traj.get("bodies", []):
        if int(b["id"]) == body_id:
            return b
    return None


@torch.no_grad()
def extract_latents_and_targets(
    model: torch.nn.Module,
    rollout: RolloutDataset,
    *,
    body_id: int = 0,
    device: torch.device,
    batch_encode: int = 64,
) -> dict[str, np.ndarray]:
    """
    Returns arrays:
      z: (N, D)
      xy, vxvy, mass, visible
    Skip frames without body_id.
    """
    model.eval()
    zs: list[np.ndarray] = []
    xys: list[list[float]] = []
    vxs: list[list[float]] = []
    masses: list[float] = []
    visibles: list[float] = []

    frame_buf: list[torch.Tensor] = []
    target_buf: list[dict[str, Any]] = []

    def _flush() -> None:
        nonlocal frame_buf, target_buf
        if not frame_buf:
            return
        batch = torch.stack(frame_buf, dim=0).to(device)
        z = model.encoder(batch).cpu().numpy()
        zs.append(z)
        for t in target_buf:
            xys.append([t["x"], t["y"]])
            vxs.append([t["vx"], t["vy"]])
            masses.append(float(t["mass"]))
            visibles.append(1.0 if t["visible"] else 0.0)
        frame_buf = []
        target_buf = []

    for i in range(len(rollout)):
        item = rollout[i]
        frames = item["frames"].astype(np.float32) / 255.0
        # (T,H,W,3) -> frames as CHW
        meta = item["meta"]
        traj = meta["trajectory"]
        for t, fr in enumerate(traj):
            b0 = _body0(fr, body_id)
            if b0 is None:
                continue
            chw = torch.from_numpy(frames[t]).permute(2, 0, 1).contiguous()
            frame_buf.append(chw)
            target_buf.append(b0)
            if len(frame_buf) >= batch_encode:
                _flush()
    _flush()

    if not zs:
        raise RuntimeError("No probe samples extracted (missing body_id?)")

    return {
        "z": np.concatenate(zs, axis=0).astype(np.float32),
        "xy": np.asarray(xys, dtype=np.float32),
        "vxvy": np.asarray(vxs, dtype=np.float32),
        "mass": np.asarray(masses, dtype=np.float32).reshape(-1, 1),
        "visible": np.asarray(visibles, dtype=np.float32).reshape(-1, 1),
    }


def _r2_score(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    y_true = y_true.reshape(len(y_true), -1)
    y_pred = y_pred.reshape(len(y_pred), -1)
    ss_res = float(np.sum((y_true - y_pred) ** 2))
    ss_tot = float(np.sum((y_true - y_true.mean(axis=0, keepdims=True)) ** 2))
    if ss_tot < 1e-12:
        return 0.0
    return 1.0 - ss_res / ss_tot


def _train_linear(
    z_train: np.ndarray,
    y_train: np.ndarray,
    z_val: np.ndarray,
    y_val: np.ndarray,
    *,
    task: str,
    epochs: int,
    lr: float,
    batch_size: int,
    device: torch.device,
) -> dict[str, float]:
    in_dim = z_train.shape[1]
    out_dim = y_train.shape[1]
    probe = nn.Linear(in_dim, out_dim).to(device)
    opt = torch.optim.Adam(probe.parameters(), lr=lr)

    xt = torch.from_numpy(z_train)
    yt = torch.from_numpy(y_train)
    loader = DataLoader(
        TensorDataset(xt, yt),
        batch_size=min(batch_size, len(xt)),
        shuffle=True,
    )

    probe.train()
    for _ in range(epochs):
        for xb, yb in loader:
            xb = xb.to(device)
            yb = yb.to(device)
            pred = probe(xb)
            if task == "classification":
                loss = F.binary_cross_entropy_with_logits(pred, yb)
            else:
                loss = F.mse_loss(pred, yb)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()

    probe.eval()
    with torch.no_grad():
        tr_pred = probe(torch.from_numpy(z_train).to(device)).cpu().numpy()
        va_pred = probe(torch.from_numpy(z_val).to(device)).cpu().numpy()

    if task == "classification":
        tr_prob = 1 / (1 + np.exp(-tr_pred))
        va_prob = 1 / (1 + np.exp(-va_pred))
        tr_acc = float(((tr_prob >= 0.5) == (y_train >= 0.5)).mean())
        va_acc = float(((va_prob >= 0.5) == (y_val >= 0.5)).mean())
        return {
            "train_acc": tr_acc,
            "val_acc": va_acc,
            "train_bce": float(
                F.binary_cross_entropy_with_logits(
                    torch.from_numpy(tr_pred), torch.from_numpy(y_train)
                ).item()
            ),
            "val_bce": float(
                F.binary_cross_entropy_with_logits(
                    torch.from_numpy(va_pred), torch.from_numpy(y_val)
                ).item()
            ),
        }

    tr_mse = float(np.mean((tr_pred - y_train) ** 2))
    va_mse = float(np.mean((va_pred - y_val) ** 2))
    return {
        "train_mse": tr_mse,
        "val_mse": va_mse,
        "train_r2": _r2_score(y_train, tr_pred),
        "val_r2": _r2_score(y_val, va_pred),
    }


def run_linear_probes(
    model: torch.nn.Module,
    train_index: str | Any,
    *,
    val_fraction: float = 0.1,
    seed_max_exclusive: int = 9000,
    body_id: int = 0,
    targets: list[str] | None = None,
    epochs: int = 30,
    lr: float = 1e-3,
    batch_size: int = 64,
    device: torch.device | None = None,
) -> dict[str, Any]:
    from pathlib import Path

    device = device or torch.device("cpu")
    targets = targets or ["xy", "vxvy", "mass", "visible"]
    train_index = Path(train_index)

    base = RolloutDataset(
        train_index,
        seed_max_exclusive=seed_max_exclusive,
        cache_frames=True,
    )
    train_entries, val_entries = split_entries_by_seed(base.entries, val_fraction)
    train_rollout = RolloutDataset(root=base.root, entries=train_entries, cache_frames=True)
    val_rollout = RolloutDataset(root=base.root, entries=val_entries, cache_frames=True)

    train_data = extract_latents_and_targets(
        model, train_rollout, body_id=body_id, device=device
    )
    val_data = extract_latents_and_targets(
        model, val_rollout, body_id=body_id, device=device
    )

    results: dict[str, Any] = {}
    task_map = {
        "xy": "regression",
        "vxvy": "regression",
        "mass": "regression",
        "visible": "classification",
    }
    for name in targets:
        if name not in task_map:
            raise ValueError(f"Unknown probe target: {name}")
        results[name] = _train_linear(
            train_data["z"],
            train_data[name],
            val_data["z"],
            val_data[name],
            task=task_map[name],
            epochs=epochs,
            lr=lr,
            batch_size=batch_size,
            device=device,
        )
        results[name]["n_train"] = int(train_data["z"].shape[0])
        results[name]["n_val"] = int(val_data["z"].shape[0])

    return {
        "targets": results,
        "n_train_episodes": len(train_entries),
        "n_val_episodes": len(val_entries),
        "body_id": body_id,
        "latent_dim": int(train_data["z"].shape[1]),
    }
