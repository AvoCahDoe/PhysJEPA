"""Thin dataset wrappers for rollouts + JEPA window sampling."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image


def load_episode_meta(episode_dir: str | Path) -> dict[str, Any]:
    episode_dir = Path(episode_dir)
    with (episode_dir / "meta.json").open("r", encoding="utf-8") as f:
        return json.load(f)


def load_episode_frames(episode_dir: str | Path) -> np.ndarray:
    episode_dir = Path(episode_dir)
    meta = load_episode_meta(episode_dir)
    T = int(meta["T"])
    frames = []
    for t in range(T):
        path = episode_dir / "frames" / f"{t:06d}.png"
        frames.append(np.asarray(Image.open(path).convert("RGB"), dtype=np.uint8))
    return np.stack(frames, axis=0)


class RolloutDataset:
    """Index-based episode dataset. Returns dict with frames (T,H,W,3) and meta."""

    def __init__(
        self,
        index_path: str | Path | None = None,
        root: str | Path | None = None,
        *,
        entries: list[dict[str, Any]] | None = None,
        cache_frames: bool = False,
        seed_max_exclusive: int | None = 9000,
    ):
        if entries is not None:
            self.entries = list(entries)
            if root is None and index_path is not None:
                self.root = Path(index_path).parent
            else:
                self.root = Path(root) if root is not None else Path(".")
        else:
            if index_path is None:
                raise ValueError("index_path or entries required")
            index_path = Path(index_path)
            self.root = Path(root) if root is not None else index_path.parent
            with index_path.open("r", encoding="utf-8") as f:
                data = json.load(f)
            self.entries = data["episodes"]
            if seed_max_exclusive is not None:
                self.entries = [
                    e
                    for e in self.entries
                    if int(e.get("seed", 0)) < seed_max_exclusive
                ]
        self.cache_frames = cache_frames
        self._frame_cache: dict[int, np.ndarray] = {}

    def __len__(self) -> int:
        return len(self.entries)

    def episode_dir(self, idx: int) -> Path:
        entry = self.entries[idx]
        rel = entry.get("path") or entry.get("episode_dir")
        return self.root / rel if not Path(rel).is_absolute() else Path(rel)

    def __getitem__(self, idx: int) -> dict[str, Any]:
        episode_dir = self.episode_dir(idx)
        if self.cache_frames and idx in self._frame_cache:
            frames = self._frame_cache[idx]
        else:
            frames = load_episode_frames(episode_dir)
            if self.cache_frames:
                self._frame_cache[idx] = frames
        meta = load_episode_meta(episode_dir)
        return {"frames": frames, "meta": meta, "path": str(episode_dir)}

    def as_torch(self, idx: int):
        """Return float tensor CHW stack over time: (T, 3, H, W) in [0,1]."""
        import torch

        item = self[idx]
        frames = item["frames"].astype(np.float32) / 255.0
        tensor = torch.from_numpy(frames).permute(0, 3, 1, 2).contiguous()
        return {"frames": tensor, "meta": item["meta"], "path": item["path"]}


class JEPAWindowDataset:
    """
    Samples (context, future) windows from rollouts.

    Each item:
      context_frames: (context_len, 3, H, W)
      future_frames:  (pred_horizon, 3, H, W)
      meta extras: episode seed, start index
    """

    def __init__(
        self,
        rollout: RolloutDataset,
        *,
        context_len: int = 4,
        pred_horizon: int = 1,
        cache_frames: bool = True,
        windows_per_episode: int = 4,
        seed: int = 0,
    ):
        import torch

        self.torch = torch
        self.rollout = rollout
        self.context_len = context_len
        self.pred_horizon = pred_horizon
        self.windows_per_episode = windows_per_episode
        self.rng = np.random.default_rng(seed)

        if cache_frames:
            self.rollout.cache_frames = True

        # Precompute valid (episode_idx, start) list for deterministic length
        self.index: list[tuple[int, int]] = []
        need = context_len + pred_horizon
        for ep_i in range(len(rollout)):
            item = rollout[ep_i]
            T = item["frames"].shape[0]
            if T < need:
                continue
            max_start = T - need
            # Fixed stratified starts + random extras for coverage
            if max_start == 0:
                starts = [0]
            else:
                lin = np.linspace(0, max_start, num=min(windows_per_episode, max_start + 1), dtype=int)
                starts = sorted(set(int(s) for s in lin))
            for s in starts:
                self.index.append((ep_i, s))

    def __len__(self) -> int:
        return len(self.index)

    def __getitem__(self, idx: int) -> dict[str, Any]:
        ep_i, start = self.index[idx]
        item = self.rollout[ep_i]
        frames_np = item["frames"].astype(np.float32) / 255.0
        # (T,H,W,3) -> (T,3,H,W)
        frames = self.torch.from_numpy(frames_np).permute(0, 3, 1, 2).contiguous()
        end_ctx = start + self.context_len
        end_fut = end_ctx + self.pred_horizon
        context = frames[start:end_ctx]
        future = frames[end_ctx:end_fut]
        return {
            "context_frames": context,
            "future_frames": future,
            "seed": item["meta"].get("seed"),
            "start": start,
            "path": item["path"],
        }


def split_entries_by_seed(
    entries: list[dict[str, Any]],
    val_fraction: float = 0.1,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Deterministic split: highest seeds go to val."""
    sorted_entries = sorted(entries, key=lambda e: int(e.get("seed", 0)))
    n = len(sorted_entries)
    n_val = max(1, int(round(n * val_fraction))) if n > 1 else 0
    if n_val == 0:
        return sorted_entries, []
    return sorted_entries[:-n_val], sorted_entries[-n_val:]
