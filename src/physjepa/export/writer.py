"""Episode export writers (React-ready JSON + PNG frames)."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from PIL import Image

from physjepa.sim.rollout import RolloutResult


def write_episode(
    out_dir: str | Path,
    result: RolloutResult,
    *,
    split: str,
    seed: int | None = None,
    violation: dict[str, Any] | None = None,
    schema_version: int = 1,
    extra_meta: dict[str, Any] | None = None,
) -> Path:
    """
    Write episode_dir/
      frames/000000.png ...
      meta.json
    """
    out_dir = Path(out_dir)
    frames_dir = out_dir / "frames"
    frames_dir.mkdir(parents=True, exist_ok=True)

    for t in range(result.T):
        path = frames_dir / f"{t:06d}.png"
        Image.fromarray(result.frames[t]).save(path)

    meta: dict[str, Any] = {
        "schema_version": schema_version,
        "split": split,
        "seed": seed if seed is not None else result.seed,
        "T": result.T,
        "resolution": list(result.resolution),
        "fps": result.fps,
        "violation": violation,
        "objects": result.objects,
        "occluders": result.occluders,
        "trajectory": result.trajectory,
    }
    if extra_meta:
        meta.update(extra_meta)

    meta_path = out_dir / "meta.json"
    with meta_path.open("w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2)
    return out_dir


def write_index(path: str | Path, entries: list[dict[str, Any]]) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump({"schema_version": 1, "episodes": entries}, f, indent=2)


def write_voe_index(path: str | Path, pairs: list[dict[str, Any]]) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump({"schema_version": 1, "pairs": pairs}, f, indent=2)


def episode_name(seed: int, prefix: str = "episode") -> str:
    return f"{prefix}_{seed:06d}"
