"""Batch collation for JEPA windows."""

from __future__ import annotations

from typing import Any

import torch


def jepa_collate(batch: list[dict[str, Any]]) -> dict[str, Any]:
    return {
        "context_frames": torch.stack([b["context_frames"] for b in batch], dim=0),
        "future_frames": torch.stack([b["future_frames"] for b in batch], dim=0),
        "seed": [b["seed"] for b in batch],
        "start": torch.tensor([b["start"] for b in batch], dtype=torch.long),
        "path": [b["path"] for b in batch],
    }
