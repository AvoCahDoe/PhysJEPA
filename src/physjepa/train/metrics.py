"""React-ready metrics writers for training runs."""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any


@dataclass
class MetricsWriter:
    run_dir: Path
    run_id: str
    config: dict[str, Any] = field(default_factory=dict)
    schema_version: int = 1

    def __post_init__(self) -> None:
        self.run_dir = Path(self.run_dir)
        self.run_dir.mkdir(parents=True, exist_ok=True)
        self.metrics_path = self.run_dir / "metrics.jsonl"
        self.summary_path = self.run_dir / "summary.json"
        self._steps: list[int] = []
        self._loss: list[float] = []
        self._val_loss: list[float] = []
        self._latent_std: list[float] = []
        self._latent_norm: list[float] = []
        self.best_val_loss: float | None = None

    def log_step(
        self,
        step: int,
        *,
        loss: float,
        latent_std: float,
        latent_norm: float,
        val_loss: float | None = None,
        extra: dict[str, Any] | None = None,
    ) -> None:
        row: dict[str, Any] = {
            "step": step,
            "loss": float(loss),
            "latent_std": float(latent_std),
            "latent_norm": float(latent_norm),
        }
        if val_loss is not None:
            row["val_loss"] = float(val_loss)
            self._val_loss.append(float(val_loss))
            if self.best_val_loss is None or val_loss < self.best_val_loss:
                self.best_val_loss = float(val_loss)
        if extra:
            row.update(extra)

        self._steps.append(step)
        self._loss.append(float(loss))
        self._latent_std.append(float(latent_std))
        self._latent_norm.append(float(latent_norm))

        with self.metrics_path.open("a", encoding="utf-8") as f:
            f.write(json.dumps(row) + "\n")

    def write_summary(self, extra: dict[str, Any] | None = None) -> Path:
        summary: dict[str, Any] = {
            "schema_version": self.schema_version,
            "run_id": self.run_id,
            "curves": {
                "step": self._steps,
                "loss": self._loss,
                "latent_std": self._latent_std,
                "latent_norm": self._latent_norm,
            },
            "best_val_loss": self.best_val_loss,
            "config": self.config,
        }
        if self._val_loss:
            # Align val points to logged steps that included val (approximate: last N)
            summary["curves"]["val_loss"] = self._val_loss
        if extra:
            summary.update(extra)
        with self.summary_path.open("w", encoding="utf-8") as f:
            json.dump(summary, f, indent=2)
        return self.summary_path
