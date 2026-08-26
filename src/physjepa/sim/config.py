"""Simulation configuration loading."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml


@dataclass
class WorldConfig:
    width: float = 1.0
    height: float = 1.0
    gravity: tuple[float, float] = (0.0, -9.8)
    damping: float = 0.98
    wall_thickness: float = 0.02
    elasticity: float = 0.85
    friction: float = 0.4


@dataclass
class RenderConfig:
    resolution: tuple[int, int] = (64, 64)
    supersample: int = 4
    background: tuple[int, int, int] = (18, 22, 28)
    fps: int = 30
    steps_per_frame: int = 2
    dt: float = 1.0 / 60.0


@dataclass
class EpisodeConfig:
    T: int = 24
    min_objects: int = 1
    max_objects: int = 3
    mass_range: tuple[float, float] = (0.5, 2.0)
    speed_range: tuple[float, float] = (0.3, 1.8)
    disk_radius_range: tuple[float, float] = (0.04, 0.08)
    box_size_range: tuple[float, float] = (0.06, 0.12)
    shapes: list[str] = field(default_factory=lambda: ["disk", "box"])
    occluder_prob: float = 0.7
    occluder_width_range: tuple[float, float] = (0.18, 0.35)
    occluder_height_range: tuple[float, float] = (0.12, 0.45)
    min_total_displacement: float = 0.08
    max_resample_attempts: int = 40


@dataclass
class ColorConfig:
    disks: list[tuple[int, int, int]] = field(
        default_factory=lambda: [(220, 80, 70), (70, 160, 220), (240, 190, 60)]
    )
    boxes: list[tuple[int, int, int]] = field(
        default_factory=lambda: [(90, 200, 120), (180, 100, 220), (240, 140, 80)]
    )
    occluder: tuple[int, int, int] = (55, 60, 70)
    walls: tuple[int, int, int] = (40, 44, 52)


@dataclass
class SimConfig:
    schema_version: int = 1
    world: WorldConfig = field(default_factory=WorldConfig)
    render: RenderConfig = field(default_factory=RenderConfig)
    episode: EpisodeConfig = field(default_factory=EpisodeConfig)
    colors: ColorConfig = field(default_factory=ColorConfig)
    raw: dict[str, Any] = field(default_factory=dict, repr=False)


def _as_tuple2(v: Any) -> tuple[float, float]:
    return (float(v[0]), float(v[1]))


def _as_tuple3i(v: Any) -> tuple[int, int, int]:
    return (int(v[0]), int(v[1]), int(v[2]))


def _as_tuple2i(v: Any) -> tuple[int, int]:
    return (int(v[0]), int(v[1]))


def load_sim_config(path: str | Path | None = None) -> SimConfig:
    """Load sim config from YAML. Defaults to configs/sim_default.yaml relative to repo root."""
    if path is None:
        path = Path(__file__).resolve().parents[3] / "configs" / "sim_default.yaml"
    else:
        path = Path(path)

    with path.open("r", encoding="utf-8") as f:
        raw = yaml.safe_load(f) or {}

    w = raw.get("world", {})
    r = raw.get("render", {})
    e = raw.get("episode", {})
    c = raw.get("colors", {})

    world = WorldConfig(
        width=float(w.get("width", 1.0)),
        height=float(w.get("height", 1.0)),
        gravity=_as_tuple2(w.get("gravity", [0.0, -9.8])),
        damping=float(w.get("damping", 0.98)),
        wall_thickness=float(w.get("wall_thickness", 0.02)),
        elasticity=float(w.get("elasticity", 0.85)),
        friction=float(w.get("friction", 0.4)),
    )
    render = RenderConfig(
        resolution=_as_tuple2i(r.get("resolution", [64, 64])),
        supersample=int(r.get("supersample", 4)),
        background=_as_tuple3i(r.get("background", [18, 22, 28])),
        fps=int(r.get("fps", 30)),
        steps_per_frame=int(r.get("steps_per_frame", 2)),
        dt=float(r.get("dt", 1.0 / 60.0)),
    )
    episode = EpisodeConfig(
        T=int(e.get("T", 24)),
        min_objects=int(e.get("min_objects", 1)),
        max_objects=int(e.get("max_objects", 3)),
        mass_range=_as_tuple2(e.get("mass_range", [0.5, 2.0])),
        speed_range=_as_tuple2(e.get("speed_range", [0.3, 1.8])),
        disk_radius_range=_as_tuple2(e.get("disk_radius_range", [0.04, 0.08])),
        box_size_range=_as_tuple2(e.get("box_size_range", [0.06, 0.12])),
        shapes=list(e.get("shapes", ["disk", "box"])),
        occluder_prob=float(e.get("occluder_prob", 0.7)),
        occluder_width_range=_as_tuple2(e.get("occluder_width_range", [0.18, 0.35])),
        occluder_height_range=_as_tuple2(e.get("occluder_height_range", [0.12, 0.45])),
        min_total_displacement=float(e.get("min_total_displacement", 0.08)),
        max_resample_attempts=int(e.get("max_resample_attempts", 40)),
    )
    colors = ColorConfig(
        disks=[_as_tuple3i(x) for x in c.get("disks", [[220, 80, 70], [70, 160, 220], [240, 190, 60]])],
        boxes=[_as_tuple3i(x) for x in c.get("boxes", [[90, 200, 120], [180, 100, 220], [240, 140, 80]])],
        occluder=_as_tuple3i(c.get("occluder", [55, 60, 70])),
        walls=_as_tuple3i(c.get("walls", [40, 44, 52])),
    )
    return SimConfig(
        schema_version=int(raw.get("schema_version", 1)),
        world=world,
        render=render,
        episode=episode,
        colors=colors,
        raw=raw,
    )
