"""Procedural training scene sampler."""

from __future__ import annotations

import math

import numpy as np

from physjepa.sim.config import SimConfig
from physjepa.sim.rollout import run_rollout, RolloutResult
from physjepa.sim.world import BodySpec, OccluderSpec, SceneSpec


def _rand_color(rng: np.random.Generator, palette: list[tuple[int, int, int]], i: int):
    return palette[i % len(palette)]


def sample_scene(config: SimConfig, seed: int) -> SceneSpec:
    rng = np.random.default_rng(seed)
    ep = config.episode
    w, h = config.world.width, config.world.height
    margin = 0.12

    n = int(rng.integers(ep.min_objects, ep.max_objects + 1))
    bodies: list[BodySpec] = []

    for i in range(n):
        shape = str(rng.choice(ep.shapes))
        mass = float(rng.uniform(*ep.mass_range))
        speed = float(rng.uniform(*ep.speed_range))
        angle = float(rng.uniform(0, 2 * math.pi))
        vx = speed * math.cos(angle)
        vy = speed * math.sin(angle)
        x = float(rng.uniform(margin, w - margin))
        y = float(rng.uniform(margin, h - margin))

        if shape == "disk":
            radius = float(rng.uniform(*ep.disk_radius_range))
            color = _rand_color(rng, config.colors.disks, i)
            bodies.append(
                BodySpec(
                    id=i,
                    shape="disk",
                    mass=mass,
                    x=x,
                    y=y,
                    vx=vx,
                    vy=vy,
                    radius=radius,
                    color=color,
                )
            )
        else:
            size = float(rng.uniform(*ep.box_size_range))
            color = _rand_color(rng, config.colors.boxes, i)
            bodies.append(
                BodySpec(
                    id=i,
                    shape="box",
                    mass=mass,
                    x=x,
                    y=y,
                    vx=vx,
                    vy=vy,
                    width=size,
                    height=size,
                    color=color,
                    angle=float(rng.uniform(0, math.pi / 2)),
                )
            )

    occluders: list[OccluderSpec] = []
    if rng.random() < ep.occluder_prob:
        ow = float(rng.uniform(*ep.occluder_width_range))
        oh = float(rng.uniform(*ep.occluder_height_range))
        ox = float(rng.uniform(ow / 2 + 0.05, w - ow / 2 - 0.05))
        oy = float(rng.uniform(oh / 2 + 0.05, h - oh / 2 - 0.05))
        occluders.append(
            OccluderSpec(
                x=ox,
                y=oy,
                width=ow,
                height=oh,
                color=config.colors.occluder,
                solid=False,
            )
        )

    return SceneSpec(bodies=bodies, occluders=occluders, seed=seed)


def _total_displacement(result: RolloutResult) -> float:
    if result.T < 2 or not result.trajectory:
        return 0.0
    first = {b["id"]: b for b in result.trajectory[0]["bodies"]}
    last = {b["id"]: b for b in result.trajectory[-1]["bodies"]}
    total = 0.0
    for bid, b0 in first.items():
        b1 = last.get(bid)
        if b1 is None:
            continue
        total += math.hypot(b1["x"] - b0["x"], b1["y"] - b0["y"])
    return total


def generate_train_episode(
    config: SimConfig,
    seed: int,
) -> tuple[SceneSpec, RolloutResult]:
    """Sample until motion threshold met or attempts exhausted."""
    ep = config.episode
    last_scene: SceneSpec | None = None
    last_result: RolloutResult | None = None

    for attempt in range(ep.max_resample_attempts):
        scene = sample_scene(config, seed + attempt * 100_003)
        scene.seed = seed
        result = run_rollout(config, scene)
        last_scene, last_result = scene, result
        if _total_displacement(result) >= ep.min_total_displacement:
            return scene, result

    assert last_scene is not None and last_result is not None
    return last_scene, last_result
