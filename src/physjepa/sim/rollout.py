"""Rollout loop: simulate, render, record trajectory metadata."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable

import numpy as np

from .config import SimConfig
from .renderer import Renderer
from .world import PhysicsWorld, SceneSpec


InterventionFn = Callable[[PhysicsWorld, int], None]


@dataclass
class RolloutResult:
    frames: np.ndarray  # (T, H, W, 3) uint8
    trajectory: list[dict[str, Any]]
    objects: list[dict[str, Any]]
    occluders: list[dict[str, Any]]
    T: int
    resolution: tuple[int, int]
    fps: int
    seed: int | None = None
    extras: dict[str, Any] = field(default_factory=dict)


def run_rollout(
    config: SimConfig,
    scene: SceneSpec,
    *,
    T: int | None = None,
    intervention: InterventionFn | None = None,
    world: PhysicsWorld | None = None,
    renderer: Renderer | None = None,
) -> RolloutResult:
    """
    Run a fixed-horizon rollout.

    `intervention(world, t)` is called at the start of frame t (before physics steps
    for that frame), enabling VoE violations at t_star.
    """
    T = config.episode.T if T is None else T
    world = world or PhysicsWorld(config)
    renderer = renderer or Renderer(config)
    world.reset(scene)

    frames: list[np.ndarray] = []
    trajectory: list[dict[str, Any]] = []
    steps = config.render.steps_per_frame

    for t in range(T):
        if intervention is not None:
            intervention(world, t)

        # Record + render after intervention, before stepping (frame 0 = initial)
        states = world.snapshot()
        trajectory.append(
            {
                "t": t,
                "bodies": [
                    {
                        "id": s.id,
                        "x": s.x,
                        "y": s.y,
                        "vx": s.vx,
                        "vy": s.vy,
                        "angle": s.angle,
                        "visible": s.visible,
                        "mass": s.mass,
                    }
                    for s in states
                ],
            }
        )
        frames.append(renderer.render(world))

        if t < T - 1:
            world.step(n=steps)

    stacked = np.stack(frames, axis=0)
    return RolloutResult(
        frames=stacked,
        trajectory=trajectory,
        objects=world.objects_meta(),
        occluders=world.occluders_meta(),
        T=T,
        resolution=config.render.resolution,
        fps=config.render.fps,
        seed=scene.seed,
    )
