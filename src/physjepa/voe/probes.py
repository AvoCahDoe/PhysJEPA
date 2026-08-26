"""VoE probe templates: matched possible / impossible pairs."""

from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable

import yaml

from physjepa.sim.config import SimConfig
from physjepa.sim.rollout import RolloutResult, run_rollout
from physjepa.sim.world import BodySpec, OccluderSpec, PhysicsWorld, SceneSpec


@dataclass
class ProbePair:
    pair_id: str
    violation_type: str
    t_star: int
    seed: int
    possible: RolloutResult
    impossible: RolloutResult
    description: str = ""
    occlusion_frames: int | None = None


def load_voe_config(path: str | Path) -> dict[str, Any]:
    with Path(path).open("r", encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


def _base_disk(
    config: SimConfig,
    *,
    x: float,
    y: float,
    vx: float,
    vy: float,
    body_id: int = 0,
    mass: float = 1.0,
    radius: float = 0.06,
) -> BodySpec:
    return BodySpec(
        id=body_id,
        shape="disk",
        mass=mass,
        x=x,
        y=y,
        vx=vx,
        vy=vy,
        radius=radius,
        color=config.colors.disks[body_id % len(config.colors.disks)],
    )


def scene_teleport_occlusion(
    config: SimConfig,
    seed: int,
    template: dict[str, Any] | None = None,
) -> SceneSpec:
    """Ball slides under a centered occluder (zero gravity); teleport while hidden.

    Template overrides (for occlusion-duration ablations):
      x0, vx, occluder_width, occluder_height, occluder_x, radius
    """
    tmpl = template or {}
    x0 = float(tmpl.get("x0", 0.12))
    vx = float(tmpl.get("vx", 0.95))
    radius = float(tmpl.get("radius", 0.05))
    occ_w = float(tmpl.get("occluder_width", 0.40))
    occ_h = float(tmpl.get("occluder_height", 0.32))
    occ_x = float(tmpl.get("occluder_x", 0.50))
    # Zero gravity keeps y fixed so the ball stays behind the occluder through t_star.
    ball = _base_disk(config, x=x0, y=0.50, vx=vx, vy=0.0, radius=radius)
    occ = OccluderSpec(
        x=occ_x,
        y=0.50,
        width=occ_w,
        height=occ_h,
        color=config.colors.occluder,
        solid=False,
    )
    return SceneSpec(
        bodies=[ball],
        occluders=[occ],
        seed=seed,
        gravity=(0.0, 0.0),
    )


def scene_pass_through_wall(config: SimConfig, seed: int) -> SceneSpec:
    """Ball aimed at right wall."""
    ball = _base_disk(config, x=0.55, y=0.45, vx=1.6, vy=0.1)
    return SceneSpec(bodies=[ball], occluders=[], seed=seed, disable_right_wall=False)


def scene_stop_without_collision(config: SimConfig, seed: int) -> SceneSpec:
    """Ball in free flight (reduced gravity feel via high initial vy)."""
    ball = _base_disk(config, x=0.25, y=0.7, vx=0.7, vy=0.4)
    return SceneSpec(bodies=[ball], occluders=[], seed=seed)


def scene_impossible_bounce(config: SimConfig, seed: int) -> SceneSpec:
    """Two balls approaching; violation flips velocity/mass without contact."""
    a = _base_disk(config, x=0.25, y=0.5, vx=0.85, vy=0.0, body_id=0, mass=1.0)
    b = _base_disk(
        config,
        x=0.75,
        y=0.5,
        vx=-0.85,
        vy=0.0,
        body_id=1,
        mass=1.0,
        radius=0.055,
    )
    b.color = config.colors.disks[1 % len(config.colors.disks)]
    return SceneSpec(bodies=[a, b], occluders=[], seed=seed)


def _wrap_scene(
    fn: Callable[..., SceneSpec],
) -> Callable[[SimConfig, int, dict[str, Any] | None], SceneSpec]:
    def builder(config: SimConfig, seed: int, template: dict[str, Any] | None = None) -> SceneSpec:
        # Teleport scene accepts template; others ignore it.
        try:
            return fn(config, seed, template)
        except TypeError:
            return fn(config, seed)

    return builder


SCENE_BUILDERS: dict[str, Callable[[SimConfig, int, dict[str, Any] | None], SceneSpec]] = {
    "teleport_occlusion": _wrap_scene(scene_teleport_occlusion),
    "pass_through_wall": _wrap_scene(scene_pass_through_wall),
    "stop_without_collision": _wrap_scene(scene_stop_without_collision),
    "impossible_bounce": _wrap_scene(scene_impossible_bounce),
}


def _make_intervention(
    vtype: str,
    t_star: int,
    template: dict[str, Any],
    *,
    apply: bool,
) -> Callable[[PhysicsWorld, int], None] | None:
    if not apply:
        return None

    fired = {"done": False}

    def intervention(world: PhysicsWorld, t: int) -> None:
        if fired["done"] or t != t_star:
            return
        fired["done"] = True
        if vtype == "teleport_occlusion":
            dx, dy = template.get("teleport_delta", [0.35, 0.0])
            states = world.snapshot()
            s0 = states[0]
            world.set_position(s0.id, s0.x + float(dx), s0.y + float(dy))
        elif vtype == "pass_through_wall":
            world.disable_collisions(0)
        elif vtype == "stop_without_collision":
            world.set_velocity(0, 0.0, 0.0)
        elif vtype == "impossible_bounce":
            states = world.snapshot()
            if len(states) >= 2:
                a, b = states[0], states[1]
                scale = float(template.get("velocity_scale", -1.4))
                mass_scale = float(template.get("mass_scale", 2.0))
                world.set_velocity(a.id, b.vx * scale, b.vy * scale)
                world.set_velocity(b.id, a.vx * scale, a.vy * scale)
                world.set_mass(a.id, a.mass * mass_scale)
            else:
                s0 = states[0]
                world.set_velocity(s0.id, -s0.vx * 1.5, -s0.vy * 1.5)
        else:
            raise ValueError(f"Unknown violation type: {vtype}")

    return intervention


def _count_occluded_frames(result: RolloutResult, body_id: int = 0) -> int:
    """Count frames where body center is marked not visible."""
    n = 0
    for fr in result.trajectory:
        for b in fr.get("bodies", []):
            if int(b.get("id", -1)) == body_id and not bool(b.get("visible", True)):
                n += 1
                break
    return n


def generate_probe_pair(
    config: SimConfig,
    template: dict[str, Any],
    seed: int,
) -> ProbePair:
    vtype = template["type"]
    t_star = int(template["t_star"])
    pair_id = f"{template['id']}_s{seed}"
    builder = SCENE_BUILDERS[vtype]
    scene = builder(config, seed, template)

    possible = run_rollout(
        config,
        deepcopy(scene),
        intervention=_make_intervention(vtype, t_star, template, apply=False),
    )
    impossible = run_rollout(
        config,
        deepcopy(scene),
        intervention=_make_intervention(vtype, t_star, template, apply=True),
    )
    # Stash occlusion length for ablation metadata
    occ_frames = _count_occluded_frames(possible)
    desc = str(template.get("description", ""))
    if "occlusion_frames_target" in template:
        desc = f"{desc} [target_occ={template['occlusion_frames_target']}]".strip()
    return ProbePair(
        pair_id=pair_id,
        violation_type=vtype,
        t_star=t_star,
        seed=seed,
        possible=possible,
        impossible=impossible,
        description=desc,
        occlusion_frames=occ_frames if occ_frames > 0 else None,
    )


def generate_all_probes(
    config: SimConfig,
    voe_cfg: dict[str, Any],
) -> list[ProbePair]:
    seeds = list(voe_cfg.get("seeds", [9001]))
    templates = list(voe_cfg.get("templates", []))
    pairs: list[ProbePair] = []
    for tmpl in templates:
        for seed in seeds:
            pairs.append(generate_probe_pair(config, tmpl, int(seed)))
    return pairs
