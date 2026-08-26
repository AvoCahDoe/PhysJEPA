"""Pymunk physics world: bodies, walls, occluders."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import pymunk

from .config import SimConfig


@dataclass
class BodySpec:
    id: int
    shape: str  # "disk" | "box"
    mass: float
    x: float
    y: float
    vx: float = 0.0
    vy: float = 0.0
    radius: float = 0.06
    width: float = 0.08
    height: float = 0.08
    color: tuple[int, int, int] = (220, 80, 70)
    angle: float = 0.0


@dataclass
class OccluderSpec:
    x: float
    y: float
    width: float
    height: float
    color: tuple[int, int, int] = (55, 60, 70)
    # If True, occluder participates in collisions; False = visual-only (default)
    solid: bool = False


@dataclass
class SceneSpec:
    bodies: list[BodySpec]
    occluders: list[OccluderSpec] = field(default_factory=list)
    seed: int | None = None
    # Optional wall gap for pass-through probes (mid-right wall segment disabled)
    disable_right_wall: bool = False
    # Optional gravity override for specialized probes (None = use config)
    gravity: tuple[float, float] | None = None


@dataclass
class BodyState:
    id: int
    x: float
    y: float
    vx: float
    vy: float
    angle: float
    visible: bool
    mass: float


class PhysicsWorld:
    """Thin wrapper around pymunk Space for our arena."""

    COLLISION_DYNAMIC = 1
    COLLISION_WALL = 2
    COLLISION_GHOST = 4  # pass-through bodies

    def __init__(self, config: SimConfig):
        self.config = config
        self.space = pymunk.Space()
        self.space.gravity = config.world.gravity
        self.space.damping = config.world.damping
        self._bodies: dict[int, pymunk.Body] = {}
        self._shapes: dict[int, pymunk.Shape] = {}
        self._body_specs: dict[int, BodySpec] = {}
        self.occluders: list[OccluderSpec] = []
        self._wall_shapes: list[pymunk.Shape] = []
        self._build_walls(disable_right=False)

    def reset(self, scene: SceneSpec) -> None:
        # Clear dynamic shapes/bodies
        for shape in list(self.space.shapes):
            if shape in self._wall_shapes:
                continue
            self.space.remove(shape)
        for body in list(self.space.bodies):
            if body.body_type == pymunk.Body.STATIC:
                continue
            self.space.remove(body)

        self._bodies.clear()
        self._shapes.clear()
        self._body_specs.clear()
        self.occluders = list(scene.occluders)

        if scene.gravity is not None:
            self.space.gravity = scene.gravity
        else:
            self.space.gravity = self.config.world.gravity

        # Rebuild walls if needed
        for w in self._wall_shapes:
            if w in self.space.shapes:
                self.space.remove(w)
        self._wall_shapes.clear()
        self._build_walls(disable_right=scene.disable_right_wall)

        for spec in scene.bodies:
            self._add_body(spec)

        # Solid occluders (rare; visual-only by default)
        for occ in self.occluders:
            if occ.solid:
                self._add_solid_occluder(occ)

    def _build_walls(self, disable_right: bool = False) -> None:
        w = self.config.world
        static = self.space.static_body
        t = w.wall_thickness
        segs = [
            ((0, 0), (w.width, 0)),  # floor
            ((0, w.height), (w.width, w.height)),  # ceiling
            ((0, 0), (0, w.height)),  # left
        ]
        if not disable_right:
            segs.append(((w.width, 0), (w.width, w.height)))

        for a, b in segs:
            seg = pymunk.Segment(static, a, b, t)
            seg.elasticity = w.elasticity
            seg.friction = w.friction
            seg.collision_type = self.COLLISION_WALL
            self.space.add(seg)
            self._wall_shapes.append(seg)

    def _add_body(self, spec: BodySpec) -> None:
        if spec.shape == "disk":
            moment = pymunk.moment_for_circle(spec.mass, 0, spec.radius)
            body = pymunk.Body(spec.mass, moment)
            body.position = (spec.x, spec.y)
            body.velocity = (spec.vx, spec.vy)
            body.angle = spec.angle
            shape = pymunk.Circle(body, spec.radius)
        elif spec.shape == "box":
            size = (spec.width, spec.height)
            moment = pymunk.moment_for_box(spec.mass, size)
            body = pymunk.Body(spec.mass, moment)
            body.position = (spec.x, spec.y)
            body.velocity = (spec.vx, spec.vy)
            body.angle = spec.angle
            shape = pymunk.Poly.create_box(body, size)
        else:
            raise ValueError(f"Unknown shape: {spec.shape}")

        shape.elasticity = self.config.world.elasticity
        shape.friction = self.config.world.friction
        shape.collision_type = self.COLLISION_DYNAMIC
        self.space.add(body, shape)
        self._bodies[spec.id] = body
        self._shapes[spec.id] = shape
        self._body_specs[spec.id] = spec

    def _add_solid_occluder(self, occ: OccluderSpec) -> None:
        body = pymunk.Body(body_type=pymunk.Body.STATIC)
        body.position = (occ.x, occ.y)
        shape = pymunk.Poly.create_box(body, (occ.width, occ.height))
        shape.elasticity = self.config.world.elasticity
        shape.friction = self.config.world.friction
        shape.collision_type = self.COLLISION_WALL
        self.space.add(body, shape)

    def step(self, dt: float | None = None, n: int = 1) -> None:
        dt = self.config.render.dt if dt is None else dt
        for _ in range(n):
            self.space.step(dt)

    def set_velocity(self, body_id: int, vx: float, vy: float) -> None:
        self._bodies[body_id].velocity = (vx, vy)

    def set_position(self, body_id: int, x: float, y: float) -> None:
        body = self._bodies[body_id]
        body.position = (x, y)
        # Clear leftover velocity impulse consistency
        body.velocity = body.velocity

    def set_mass(self, body_id: int, mass: float) -> None:
        body = self._bodies[body_id]
        body.mass = mass
        spec = self._body_specs[body_id]
        if spec.shape == "disk":
            body.moment = pymunk.moment_for_circle(mass, 0, spec.radius)
        else:
            body.moment = pymunk.moment_for_box(mass, (spec.width, spec.height))
        spec.mass = mass

    def disable_collisions(self, body_id: int) -> None:
        """Make a body pass through walls and other objects."""
        shape = self._shapes[body_id]
        shape.collision_type = self.COLLISION_GHOST
        shape.sensor = True

    def body_aabb(self, body_id: int) -> tuple[float, float, float, float]:
        """Return (minx, miny, maxx, maxy) in world coords."""
        shape = self._shapes[body_id]
        bb = shape.bb
        return (bb.left, bb.bottom, bb.right, bb.top)

    def is_visible(self, body_id: int) -> bool:
        """
        True unless the body center lies inside a visual occluder.

        Center-in-occluder matches object-permanence probes: the encoder cannot
        see the object, but physics + trajectory metadata continue.
        """
        if not self.occluders:
            return True
        body = self._bodies[body_id]
        cx, cy = float(body.position.x), float(body.position.y)
        for occ in self.occluders:
            ox0 = occ.x - occ.width / 2
            oy0 = occ.y - occ.height / 2
            ox1 = occ.x + occ.width / 2
            oy1 = occ.y + occ.height / 2
            if ox0 <= cx <= ox1 and oy0 <= cy <= oy1:
                return False
        return True

    def snapshot(self) -> list[BodyState]:
        states: list[BodyState] = []
        for bid, body in self._bodies.items():
            spec = self._body_specs[bid]
            states.append(
                BodyState(
                    id=bid,
                    x=float(body.position.x),
                    y=float(body.position.y),
                    vx=float(body.velocity.x),
                    vy=float(body.velocity.y),
                    angle=float(body.angle),
                    visible=self.is_visible(bid),
                    mass=float(spec.mass),
                )
            )
        states.sort(key=lambda s: s.id)
        return states

    def objects_meta(self) -> list[dict[str, Any]]:
        out = []
        for bid, spec in sorted(self._body_specs.items()):
            entry: dict[str, Any] = {
                "id": bid,
                "shape": spec.shape,
                "mass": spec.mass,
                "color": list(spec.color),
            }
            if spec.shape == "disk":
                entry["radius"] = spec.radius
            else:
                entry["width"] = spec.width
                entry["height"] = spec.height
            out.append(entry)
        return out

    def occluders_meta(self) -> list[dict[str, Any]]:
        return [
            {
                "x": o.x,
                "y": o.y,
                "width": o.width,
                "height": o.height,
                "color": list(o.color),
                "solid": o.solid,
            }
            for o in self.occluders
        ]
