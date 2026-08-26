"""Orthographic 64x64 RGB renderer with supersampling."""

from __future__ import annotations

import math

import numpy as np
from PIL import Image, ImageDraw

from .config import SimConfig
from .world import BodySpec, OccluderSpec, PhysicsWorld


class Renderer:
    def __init__(self, config: SimConfig):
        self.config = config
        w, h = config.render.resolution
        ss = max(1, config.render.supersample)
        self.out_w, self.out_h = w, h
        self.ss = ss
        self.draw_w = w * ss
        self.draw_h = h * ss
        self.world_w = config.world.width
        self.world_h = config.world.height

    def _to_px(self, x: float, y: float) -> tuple[float, float]:
        # World origin bottom-left → image top-left
        px = (x / self.world_w) * self.draw_w
        py = ((self.world_h - y) / self.world_h) * self.draw_h
        return px, py

    def _scale(self, v: float) -> float:
        return (v / self.world_w) * self.draw_w

    def render(self, world: PhysicsWorld) -> np.ndarray:
        """Return HxWx3 uint8 RGB array at configured resolution."""
        bg = self.config.render.background
        img = Image.new("RGB", (self.draw_w, self.draw_h), bg)
        draw = ImageDraw.Draw(img)

        # Walls as thin border
        wall_c = self.config.colors.walls
        t = max(1, int(self._scale(self.config.world.wall_thickness)))
        draw.rectangle([0, 0, self.draw_w - 1, self.draw_h - 1], outline=wall_c, width=t)

        # Dynamic bodies (only if visible — occluded objects omitted from RGB)
        for state in world.snapshot():
            if not state.visible:
                continue
            spec = world._body_specs[state.id]
            self._draw_body(draw, spec, state.x, state.y, state.angle)

        # Occluders on top (opaque)
        for occ in world.occluders:
            self._draw_occluder(draw, occ)

        if self.ss > 1:
            img = img.resize((self.out_w, self.out_h), Image.Resampling.LANCZOS)
        else:
            img = img.resize((self.out_w, self.out_h), Image.Resampling.NEAREST)

        return np.asarray(img, dtype=np.uint8)

    def _draw_body(
        self,
        draw: ImageDraw.ImageDraw,
        spec: BodySpec,
        x: float,
        y: float,
        angle: float,
    ) -> None:
        color = spec.color
        if spec.shape == "disk":
            cx, cy = self._to_px(x, y)
            r = self._scale(spec.radius)
            draw.ellipse([cx - r, cy - r, cx + r, cy + r], fill=color)
        else:
            hw, hh = spec.width / 2, spec.height / 2
            corners = [(-hw, -hh), (hw, -hh), (hw, hh), (-hw, hh)]
            ca, sa = math.cos(angle), math.sin(angle)
            pts = []
            for lx, ly in corners:
                wx = x + lx * ca - ly * sa
                wy = y + lx * sa + ly * ca
                pts.append(self._to_px(wx, wy))
            draw.polygon(pts, fill=color)

    def _draw_occluder(self, draw: ImageDraw.ImageDraw, occ: OccluderSpec) -> None:
        x0, y1 = self._to_px(occ.x - occ.width / 2, occ.y + occ.height / 2)
        x1, y0 = self._to_px(occ.x + occ.width / 2, occ.y - occ.height / 2)
        # After flip: top-left is (x0,y1)-> wait: _to_px flips y so higher world y = smaller py
        # left-top in image: min px of left edge, min py of top edge
        left = min(x0, x1)
        right = max(x0, x1)
        top = min(y0, y1)
        bottom = max(y0, y1)
        draw.rectangle([left, top, right, bottom], fill=occ.color)
