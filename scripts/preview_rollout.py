#!/usr/bin/env python3
"""Preview an episode as a GIF (and optional matplotlib scrub)."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import imageio.v2 as imageio
import numpy as np
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))


def main() -> None:
    parser = argparse.ArgumentParser(description="Preview rollout episode")
    parser.add_argument("episode", type=Path, help="Path to episode directory")
    parser.add_argument("--gif", type=Path, default=None, help="Output GIF path")
    parser.add_argument("--show", action="store_true", help="Show matplotlib animation")
    args = parser.parse_args()

    ep = args.episode
    with (ep / "meta.json").open("r", encoding="utf-8") as f:
        meta = json.load(f)

    T = int(meta["T"])
    fps = int(meta.get("fps", 30))
    frames = []
    for t in range(T):
        frames.append(np.asarray(Image.open(ep / "frames" / f"{t:06d}.png")))

    gif_path = args.gif or (ep / "preview.gif")
    imageio.mimsave(gif_path, frames, fps=min(fps, 15))
    print(f"Wrote {gif_path} ({T} frames, seed={meta.get('seed')})")

    if meta.get("violation"):
        print(f"violation: {meta['violation']}")

    if args.show:
        import matplotlib.pyplot as plt
        from matplotlib import animation

        fig, ax = plt.subplots()
        im = ax.imshow(frames[0])
        ax.set_axis_off()
        title = ax.set_title("t=0")

        def update(i):
            im.set_data(frames[i])
            title.set_text(f"t={i}")
            return [im, title]

        animation.FuncAnimation(fig, update, frames=T, interval=1000 / max(fps, 1), blit=False)
        plt.show()


if __name__ == "__main__":
    main()
