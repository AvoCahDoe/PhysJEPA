#!/usr/bin/env python3
"""Generate procedural training rollouts."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from physjepa.export import episode_name, write_episode, write_index
from physjepa.gen import generate_train_episode
from physjepa.sim.config import load_sim_config


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate train rollouts")
    parser.add_argument("--n", type=int, default=100, help="Number of episodes")
    parser.add_argument("--out", type=Path, default=ROOT / "data" / "train")
    parser.add_argument("--config", type=Path, default=ROOT / "configs" / "sim_default.yaml")
    parser.add_argument("--seed-start", type=int, default=0, help="First train seed (inclusive)")
    parser.add_argument(
        "--seed-end-exclusive-voe",
        type=int,
        default=9000,
        help="Train seeds must be < this (VoE uses >= 9000 by convention)",
    )
    args = parser.parse_args()

    config = load_sim_config(args.config)
    args.out.mkdir(parents=True, exist_ok=True)

    entries = []
    for i in range(args.n):
        seed = args.seed_start + i
        if seed >= args.seed_end_exclusive_voe:
            raise SystemExit(
                f"Train seed {seed} collides with VoE range (>= {args.seed_end_exclusive_voe})"
            )
        _scene, result = generate_train_episode(config, seed)
        name = episode_name(seed)
        ep_dir = args.out / name
        write_episode(
            ep_dir,
            result,
            split="train",
            seed=seed,
            violation=None,
            schema_version=config.schema_version,
        )
        entries.append(
            {
                "seed": seed,
                "path": name,
                "T": result.T,
                "n_objects": len(result.objects),
            }
        )
        if (i + 1) % 20 == 0 or i == 0:
            print(f"[{i + 1}/{args.n}] wrote {ep_dir}")

    write_index(args.out / "index.json", entries)
    print(f"Done. {len(entries)} episodes -> {args.out / 'index.json'}")


if __name__ == "__main__":
    main()
