#!/usr/bin/env python3
"""Generate held-out VoE possible/impossible probe pairs."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from physjepa.export import write_episode, write_voe_index
from physjepa.sim.config import load_sim_config
from physjepa.voe import generate_all_probes, load_voe_config


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate VoE probe pairs")
    parser.add_argument("--out", type=Path, default=ROOT / "data" / "voe")
    parser.add_argument("--config", type=Path, default=ROOT / "configs" / "sim_default.yaml")
    parser.add_argument("--voe-config", type=Path, default=ROOT / "configs" / "voe_probes.yaml")
    args = parser.parse_args()

    config = load_sim_config(args.config)
    voe_cfg = load_voe_config(args.voe_config)

    # Apply VoE sim overrides
    sim_over = voe_cfg.get("sim", {})
    if "T" in sim_over:
        config.episode.T = int(sim_over["T"])
    if "fps" in sim_over:
        config.render.fps = int(sim_over["fps"])
    if "resolution" in sim_over:
        r = sim_over["resolution"]
        config.render.resolution = (int(r[0]), int(r[1]))

    args.out.mkdir(parents=True, exist_ok=True)
    pairs = generate_all_probes(config, voe_cfg)
    index_entries = []

    for pair in pairs:
        pair_dir = args.out / pair.pair_id
        poss_dir = pair_dir / "possible"
        imposs_dir = pair_dir / "impossible"

        violation_base = {
            "type": pair.violation_type,
            "t_star": pair.t_star,
            "pair_id": pair.pair_id,
        }

        write_episode(
            poss_dir,
            pair.possible,
            split="voe",
            seed=pair.seed,
            violation=None,
            schema_version=config.schema_version,
            extra_meta={"pair_role": "possible", "pair_id": pair.pair_id},
        )
        write_episode(
            imposs_dir,
            pair.impossible,
            split="voe",
            seed=pair.seed,
            violation=violation_base,
            schema_version=config.schema_version,
            extra_meta={"pair_role": "impossible", "pair_id": pair.pair_id},
        )

        pair_meta = {
            "schema_version": 1,
            "pair_id": pair.pair_id,
            "violation_type": pair.violation_type,
            "t_star": pair.t_star,
            "matched_seed": pair.seed,
            "description": pair.description,
            "occlusion_frames": pair.occlusion_frames,
            "possible": "possible",
            "impossible": "impossible",
        }
        with (pair_dir / "pair_meta.json").open("w", encoding="utf-8") as f:
            json.dump(pair_meta, f, indent=2)

        entry = {
            "pair_id": pair.pair_id,
            "violation_type": pair.violation_type,
            "t_star": pair.t_star,
            "matched_seed": pair.seed,
            "possible": f"{pair.pair_id}/possible",
            "impossible": f"{pair.pair_id}/impossible",
        }
        if pair.occlusion_frames is not None:
            entry["occlusion_frames"] = pair.occlusion_frames
        index_entries.append(entry)
        print(f"wrote {pair_dir}")

    write_voe_index(args.out / "index.json", index_entries)
    print(f"Done. {len(index_entries)} pairs -> {args.out / 'index.json'}")


if __name__ == "__main__":
    main()
