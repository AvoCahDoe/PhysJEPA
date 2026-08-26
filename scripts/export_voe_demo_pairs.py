#!/usr/bin/env python3
"""Export one VoE showcase pair per violation type for dashboard replay."""

from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def main() -> None:
    parser = argparse.ArgumentParser(description="Export VoE demo pairs for dashboard")
    parser.add_argument(
        "--voe-index",
        type=Path,
        default=ROOT / "data" / "voe" / "index.json",
    )
    parser.add_argument(
        "--voe-root",
        type=Path,
        default=ROOT / "data" / "voe",
    )
    parser.add_argument(
        "--out",
        type=Path,
        default=ROOT / "viz" / "fixtures" / "voe_demo",
    )
    parser.add_argument("--seed-suffix", type=str, default="9001")
    args = parser.parse_args()

    idx_path = args.voe_index if args.voe_index.is_absolute() else ROOT / args.voe_index
    voe_root = args.voe_root if args.voe_root.is_absolute() else ROOT / args.voe_root
    out = args.out if args.out.is_absolute() else ROOT / args.out

    if not idx_path.exists():
        raise SystemExit(f"Missing VoE index: {idx_path}")

    data = json.loads(idx_path.read_text(encoding="utf-8"))
    pairs = data.get("pairs", [])

    seen_types: set[str] = set()
    exported: list[dict] = []

    if out.exists():
        shutil.rmtree(out)
    out.mkdir(parents=True, exist_ok=True)

    for entry in pairs:
        vtype = entry["violation_type"]
        if vtype in seen_types:
            continue
        if not str(entry.get("matched_seed", "")).endswith(args.seed_suffix.lstrip("s")):
            if not entry["pair_id"].endswith(f"_s{args.seed_suffix}"):
                continue

        pair_id = entry["pair_id"]
        src_pair = voe_root / pair_id
        if not src_pair.exists():
            print(f"skip missing {src_pair}")
            continue

        dst_pair = out / pair_id
        for sub in ("possible", "impossible"):
            src_branch = src_pair / sub
            dst_branch = dst_pair / sub
            if not src_branch.exists():
                raise SystemExit(f"Missing branch {src_branch}")
            shutil.copytree(src_branch, dst_branch)

        pair_meta = src_pair / "pair_meta.json"
        if pair_meta.exists():
            shutil.copy2(pair_meta, dst_pair / "pair_meta.json")

        exported.append(
            {
                "pair_id": pair_id,
                "violation_type": vtype,
                "t_star": entry["t_star"],
                "base": pair_id,
                "description": json.loads(pair_meta.read_text(encoding="utf-8")).get(
                    "description"
                )
                if pair_meta.exists()
                else None,
            }
        )
        seen_types.add(vtype)
        print(f"exported {pair_id} ({vtype})")

    if not exported:
        raise SystemExit("No pairs exported — run generate_voe.py first")

    payload = {"schema_version": 1, "pairs": exported}
    (out / "index.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(f"Wrote {len(exported)} pairs -> {out / 'index.json'}")


if __name__ == "__main__":
    main()
