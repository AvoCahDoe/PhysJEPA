#!/usr/bin/env python3
"""CPU smoke test for pixel baseline."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import torch
from torch.utils.data import DataLoader

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from physjepa.data import JEPAWindowDataset, RolloutDataset
from physjepa.models import PixelVideoModule
from physjepa.train import jepa_collate


def main() -> None:
    parser = argparse.ArgumentParser(description="Pixel baseline smoke test")
    parser.add_argument("--index", type=Path, default=ROOT / "data" / "train" / "index.json")
    parser.add_argument("--steps", type=int, default=20)
    args = parser.parse_args()

    device = torch.device("cpu")
    base = RolloutDataset(args.index, cache_frames=True, seed_max_exclusive=9000)
    rollout = RolloutDataset(root=base.root, entries=base.entries[:4], cache_frames=True)
    ds = JEPAWindowDataset(rollout, context_len=4, pred_horizon=1, windows_per_episode=8, seed=0)
    loader = DataLoader(ds, batch_size=4, shuffle=True, collate_fn=jepa_collate)

    model = PixelVideoModule(
        latent_dim=128, base_channels=16, context_len=4, pred_horizon=1, decoder_channels=128
    ).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=1e-3)

    losses = []
    it = iter(loader)
    for step in range(args.steps):
        try:
            batch = next(it)
        except StopIteration:
            it = iter(loader)
            batch = next(it)
        ctx = batch["context_frames"].to(device)
        fut = batch["future_frames"].to(device)
        out = model(ctx, fut)
        opt.zero_grad(set_to_none=True)
        out["loss"].backward()
        opt.step()
        losses.append(float(out["loss"].item()))
        print(f"step={step+1} loss={losses[-1]:.4f} pix_mse={out['pixel_mse'].item():.4f}")

    ckpt = ROOT / "runs" / "pixel" / "smoke" / "ckpt_smoke.pt"
    ckpt.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"model": model.state_dict(), "model_type": "pixel"}, ckpt)

    early = sum(losses[:5]) / 5
    late = sum(losses[-5:]) / 5
    if late >= early * 0.98 and late >= early - 0.01:
        raise SystemExit("FAIL: loss did not decrease")
    print(f"early={early:.4f} late={late:.4f}")
    print("SMOKE OK")


if __name__ == "__main__":
    main()
