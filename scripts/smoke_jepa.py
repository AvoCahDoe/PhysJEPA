#!/usr/bin/env python3
"""CPU smoke: short overfit on a few episodes; check loss drop + non-collapse."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import torch
from torch.utils.data import DataLoader

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from physjepa.data import JEPAWindowDataset, RolloutDataset
from physjepa.models import JEPAModule
from physjepa.train import jepa_collate


def main() -> None:
    parser = argparse.ArgumentParser(description="JEPA smoke test")
    parser.add_argument("--index", type=Path, default=ROOT / "data" / "train" / "index.json")
    parser.add_argument("--n-episodes", type=int, default=4)
    parser.add_argument("--steps", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=8)
    args = parser.parse_args()

    device = torch.device("cpu")
    base = RolloutDataset(args.index, cache_frames=True, seed_max_exclusive=9000)
    entries = base.entries[: args.n_episodes]
    rollout = RolloutDataset(root=base.root, entries=entries, cache_frames=True)
    ds = JEPAWindowDataset(
        rollout,
        context_len=4,
        pred_horizon=1,
        windows_per_episode=8,
        seed=0,
    )
    loader = DataLoader(
        ds,
        batch_size=min(args.batch_size, len(ds)),
        shuffle=True,
        collate_fn=jepa_collate,
    )

    model = JEPAModule(context_len=4, pred_horizon=1, latent_dim=128, base_channels=16)
    model.to(device)
    opt = torch.optim.AdamW(
        list(model.encoder.parameters()) + list(model.predictor.parameters()),
        lr=1e-3,
    )

    losses: list[float] = []
    stds: list[float] = []
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
        model.ema_update(0.99)
        losses.append(float(out["loss"].item()))
        stds.append(float(out["latent_std"].item()))
        print(f"step={step+1} loss={losses[-1]:.4f} std={stds[-1]:.4f}")

    # Checkpoint round-trip
    ckpt = ROOT / "runs" / "jepa" / "smoke" / "ckpt_smoke.pt"
    ckpt.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"model": model.state_dict()}, ckpt)
    model2 = JEPAModule(context_len=4, pred_horizon=1, latent_dim=128, base_channels=16)
    model2.load_state_dict(torch.load(ckpt, map_location="cpu", weights_only=True)["model"])
    model2.eval()
    with torch.no_grad():
        batch = next(iter(loader))
        out2 = model2(batch["context_frames"], batch["future_frames"])
    print(f"reload_ok loss={float(out2['loss'].item()):.4f}")

    early = sum(losses[:5]) / 5
    late = sum(losses[-5:]) / 5
    mean_std = sum(stds) / len(stds)
    ok_loss = late < early * 0.98 or late < early - 0.01
    ok_std = mean_std > 1e-3
    print(f"early_loss={early:.4f} late_loss={late:.4f} mean_std={mean_std:.4f}")
    if not ok_loss:
        raise SystemExit("FAIL: loss did not decrease")
    if not ok_std:
        raise SystemExit("FAIL: latent std collapsed")
    print("SMOKE OK")


if __name__ == "__main__":
    main()
