"""JEPA training loop helpers."""

from __future__ import annotations

import math
from typing import Any

import torch
from torch.utils.data import DataLoader

from physjepa.models import JEPAModule
from physjepa.train.metrics import MetricsWriter


def ema_momentum_cosine(
    step: int,
    total_steps: int,
    base: float = 0.996,
    final: float = 1.0,
) -> float:
    """Cosine schedule from base toward final."""
    if total_steps <= 1:
        return final
    t = min(step, total_steps - 1) / (total_steps - 1)
    return final - (final - base) * (1 + math.cos(math.pi * t)) / 2


@torch.no_grad()
def evaluate(
    model: JEPAModule,
    loader: DataLoader,
    device: torch.device,
) -> dict[str, float]:
    model.eval()
    total_loss = 0.0
    total_std = 0.0
    total_norm = 0.0
    n = 0
    for batch in loader:
        ctx = batch["context_frames"].to(device)
        fut = batch["future_frames"].to(device)
        out = model(ctx, fut)
        bs = ctx.shape[0]
        total_loss += float(out["loss"].item()) * bs
        total_std += float(out["latent_std"].item()) * bs
        total_norm += float(out["latent_norm"].item()) * bs
        n += bs
    model.train()
    if n == 0:
        return {"val_loss": float("nan"), "latent_std": float("nan"), "latent_norm": float("nan")}
    return {
        "val_loss": total_loss / n,
        "latent_std": total_std / n,
        "latent_norm": total_norm / n,
    }


def train_jepa(
    model: JEPAModule,
    train_loader: DataLoader,
    val_loader: DataLoader | None,
    optimizer: torch.optim.Optimizer,
    *,
    device: torch.device,
    epochs: int,
    grad_clip: float,
    ema_base: float,
    ema_final: float,
    log_every: int,
    metrics: MetricsWriter,
    ckpt_dir: Any,
) -> dict[str, Any]:
    from pathlib import Path

    ckpt_dir = Path(ckpt_dir)
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    model.to(device)
    model.train()

    steps_per_epoch = max(1, len(train_loader))
    total_steps = epochs * steps_per_epoch
    global_step = 0
    best_val = float("inf")

    for epoch in range(epochs):
        for batch in train_loader:
            ctx = batch["context_frames"].to(device)
            fut = batch["future_frames"].to(device)

            out = model(ctx, fut)
            loss = out["loss"]

            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            if grad_clip > 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
            optimizer.step()

            m = ema_momentum_cosine(global_step, total_steps, ema_base, ema_final)
            model.ema_update(m)

            global_step += 1
            do_log = global_step % log_every == 0 or global_step == 1
            val_loss = None
            if do_log and val_loader is not None and len(val_loader) > 0:
                ev = evaluate(model, val_loader, device)
                val_loss = ev["val_loss"]
                if val_loss < best_val:
                    best_val = val_loss
                    torch.save(
                        {
                            "model": model.state_dict(),
                            "model_type": "jepa",
                            "step": global_step,
                            "epoch": epoch,
                            "val_loss": val_loss,
                        },
                        ckpt_dir / "ckpt_best.pt",
                    )

            if do_log:
                metrics.log_step(
                    global_step,
                    loss=float(loss.item()),
                    latent_std=float(out["latent_std"].item()),
                    latent_norm=float(out["latent_norm"].item()),
                    val_loss=val_loss,
                    extra={"epoch": epoch, "ema_momentum": m},
                )
                print(
                    f"step={global_step} epoch={epoch} loss={loss.item():.4f} "
                    f"std={out['latent_std'].item():.4f} "
                    + (f"val={val_loss:.4f}" if val_loss is not None else "")
                )

        torch.save(
            {
                "model": model.state_dict(),
                "model_type": "jepa",
                "step": global_step,
                "epoch": epoch,
                "optimizer": optimizer.state_dict(),
            },
            ckpt_dir / "ckpt_last.pt",
        )

    summary_extra = {"epochs": epochs, "steps": global_step}
    metrics.write_summary(extra=summary_extra)
    return {"steps": global_step, "best_val_loss": metrics.best_val_loss}
