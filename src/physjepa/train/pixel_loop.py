"""Pixel baseline training loop."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import torch
from torch.utils.data import DataLoader

from physjepa.models import PixelVideoModule
from physjepa.train.metrics import MetricsWriter


@torch.no_grad()
def evaluate_pixel(
    model: PixelVideoModule,
    loader: DataLoader,
    device: torch.device,
) -> dict[str, float]:
    model.eval()
    total_loss = 0.0
    total_std = 0.0
    total_mse = 0.0
    n = 0
    for batch in loader:
        ctx = batch["context_frames"].to(device)
        fut = batch["future_frames"].to(device)
        out = model(ctx, fut)
        bs = ctx.shape[0]
        total_loss += float(out["loss"].item()) * bs
        total_std += float(out["latent_std"].item()) * bs
        total_mse += float(out["pixel_mse"].item()) * bs
        n += bs
    model.train()
    if n == 0:
        return {"val_loss": float("nan"), "latent_std": float("nan"), "pixel_mse": float("nan")}
    return {
        "val_loss": total_loss / n,
        "latent_std": total_std / n,
        "pixel_mse": total_mse / n,
    }


def train_pixel(
    model: PixelVideoModule,
    train_loader: DataLoader,
    val_loader: DataLoader | None,
    optimizer: torch.optim.Optimizer,
    *,
    device: torch.device,
    epochs: int,
    grad_clip: float,
    log_every: int,
    metrics: MetricsWriter,
    ckpt_dir: Any,
) -> dict[str, Any]:
    ckpt_dir = Path(ckpt_dir)
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    model.to(device)
    model.train()

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

            global_step += 1
            do_log = global_step % log_every == 0 or global_step == 1
            val_loss = None
            if do_log and val_loader is not None and len(val_loader) > 0:
                ev = evaluate_pixel(model, val_loader, device)
                val_loss = ev["val_loss"]
                if val_loss < best_val:
                    best_val = val_loss
                    torch.save(
                        {
                            "model": model.state_dict(),
                            "model_type": "pixel",
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
                    extra={
                        "epoch": epoch,
                        "pixel_mse": float(out["pixel_mse"].item()),
                    },
                )
                print(
                    f"step={global_step} epoch={epoch} loss={loss.item():.4f} "
                    f"pix_mse={out['pixel_mse'].item():.4f} std={out['latent_std'].item():.4f} "
                    + (f"val={val_loss:.4f}" if val_loss is not None else "")
                )

        torch.save(
            {
                "model": model.state_dict(),
                "model_type": "pixel",
                "step": global_step,
                "epoch": epoch,
                "optimizer": optimizer.state_dict(),
            },
            ckpt_dir / "ckpt_last.pt",
        )

    metrics.write_summary(extra={"epochs": epochs, "steps": global_step, "model_type": "pixel"})
    return {"steps": global_step, "best_val_loss": metrics.best_val_loss}
