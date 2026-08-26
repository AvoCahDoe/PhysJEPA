"""Pixel video-prediction baseline: encode context, decode next frame(s)."""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from .decoder import ConvPixelDecoder
from .encoder import CNNEncoder
from .predictor import FutureLatentMLP


class PixelVideoModule(nn.Module):
    """
    Comparable-capacity pixel baseline for JEPA ablation.

    Context frames -> shared CNN encoder -> MLP bottleneck -> conv decoder -> pixels.
    Loss: smooth L1 on predicted vs actual future frames.
    """

    def __init__(
        self,
        latent_dim: int = 256,
        base_channels: int = 32,
        context_len: int = 4,
        pred_horizon: int = 1,
        predictor_hidden: int = 512,
        predictor_layers: int = 3,
        decoder_channels: int = 256,
    ):
        super().__init__()
        self.context_len = context_len
        self.pred_horizon = pred_horizon
        self.latent_dim = latent_dim

        self.encoder = CNNEncoder(latent_dim=latent_dim, base_channels=base_channels)
        self.bottleneck = FutureLatentMLP(
            latent_dim=latent_dim,
            context_len=context_len,
            pred_horizon=pred_horizon,
            hidden_dim=predictor_hidden,
            n_layers=predictor_layers,
        )
        self.decoder = ConvPixelDecoder(
            in_dim=pred_horizon * latent_dim,
            base_channels=decoder_channels,
        )

    def encode(self, frames: torch.Tensor) -> torch.Tensor:
        """frames: (B, T, 3, H, W) -> (B, T, D)"""
        b, t, c, h, w = frames.shape
        flat = frames.reshape(b * t, c, h, w)
        z = self.encoder(flat)
        return z.view(b, t, -1)

    def predict_pixels(self, context_frames: torch.Tensor) -> torch.Tensor:
        """
        context_frames: (B, context_len, 3, H, W)
        returns: (B, pred_horizon, 3, H, W)
        """
        context_z = self.encode(context_frames)
        flat = self.bottleneck(context_z).reshape(
            context_z.shape[0], self.pred_horizon * self.latent_dim
        )
        pix = self.decoder(flat)
        if self.pred_horizon == 1:
            return pix.unsqueeze(1)
        # Split per future step when horizon > 1
        chunks = flat.view(context_z.shape[0], self.pred_horizon, self.latent_dim)
        outs = [self.decoder(chunks[:, i]) for i in range(self.pred_horizon)]
        return torch.stack(outs, dim=1)

    def forward(
        self,
        context_frames: torch.Tensor,
        future_frames: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        pred = self.predict_pixels(context_frames)
        loss = F.smooth_l1_loss(pred, future_frames)

        with torch.no_grad():
            context_z = self.encode(context_frames)
            latent_std = context_z.float().std(dim=0).mean()
            latent_norm = context_z.float().norm(dim=-1).mean()
            pixel_mse = F.mse_loss(pred, future_frames)

        return {
            "loss": loss,
            "pred_pixels": pred,
            "context_z": context_z,
            "latent_std": latent_std,
            "latent_norm": latent_norm,
            "pixel_mse": pixel_mse,
        }

    def count_parameters(self) -> dict[str, int]:
        def _n(m: nn.Module) -> int:
            return sum(p.numel() for p in m.parameters() if p.requires_grad)

        return {
            "encoder": _n(self.encoder),
            "bottleneck": _n(self.bottleneck),
            "decoder": _n(self.decoder),
            "trainable_total": _n(self.encoder) + _n(self.bottleneck) + _n(self.decoder),
        }
