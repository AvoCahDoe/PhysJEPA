"""JEPA module: context encoder + EMA target + future-latent predictor."""

from __future__ import annotations

import copy

import torch
import torch.nn as nn
import torch.nn.functional as F

from .encoder import CNNEncoder
from .predictor import build_predictor


class JEPAModule(nn.Module):
    def __init__(
        self,
        latent_dim: int = 256,
        base_channels: int = 32,
        context_len: int = 4,
        pred_horizon: int = 1,
        predictor_hidden: int = 512,
        predictor_layers: int = 3,
        ema_momentum: float = 0.996,
        predictor_type: str = "mlp",
    ):
        super().__init__()
        self.context_len = context_len
        self.pred_horizon = pred_horizon
        self.latent_dim = latent_dim
        self.ema_momentum = ema_momentum
        self.predictor_type = predictor_type

        self.encoder = CNNEncoder(latent_dim=latent_dim, base_channels=base_channels)
        self.target_encoder = copy.deepcopy(self.encoder)
        for p in self.target_encoder.parameters():
            p.requires_grad = False

        self.predictor = build_predictor(
            predictor_type,
            latent_dim=latent_dim,
            context_len=context_len,
            pred_horizon=pred_horizon,
            hidden_dim=predictor_hidden,
            n_layers=predictor_layers,
        )

    @torch.no_grad()
    def ema_update(self, momentum: float | None = None) -> None:
        m = self.ema_momentum if momentum is None else momentum
        for p_online, p_target in zip(
            self.encoder.parameters(), self.target_encoder.parameters()
        ):
            p_target.data.mul_(m).add_(p_online.data, alpha=1.0 - m)
        for b_online, b_target in zip(
            self.encoder.buffers(), self.target_encoder.buffers()
        ):
            b_target.data.copy_(b_online.data)

    def encode(self, frames: torch.Tensor) -> torch.Tensor:
        """
        Encode a batch of frame sequences.

        frames: (B, T, 3, H, W) -> (B, T, D)
        """
        b, t, c, h, w = frames.shape
        flat = frames.reshape(b * t, c, h, w)
        z = self.encoder(flat)
        return z.view(b, t, -1)

    @torch.no_grad()
    def encode_target(self, frames: torch.Tensor) -> torch.Tensor:
        b, t, c, h, w = frames.shape
        flat = frames.reshape(b * t, c, h, w)
        z = self.target_encoder(flat)
        return z.view(b, t, -1)

    def forward(
        self,
        context_frames: torch.Tensor,
        future_frames: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        """
        context_frames: (B, context_len, 3, H, W)
        future_frames:  (B, pred_horizon, 3, H, W)
        """
        context_z = self.encode(context_frames)
        pred_z = self.predictor(context_z)

        with torch.no_grad():
            target_z = self.encode_target(future_frames)

        loss = F.smooth_l1_loss(pred_z, target_z)

        with torch.no_grad():
            latent_std = context_z.float().std(dim=0).mean()
            latent_norm = context_z.float().norm(dim=-1).mean()

        return {
            "loss": loss,
            "pred_z": pred_z,
            "target_z": target_z,
            "context_z": context_z,
            "latent_std": latent_std,
            "latent_norm": latent_norm,
        }

    def count_parameters(self) -> dict[str, int]:
        def _n(m: nn.Module) -> int:
            return sum(p.numel() for p in m.parameters() if p.requires_grad)

        return {
            "encoder": _n(self.encoder),
            "predictor": _n(self.predictor),
            "trainable_total": _n(self.encoder) + _n(self.predictor),
        }
