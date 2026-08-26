"""Future-latent predictors: MLP, GRU, and tiny Transformer."""

from __future__ import annotations

import torch
import torch.nn as nn


class FutureLatentMLP(nn.Module):
    """
    Predict k future latents from (context_len) context latents.

    Input:  (B, context_len, D)
    Output: (B, pred_horizon, D)
    """

    def __init__(
        self,
        latent_dim: int = 256,
        context_len: int = 4,
        pred_horizon: int = 1,
        hidden_dim: int = 512,
        n_layers: int = 3,
    ):
        super().__init__()
        self.latent_dim = latent_dim
        self.context_len = context_len
        self.pred_horizon = pred_horizon

        in_dim = context_len * latent_dim
        out_dim = pred_horizon * latent_dim
        layers: list[nn.Module] = []
        prev = in_dim
        for _ in range(max(1, n_layers - 1)):
            layers.extend([nn.Linear(prev, hidden_dim), nn.GELU()])
            prev = hidden_dim
        layers.append(nn.Linear(prev, out_dim))
        self.net = nn.Sequential(*layers)

    def forward(self, context: torch.Tensor) -> torch.Tensor:
        b = context.shape[0]
        flat = context.reshape(b, -1)
        out = self.net(flat)
        return out.view(b, self.pred_horizon, self.latent_dim)


class FutureLatentGRU(nn.Module):
    """GRU over context latents; project final state to future latents."""

    def __init__(
        self,
        latent_dim: int = 256,
        context_len: int = 4,
        pred_horizon: int = 1,
        hidden_dim: int = 512,
        n_layers: int = 1,
    ):
        super().__init__()
        self.latent_dim = latent_dim
        self.context_len = context_len
        self.pred_horizon = pred_horizon
        self.gru = nn.GRU(
            input_size=latent_dim,
            hidden_size=hidden_dim,
            num_layers=n_layers,
            batch_first=True,
        )
        self.head = nn.Linear(hidden_dim, pred_horizon * latent_dim)

    def forward(self, context: torch.Tensor) -> torch.Tensor:
        # context: (B, T, D)
        _, h_n = self.gru(context)
        # h_n: (num_layers, B, H) -> last layer
        h = h_n[-1]
        out = self.head(h)
        return out.view(context.shape[0], self.pred_horizon, self.latent_dim)


class FutureLatentTransformer(nn.Module):
    """Tiny Transformer encoder over context + learnable future query tokens."""

    def __init__(
        self,
        latent_dim: int = 256,
        context_len: int = 4,
        pred_horizon: int = 1,
        hidden_dim: int = 512,
        n_layers: int = 2,
        n_heads: int = 4,
    ):
        super().__init__()
        self.latent_dim = latent_dim
        self.context_len = context_len
        self.pred_horizon = pred_horizon
        self.input_proj = nn.Linear(latent_dim, hidden_dim)
        max_len = context_len + pred_horizon + 8
        self.pos = nn.Parameter(torch.zeros(1, max_len, hidden_dim))
        nn.init.normal_(self.pos, std=0.02)
        self.future_queries = nn.Parameter(torch.zeros(1, pred_horizon, hidden_dim))
        nn.init.normal_(self.future_queries, std=0.02)
        layer = nn.TransformerEncoderLayer(
            d_model=hidden_dim,
            nhead=n_heads,
            dim_feedforward=hidden_dim * 2,
            batch_first=True,
            activation="gelu",
            norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(layer, num_layers=n_layers)
        self.out_proj = nn.Linear(hidden_dim, latent_dim)

    def forward(self, context: torch.Tensor) -> torch.Tensor:
        b, t, _ = context.shape
        ctx = self.input_proj(context)
        queries = self.future_queries.expand(b, -1, -1)
        tokens = torch.cat([ctx, queries], dim=1)
        tokens = tokens + self.pos[:, : tokens.shape[1], :]
        encoded = self.encoder(tokens)
        future = encoded[:, t : t + self.pred_horizon, :]
        return self.out_proj(future)


def build_predictor(
    predictor_type: str = "mlp",
    *,
    latent_dim: int = 256,
    context_len: int = 4,
    pred_horizon: int = 1,
    hidden_dim: int = 512,
    n_layers: int = 3,
) -> nn.Module:
    ptype = predictor_type.lower().strip()
    if ptype == "mlp":
        return FutureLatentMLP(
            latent_dim=latent_dim,
            context_len=context_len,
            pred_horizon=pred_horizon,
            hidden_dim=hidden_dim,
            n_layers=n_layers,
        )
    if ptype == "gru":
        return FutureLatentGRU(
            latent_dim=latent_dim,
            context_len=context_len,
            pred_horizon=pred_horizon,
            hidden_dim=hidden_dim,
            n_layers=max(1, min(n_layers, 2)),
        )
    if ptype in ("transformer", "tf", "vit"):
        # Keep heads divisible into hidden_dim
        heads = 4 if hidden_dim % 4 == 0 else 2
        return FutureLatentTransformer(
            latent_dim=latent_dim,
            context_len=context_len,
            pred_horizon=pred_horizon,
            hidden_dim=hidden_dim,
            n_layers=max(1, min(n_layers, 4)),
            n_heads=heads,
        )
    raise ValueError(f"Unknown predictor_type: {predictor_type}")
