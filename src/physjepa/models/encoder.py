"""Tiny ResNet-style CNN encoder: 64x64 RGB -> D-dim latent."""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class ResidualBlock(nn.Module):
    def __init__(self, channels: int):
        super().__init__()
        self.conv1 = nn.Conv2d(channels, channels, 3, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(channels)
        self.conv2 = nn.Conv2d(channels, channels, 3, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(channels)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = F.relu(self.bn1(self.conv1(x)), inplace=True)
        h = self.bn2(self.conv2(h))
        return F.relu(x + h, inplace=True)


class CNNEncoder(nn.Module):
    """
    Stem + residual stages + global average pool + linear projection.

    Input:  (B, 3, H, W) float in [0, 1]
    Output: (B, D)
    """

    def __init__(self, latent_dim: int = 256, base_channels: int = 32):
        super().__init__()
        c = base_channels
        self.stem = nn.Sequential(
            nn.Conv2d(3, c, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(c),
            nn.ReLU(inplace=True),
        )
        self.stage1 = nn.Sequential(
            nn.Conv2d(c, c * 2, 3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(c * 2),
            nn.ReLU(inplace=True),
            ResidualBlock(c * 2),
        )
        self.stage2 = nn.Sequential(
            nn.Conv2d(c * 2, c * 4, 3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(c * 4),
            nn.ReLU(inplace=True),
            ResidualBlock(c * 4),
        )
        self.stage3 = nn.Sequential(
            nn.Conv2d(c * 4, c * 8, 3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(c * 8),
            nn.ReLU(inplace=True),
            ResidualBlock(c * 8),
        )
        self.proj = nn.Linear(c * 8, latent_dim)
        self.latent_dim = latent_dim

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.stem(x)
        h = self.stage1(h)
        h = self.stage2(h)
        h = self.stage3(h)
        h = h.mean(dim=(2, 3))
        return self.proj(h)
