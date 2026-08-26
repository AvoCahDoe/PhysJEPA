"""Conv decoder: latent vector -> 64x64 RGB frame."""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class ConvPixelDecoder(nn.Module):
    """
    Map a flat conditioning vector to one RGB frame.

    8x8 feature map upsampled 3x to 64x64.
    """

    def __init__(self, in_dim: int, base_channels: int = 256, out_size: int = 64):
        super().__init__()
        self.out_size = out_size
        self.fc = nn.Linear(in_dim, base_channels * 8 * 8)
        self.decode = nn.Sequential(
            nn.ConvTranspose2d(base_channels, 128, kernel_size=4, stride=2, padding=1),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(128, 64, kernel_size=4, stride=2, padding=1),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(64, 32, kernel_size=4, stride=2, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 3, kernel_size=3, padding=1),
        )

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        h = self.fc(z)
        h = h.view(z.shape[0], -1, 8, 8)
        h = self.decode(h)
        return h.clamp(0.0, 1.0)
