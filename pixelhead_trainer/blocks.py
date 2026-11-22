"""Core 2D & 3D Causal ConvNeXt building blocks and exact N-D PixelShuffle rearrangements.

License: MIT
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class ConvNeXtBlock2d(nn.Module):
    """2D ConvNeXt block with 3x3 depthwise conv, channel LayerNorm, 4x GELU MLP, and LayerScale."""

    def __init__(self, dim: int, mlp_ratio: int = 4, ls_init: float = 1.0):
        super().__init__()
        self.dw = nn.Conv2d(dim, dim, kernel_size=3, padding=1, groups=dim)
        self.norm = nn.LayerNorm(dim, eps=1e-6)
        hidden = dim * mlp_ratio
        self.fc1 = nn.Linear(dim, hidden)
        self.fc2 = nn.Linear(hidden, dim)
        self.gamma = nn.Parameter(torch.full((dim,), float(ls_init)))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.dw(x).movedim(1, -1)  # [B, H, W, C]
        h = self.fc2(F.gelu(self.fc1(self.norm(h))))
        h = (h * self.gamma).movedim(-1, 1)
        return x + h


class CausalDepthwiseConv3d(nn.Module):
    """3D depthwise conv with left-only causal temporal padding and replicate spatial padding."""

    def __init__(self, dim: int, kt: int = 3, kh: int = 3, kw: int = 3):
        super().__init__()
        self.pad_t = kt - 1
        self.pad_h = kh // 2
        self.pad_w = kw // 2
        self.conv = nn.Conv3d(dim, dim, kernel_size=(kt, kh, kw), padding=0, groups=dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [B, C, T, H, W]; replicate-pad first frame so t=0 has clean causal boundary
        if self.pad_t > 0:
            first = x[:, :, :1].expand(-1, -1, self.pad_t, -1, -1)
            x = torch.cat([first, x], dim=2)
        if self.pad_h > 0 or self.pad_w > 0:
            x = F.pad(x, (self.pad_w, self.pad_w, self.pad_h, self.pad_h, 0, 0), mode="replicate")
        return self.conv(x)


class CausalConvNeXtBlock3d(nn.Module):
    """3D Causal ConvNeXt block with channel LayerNorm, 4x GELU pointwise MLP, and LayerScale."""

    def __init__(self, dim: int, mlp_ratio: int = 4, ls_init: float = 1.0):
        super().__init__()
        self.dw = CausalDepthwiseConv3d(dim, 3, 3, 3)
        self.norm = nn.LayerNorm(dim, eps=1e-6)
        hidden = dim * mlp_ratio
        self.fc1 = nn.Linear(dim, hidden)
        self.fc2 = nn.Linear(hidden, dim)
        self.gamma = nn.Parameter(torch.full((dim,), float(ls_init)))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.dw(x).movedim(1, -1)  # [B, T, H, W, C]
        h = self.fc2(F.gelu(self.fc1(self.norm(h))))
        h = (h * self.gamma).movedim(-1, 1).contiguous()
        return x + h


def unpatchify_2d(tokens: torch.Tensor, patch: int, out_channels: int) -> torch.Tensor:
    """Convert `[B, gh, gw, patch * patch * C]` -> `[B, C, gh * patch, gw * patch]`."""
    b, gh, gw, _ = tokens.shape
    x = tokens.view(b, gh, gw, patch, patch, out_channels)
    x = x.permute(0, 5, 1, 3, 2, 4).contiguous()
    return x.view(b, out_channels, gh * patch, gw * patch)


def unpatchify_3d(
    tokens: torch.Tensor, pt: int, ph: int, pw: int, out_channels: int = 3
) -> torch.Tensor:
    """Convert `[B, T_g, H_g, W_g, pt * ph * pw * C]` -> `[B, C, T_g * pt, H_g * ph, W_g * pw]`."""
    b, tg, gh, gw, _ = tokens.shape
    x = tokens.view(b, tg, gh, gw, pt, ph, pw, out_channels)
    x = x.permute(0, 7, 1, 4, 2, 5, 3, 6).contiguous()
    return x.view(b, out_channels, tg * pt, gh * ph, gw * pw)


def spatial_pixel_shuffle_3d(
    x: torch.Tensor, norm: nn.LayerNorm, proj: nn.Linear, scale: int = 2
) -> torch.Tensor:
    """Exact per-frame `(1, scale, scale)` spatial PixelShuffle on `[B, C, T, H, W]`.

    Important: Explicitly unflatten to 7D `(B, T, H, W, C_out, scale, scale)` and permute
    `(0, 4, 1, 2, 5, 3, 6)` so `T`, `H`, and `W` axes can never scramble on non-square videos.
    """
    b, _, t, h, w = x.shape
    u = proj(norm(x.movedim(1, -1)))  # [B, T, H, W, C_out * scale * scale]
    c_out = u.shape[-1] // (scale * scale)
    u = u.view(b, t, h, w, c_out, scale, scale)
    return u.permute(0, 4, 1, 2, 5, 3, 6).contiguous().view(b, c_out, t, h * scale, w * scale)


def spatiotemporal_pixel_shuffle_3d(
    x: torch.Tensor, norm: nn.LayerNorm, proj: nn.Linear, pt: int = 2, ps: int = 2
) -> torch.Tensor:
    """Exact 3D `(pt, ps, ps)` spatiotemporal PixelShuffle on `[B, C, T, H, W]`.

    Unflattens to 8D `(B, T, H, W, C_out, pt, ps, ps)` and permutes `(0, 4, 1, 5, 2, 6, 3, 7)`.
    """
    b, _, t, h, w = x.shape
    u = proj(norm(x.movedim(1, -1)))  # [B, T, H, W, C_out * pt * ps * ps]
    c_out = u.shape[-1] // (pt * ps * ps)
    u = u.view(b, t, h, w, c_out, pt, ps, ps)
    return u.permute(0, 4, 1, 5, 2, 6, 3, 7).contiguous().view(b, c_out, t * pt, h * ps, w * ps)
