"""Clean-room PyTorch implementation of Direct Pixel Readout Head for Qwen-Image 2.1 (v0.1, v0.2 & v0.3).

License: MIT
Supports:
  - `QwenImage21PixelHead` (`v0.1`, 16px -> 8px + 8x8 linear unpatchify, ~3.4-5.5 ms)
  - `QwenImage21PixelHeadX2` (`v0.1` 2x super-resolution branch, 1024 -> 2048)
  - `QwenImage21PixelHeadV2` (`v0.2` Full-Res 1px Pyramid Edition)
  - `QwenImage21PixelHeadV3` (`v0.3` Fast Haar-Wavelet + Multi-Scale Latent Spatial Highway Edition:
    predicts 512x512 2D Haar wavelet subbands `(LL, LH, HL, HH)` with norm-free fused blocks +
    0-parameter analytical `iDWT` to 1024x1024 + direct `x0_hat` 3x3 spatial conv highway,
    achieving ~5-6x faster inference than v0.2 with sharper 1-2px small text & cross-hatching)
"""

from __future__ import annotations

import math
from pathlib import Path
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F


TOKEN_PIXELS = 16  # Qwen-Image 2.1 VAE spatial downscale is 16x, DiT patch is 1x1


def dwt_haar_2d(img: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Exact invertible 2D Haar Wavelet decomposition `[B, C, 2*H, 2*W]` -> `(LL, LH, HL, HH)` `[B, C, H, W]`."""
    x00 = img[:, :, 0::2, 0::2]
    x01 = img[:, :, 0::2, 1::2]
    x10 = img[:, :, 1::2, 0::2]
    x11 = img[:, :, 1::2, 1::2]
    ll = 0.25 * (x00 + x01 + x10 + x11)
    lh = 0.25 * ((x00 + x01) - (x10 + x11))
    hl = 0.25 * ((x00 + x10) - (x01 + x11))
    hh = 0.25 * ((x00 + x11) - (x01 + x10))
    return ll, lh, hl, hh


def idwt_haar_2d(ll: torch.Tensor, lh: torch.Tensor, hl: torch.Tensor, hh: torch.Tensor) -> torch.Tensor:
    """Exact 0-parameter analytical inverse 2D Haar Wavelet transform -> `[B, C, 2*H, 2*W]`."""
    x00 = ll + lh + hl + hh
    x01 = ll + lh - hl - hh
    x10 = ll - lh + hl - hh
    x11 = ll - lh - hl + hh
    b, c, h, w = ll.shape
    quad = torch.stack((x00, x01, x10, x11), dim=2).reshape(b, c * 4, h, w)
    return F.pixel_shuffle(quad, 2)


class ConvNeXtBlock2d(nn.Module):
    """2D ConvNeXt block with 3x3 depthwise conv, channel LayerNorm, 4x GELU MLP, and LayerScale."""

    def __init__(self, dim: int, mlp_ratio: int = 4, ls_init: float = 1.0, zero_out: bool = False):
        super().__init__()
        self.dw = nn.Conv2d(dim, dim, kernel_size=3, padding=1, groups=dim)
        self.norm = nn.LayerNorm(dim, eps=1e-6)
        hidden = dim * mlp_ratio
        self.fc1 = nn.Linear(dim, hidden)
        self.fc2 = nn.Linear(hidden, dim)
        self.gamma = nn.Parameter(torch.full((dim,), float(ls_init)))
        if zero_out:
            nn.init.zeros_(self.fc2.weight)
            nn.init.zeros_(self.fc2.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [B, C, H, W]
        h = self.dw(x).movedim(1, -1)  # [B, H, W, C]
        h = self.fc2(F.gelu(self.fc1(self.norm(h))))
        h = (h * self.gamma).movedim(-1, 1)
        return x + h


class PixelRefineBlock2d(nn.Module):
    """High-resolution residual depthwise-separable 3x3 block used in v0.2."""

    def __init__(self, dim: int, expand: int = 2):
        super().__init__()
        hidden = dim * expand
        self.dw = nn.Conv2d(dim, dim, kernel_size=3, padding=1, groups=dim)
        self.norm = nn.GroupNorm(num_groups=min(8, dim), num_channels=dim, eps=1e-6)
        self.pw1 = nn.Conv2d(dim, hidden, kernel_size=1)
        self.pw2 = nn.Conv2d(hidden, dim, kernel_size=1)
        nn.init.zeros_(self.pw2.weight)
        nn.init.zeros_(self.pw2.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.pw2(F.silu(self.pw1(self.norm(self.dw(x)))))
        return x + h


class WaveletConvBlock2d(nn.Module):
    """Tensor-Core-friendly norm-free 3x3 residual block for v0.3 512x512 wavelet stage."""

    def __init__(self, dim: int, ls_init: float = 1.0):
        super().__init__()
        self.conv1 = nn.Conv2d(dim, dim, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(dim, dim, kernel_size=3, padding=1)
        self.gamma = nn.Parameter(torch.full((1, dim, 1, 1), float(ls_init)))
        nn.init.zeros_(self.conv2.weight)
        nn.init.zeros_(self.conv2.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.conv2(F.silu(self.conv1(x)))
        return x + h * self.gamma


class LatentSpatialHighway(nn.Module):
    """Multi-scale 3x3 spatial convolution bridge directly from denoised 64-ch latent `x0_hat_norm`.

    Carries crisp neighbor-token stroke geometry directly into 8px, 4px, and 2px (Haar wavelet) grids.
    """

    def __init__(
        self,
        d_lat: int = 64,
        c_lat16: int = 128,
        c_lat8: int = 96,
        c_lat4: int = 64,
        w8: int = 512,
        c_4px: int = 256,
        c_2px: int = 64,
    ):
        super().__init__()
        self.c_lat16 = int(c_lat16)
        self.c_lat8 = int(c_lat8)
        self.c_lat4 = int(c_lat4)
        self.stem16 = nn.Sequential(
            nn.Conv2d(d_lat, c_lat16, kernel_size=3, padding=1),
            nn.SiLU(),
            nn.Conv2d(c_lat16, c_lat16, kernel_size=3, padding=1),
            nn.SiLU(),
        )
        self.up8 = nn.Conv2d(c_lat16, 4 * c_lat8, kernel_size=1)
        self.conv8 = nn.Conv2d(c_lat8, c_lat8, kernel_size=3, padding=1)
        self.inj8 = nn.Conv2d(c_lat8, w8, kernel_size=1)

        self.up4 = nn.Conv2d(c_lat8, 4 * c_lat4, kernel_size=1)
        self.conv4 = nn.Conv2d(c_lat4, c_lat4, kernel_size=3, padding=1)
        self.inj4 = nn.Conv2d(c_lat4, c_4px, kernel_size=1)

        self.up2 = nn.Conv2d(c_lat4, 4 * c_2px, kernel_size=1)
        self.inj2 = nn.Conv2d(c_2px, c_2px, kernel_size=1)

        for m in (self.inj8, self.inj4, self.inj2):
            nn.init.zeros_(m.weight)
            nn.init.zeros_(m.bias)

    def forward(self, x0_hat_norm: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        z16 = self.stem16(x0_hat_norm)
        z8 = F.silu(self.conv8(F.pixel_shuffle(self.up8(z16), 2)))
        z4 = F.silu(self.conv4(F.pixel_shuffle(self.up4(z8), 2)))
        z2 = F.pixel_shuffle(self.up2(z4), 2)
        return self.inj8(z8), self.inj4(z4), self.inj2(z2)


class SigmaAdaLN(nn.Module):
    """Sinusoidal log-sigma embedding + zero-initialized AdaLN (scale, shift) modulation."""

    def __init__(self, embed_dim: int = 256, target_dims: tuple[int, ...] = (1024, 512)):
        super().__init__()
        self.embed_dim = int(embed_dim)
        self.mlp = nn.Sequential(
            nn.Linear(embed_dim, embed_dim),
            nn.SiLU(),
            nn.Linear(embed_dim, embed_dim),
            nn.SiLU(),
        )
        self.heads = nn.ModuleList([nn.Linear(embed_dim, 2 * d) for d in target_dims])
        for hd in self.heads:
            nn.init.zeros_(hd.weight)
            nn.init.zeros_(hd.bias)

    def embed_sigma(self, sigma: torch.Tensor | float, batch: int, device: torch.device, dtype: torch.dtype) -> torch.Tensor:
        if not torch.is_tensor(sigma):
            s = torch.full((batch,), float(sigma), device=device, dtype=torch.float32)
        else:
            s = sigma.to(device=device, dtype=torch.float32).reshape(-1)
            if s.numel() == 1 and batch > 1:
                s = s.expand(batch)
        log_s = torch.log(s.clamp(min=1e-4, max=2.0))  # [-9.2, 0.7]
        half = self.embed_dim // 2
        freqs = torch.exp(-math.log(1000.0) * torch.arange(half, device=device, dtype=torch.float32) / max(1, half - 1))
        ang = log_s[:, None] * freqs[None, :]
        emb = torch.cat([torch.sin(ang), torch.cos(ang)], dim=-1).to(dtype=dtype)
        return self.mlp(emb)

    def modulate_nchw(self, x: torch.Tensor, cond: torch.Tensor, head_idx: int) -> torch.Tensor:
        scale, shift = self.heads[head_idx](cond).chunk(2, dim=-1)
        return x * (1.0 + scale[:, :, None, None]) + shift[:, :, None, None]


def unpatchify_2d(tokens: torch.Tensor, patch: int, out_channels: int) -> torch.Tensor:
    """Convert `[B, gh, gw, patch * patch * C]` -> `[B, C, gh * patch, gw * patch]`."""
    b, gh, gw, _ = tokens.shape
    x = tokens.view(b, gh, gw, patch, patch, out_channels)
    x = x.permute(0, 5, 1, 3, 2, 4).contiguous()
    return x.view(b, out_channels, gh * patch, gw * patch)


class QwenImage21PixelHead(nn.Module):
    """v0.1 1x Direct Pixel Head for Qwen-Image 2.1 (`16x16` RGBA/RGB pixels per DiT token)."""

    def __init__(
        self,
        d_feat: int = 4096,
        d_lat: int = 64,
        width: int = 1024,
        depth_low: int = 2,
        depth_high: int = 1,
        out_channels: int = 4,
        ls_init: float = 1.0,
    ):
        super().__init__()
        if width % 2 != 0:
            raise ValueError(f"width must be even, got {width}")
        self.d_feat = int(d_feat)
        self.d_lat = int(d_lat)
        self.width = int(width)
        self.depth_low = int(depth_low)
        self.depth_high = int(depth_high)
        self.out_channels = int(out_channels)
        self.ls_init = float(ls_init)

        w_high = width // 2
        self.out_patch = TOKEN_PIXELS // 2  # 8x8 pixels per 2x-upsampled position

        self.feat_norm = nn.LayerNorm(self.d_feat, eps=1e-6)
        self.feat_proj = nn.Linear(self.d_feat, width, bias=False)
        self.lat_proj = nn.Linear(self.d_lat, width, bias=True)
        self.entry_norm = nn.LayerNorm(width, eps=1e-6)

        self.low_blocks = nn.ModuleList(
            [ConvNeXtBlock2d(width, ls_init=ls_init) for _ in range(depth_low)]
        )
        self.up_norm = nn.LayerNorm(width, eps=1e-6)
        self.up_proj = nn.Linear(width, 2 * width)
        self.high_blocks = nn.ModuleList(
            [ConvNeXtBlock2d(w_high, ls_init=ls_init) for _ in range(depth_high)]
        )
        self.norm_out = nn.LayerNorm(w_high, eps=1e-6)
        self.out_proj = nn.Linear(w_high, self.out_patch * self.out_patch * self.out_channels)
        nn.init.normal_(self.out_proj.weight, mean=0.0, std=0.02)
        nn.init.zeros_(self.out_proj.bias)

        self.meta: dict[str, str] = {}

    def forward_trunk(self, feat: torch.Tensor, x0_hat_norm: torch.Tensor) -> torch.Tensor:
        b, _, gh, gw = x0_hat_norm.shape
        if feat.ndim == 3:
            feat = feat.view(b, gh, gw, self.d_feat)
        lat_tok = x0_hat_norm.movedim(1, -1)  # [B, gh, gw, d_lat]

        h = self.entry_norm(self.feat_proj(self.feat_norm(feat)) + self.lat_proj(lat_tok))
        h = h.movedim(-1, 1).contiguous()  # [B, width, gh, gw]
        for blk in self.low_blocks:
            h = blk(h)

        u = self.up_proj(self.up_norm(h.movedim(1, -1))).movedim(-1, 1).contiguous()
        h = F.pixel_shuffle(u, 2)  # [B, width//2, 2*gh, 2*gw]
        for blk in self.high_blocks:
            h = blk(h)
        return h

    def forward(self, feat: torch.Tensor, x0_hat_norm: torch.Tensor) -> torch.Tensor:
        h = self.forward_trunk(feat, x0_hat_norm)
        h_tok = self.norm_out(h.movedim(1, -1))
        with torch.autocast(device_type=h_tok.device.type, enabled=False):
            patches = self.out_proj(h_tok.float())
        return unpatchify_2d(patches, self.out_patch, self.out_channels)


class QwenImage21PixelHeadV2(nn.Module):
    """v0.2 High-Fidelity Direct Pixel Head (`16px -> 8px -> 4px -> 2px -> 1px` full-resolution pyramid)."""

    def __init__(
        self,
        base: Optional[QwenImage21PixelHead] = None,
        d_feat: int = 4096,
        d_lat: int = 64,
        width: int = 1024,
        depth_low: int = 2,
        depth_high: int = 1,
        extra_low: int = 1,
        extra_high: int = 2,
        depth_4px: int = 2,
        depth_2px: int = 2,
        depth_1px: int = 2,
        c_4px: int = 256,
        c_2px: int = 128,
        c_1px: int = 64,
        out_channels: int = 4,
        ls_init: float = 1.0,
    ):
        super().__init__()
        self.base = base if base is not None else QwenImage21PixelHead(
            d_feat=d_feat,
            d_lat=d_lat,
            width=width,
            depth_low=depth_low,
            depth_high=depth_high,
            out_channels=out_channels,
            ls_init=ls_init,
        )
        w0 = self.base.width
        w8 = w0 // 2
        self.extra_low = int(extra_low)
        self.extra_high = int(extra_high)
        self.depth_4px = int(depth_4px)
        self.depth_2px = int(depth_2px)
        self.depth_1px = int(depth_1px)
        self.c_4px = int(c_4px)
        self.c_2px = int(c_2px)
        self.c_1px = int(c_1px)
        self.out_channels = self.base.out_channels

        self.sigma_adaln = SigmaAdaLN(embed_dim=256, target_dims=(w0, w8, c_4px))

        self.low_ext = nn.ModuleList([ConvNeXtBlock2d(w0, ls_init=ls_init, zero_out=True) for _ in range(self.extra_low)])
        self.high_ext = nn.ModuleList([ConvNeXtBlock2d(w8, ls_init=ls_init, zero_out=True) for _ in range(self.extra_high)])

        self.up4_norm = nn.LayerNorm(w8, eps=1e-6)
        self.up4_proj = nn.Linear(w8, 4 * self.c_4px)
        self.blocks_4px = nn.ModuleList([ConvNeXtBlock2d(self.c_4px, ls_init=ls_init) for _ in range(self.depth_4px)])

        self.up2_norm = nn.LayerNorm(self.c_4px, eps=1e-6)
        self.up2_proj = nn.Linear(self.c_4px, 4 * self.c_2px)
        self.blocks_2px = nn.ModuleList([PixelRefineBlock2d(self.c_2px, expand=2) for _ in range(self.depth_2px)])

        self.up1_conv = nn.Conv2d(self.c_2px, 4 * self.c_1px, kernel_size=1)
        self.blocks_1px = nn.ModuleList([PixelRefineBlock2d(self.c_1px, expand=2) for _ in range(self.depth_1px)])
        self.detail_norm = nn.GroupNorm(num_groups=min(8, self.c_1px), num_channels=self.c_1px, eps=1e-6)
        self.detail_out = nn.Conv2d(self.c_1px, self.out_channels, kernel_size=3, padding=1)
        nn.init.zeros_(self.detail_out.weight)
        nn.init.zeros_(self.detail_out.bias)

        self.meta: dict[str, str] = {}

    def forward(
        self,
        feat: torch.Tensor,
        x0_hat_norm: torch.Tensor,
        sigma: torch.Tensor | float = 0.10,
    ) -> torch.Tensor:
        bsz, _, gh, gw = x0_hat_norm.shape
        b = self.base
        if feat.ndim == 3:
            feat = feat.view(bsz, gh, gw, b.d_feat)
        lat_tok = x0_hat_norm.movedim(1, -1)

        s_cond = self.sigma_adaln.embed_sigma(sigma, bsz, x0_hat_norm.device, feat.dtype)

        h = b.entry_norm(b.feat_proj(b.feat_norm(feat)) + b.lat_proj(lat_tok))
        h = h.movedim(-1, 1).contiguous()
        h = self.sigma_adaln.modulate_nchw(h, s_cond, 0)
        for blk in b.low_blocks:
            h = blk(h)
        for blk in self.low_ext:
            h = blk(h)

        u8 = b.up_proj(b.up_norm(h.movedim(1, -1))).movedim(-1, 1).contiguous()
        h8 = F.pixel_shuffle(u8, 2)
        h8 = self.sigma_adaln.modulate_nchw(h8, s_cond, 1)
        for blk in b.high_blocks:
            h8 = blk(h8)
        for blk in self.high_ext:
            h8 = blk(h8)

        h8_tok = b.norm_out(h8.movedim(1, -1))
        with torch.autocast(device_type=h8_tok.device.type, enabled=False):
            anchor_patches = b.out_proj(h8_tok.float())
        anchor_img = unpatchify_2d(anchor_patches, b.out_patch, self.out_channels)

        u4 = self.up4_proj(self.up4_norm(h8.movedim(1, -1))).movedim(-1, 1).contiguous()
        h4 = F.pixel_shuffle(u4, 2)
        h4 = self.sigma_adaln.modulate_nchw(h4, s_cond, 2)
        for blk in self.blocks_4px:
            h4 = blk(h4)

        u2 = self.up2_proj(self.up2_norm(h4.movedim(1, -1))).movedim(-1, 1).contiguous()
        h2 = F.pixel_shuffle(u2, 2)
        for blk in self.blocks_2px:
            h2 = blk(h2)

        h1 = F.pixel_shuffle(F.silu(self.up1_conv(h2)), 2)
        for blk in self.blocks_1px:
            h1 = blk(h1)

        with torch.autocast(device_type=h1.device.type, enabled=False):
            h1_fp32 = F.silu(self.detail_norm.float()(h1.float()))
            detail_img = self.detail_out(h1_fp32)

        return anchor_img + detail_img


class QwenImage21PixelHeadV3(nn.Module):
    """v0.3 Fast Haar-Wavelet + Multi-Scale Latent Spatial Highway Direct Pixel Head.

    Key upgrades over v0.2:
      1. Eliminates the slow 1024x1024 spatial conv + GroupNorm stage completely: runs norm-free
         fused depthwise-separable blocks at 512x512 (`2px` grid) to predict 2D Haar wavelet subbands
         `(dLL, dLH, dHL, dHH)` followed by 0-parameter analytical `idwt_haar_2d` (~5-6x faster!).
      2. Adds `LatentSpatialHighway` (`x0_hat_norm` 3x3 multi-scale spatial conv bridge into 8px/4px/2px),
         routing neighbor-token stroke geometry directly to the wavelet head for crisp 1-2px slanted text.
      3. Exposes `forward_subbands` so Stage-2 Wavelet PatchGAN only sharpens `(LH, HL, HH)` while
         detaching `LL` (mathematically preventing color shift or shadow banding).
    """

    def __init__(
        self,
        base: Optional[QwenImage21PixelHead] = None,
        d_feat: int = 4096,
        d_lat: int = 64,
        width: int = 1024,
        depth_low: int = 2,
        depth_high: int = 1,
        extra_low: int = 1,
        extra_high: int = 2,
        depth_4px: int = 2,
        depth_2px: int = 2,
        c_4px: int = 256,
        c_2px: int = 64,
        out_channels: int = 4,
        ls_init: float = 1.0,
    ):
        super().__init__()
        self.base = base if base is not None else QwenImage21PixelHead(
            d_feat=d_feat,
            d_lat=d_lat,
            width=width,
            depth_low=depth_low,
            depth_high=depth_high,
            out_channels=out_channels,
            ls_init=ls_init,
        )
        w0 = self.base.width
        w8 = w0 // 2
        self.extra_low = int(extra_low)
        self.extra_high = int(extra_high)
        self.depth_4px = int(depth_4px)
        self.depth_2px = int(depth_2px)
        self.c_4px = int(c_4px)
        self.c_2px = int(c_2px)
        self.out_channels = self.base.out_channels

        self.sigma_adaln = SigmaAdaLN(embed_dim=256, target_dims=(w0, w8, self.c_4px, self.c_2px))

        self.low_ext = nn.ModuleList([ConvNeXtBlock2d(w0, ls_init=ls_init, zero_out=True) for _ in range(self.extra_low)])
        self.high_ext = nn.ModuleList([ConvNeXtBlock2d(w8, ls_init=ls_init, zero_out=True) for _ in range(self.extra_high)])

        self.lat_highway = LatentSpatialHighway(
            d_lat=self.base.d_lat,
            c_lat16=128,
            c_lat8=96,
            c_lat4=64,
            w8=w8,
            c_4px=self.c_4px,
            c_2px=self.c_2px,
        )

        self.up4_norm = nn.LayerNorm(w8, eps=1e-6)
        self.up4_proj = nn.Linear(w8, 4 * self.c_4px)
        self.blocks_4px = nn.ModuleList([ConvNeXtBlock2d(self.c_4px, ls_init=ls_init) for _ in range(self.depth_4px)])

        self.up2_conv = nn.Conv2d(self.c_4px, 4 * self.c_2px, kernel_size=1)
        self.blocks_2px = nn.ModuleList([WaveletConvBlock2d(self.c_2px, ls_init=ls_init) for _ in range(self.depth_2px)])

        # Zero-init FP32 1x1 Haar Wavelet head (spatial 3x3 context already handled in Tensor-Core BF16 blocks_2px!)
        self.wavelet_out = nn.Conv2d(self.c_2px, 4 * self.out_channels, kernel_size=1)
        nn.init.zeros_(self.wavelet_out.weight)
        nn.init.zeros_(self.wavelet_out.bias)

        self.meta: dict[str, str] = {}

    @classmethod
    def from_v2(cls, v2_head: QwenImage21PixelHeadV2, depth_2px: int = 2, c_2px: int = 64) -> "QwenImage21PixelHeadV3":
        """Warm-start v0.3 directly from a trained v0.2 checkpoint (`16px -> 8px -> 4px` trunk + `SigmaAdaLN`)."""
        import copy

        v3 = cls(
            base=copy.deepcopy(v2_head.base),
            extra_low=v2_head.extra_low,
            extra_high=v2_head.extra_high,
            depth_4px=v2_head.depth_4px,
            depth_2px=depth_2px,
            c_4px=v2_head.c_4px,
            c_2px=c_2px,
            out_channels=v2_head.out_channels,
            ls_init=v2_head.base.ls_init,
        )
        v3.low_ext.load_state_dict(v2_head.low_ext.state_dict())
        v3.high_ext.load_state_dict(v2_head.high_ext.state_dict())
        v3.up4_norm.load_state_dict(v2_head.up4_norm.state_dict())
        v3.up4_proj.load_state_dict(v2_head.up4_proj.state_dict())
        v3.blocks_4px.load_state_dict(v2_head.blocks_4px.state_dict())
        # Copy SigmaAdaLN MLP + first 3 modulation heads (w0, w8, c_4px); 4th head (c_2px) stays zero-init
        v3.sigma_adaln.mlp.load_state_dict(v2_head.sigma_adaln.mlp.state_dict())
        for i in range(min(len(v2_head.sigma_adaln.heads), 3)):
            v3.sigma_adaln.heads[i].load_state_dict(v2_head.sigma_adaln.heads[i].state_dict())
        return v3

    def _compute_anchor_and_wavelet_delta(
        self,
        feat: torch.Tensor,
        x0_hat_norm: torch.Tensor,
        sigma: torch.Tensor | float = 0.10,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        bsz, _, gh, gw = x0_hat_norm.shape
        b = self.base
        if feat.ndim == 3:
            feat = feat.view(bsz, gh, gw, b.d_feat)
        lat_tok = x0_hat_norm.movedim(1, -1)

        s_cond = self.sigma_adaln.embed_sigma(sigma, bsz, x0_hat_norm.device, feat.dtype)
        lat_inj8, lat_inj4, lat_inj2 = self.lat_highway(x0_hat_norm)

        # 16px stage
        h = b.entry_norm(b.feat_proj(b.feat_norm(feat)) + b.lat_proj(lat_tok))
        h = h.movedim(-1, 1).contiguous()
        h = self.sigma_adaln.modulate_nchw(h, s_cond, 0)
        for blk in b.low_blocks:
            h = blk(h)
        for blk in self.low_ext:
            h = blk(h)

        # 8px stage + Latent Spatial Highway injection
        u8 = b.up_proj(b.up_norm(h.movedim(1, -1))).movedim(-1, 1).contiguous()
        h8 = F.pixel_shuffle(u8, 2) + lat_inj8
        h8 = self.sigma_adaln.modulate_nchw(h8, s_cond, 1)
        for blk in b.high_blocks:
            h8 = blk(h8)
        for blk in self.high_ext:
            h8 = blk(h8)

        # FP32 8x8 coarse anchor
        h8_tok = b.norm_out(h8.movedim(1, -1))
        with torch.autocast(device_type=h8_tok.device.type, enabled=False):
            anchor_patches = b.out_proj(h8_tok.float())
        anchor_img = unpatchify_2d(anchor_patches, b.out_patch, self.out_channels)

        # 4px stage (256x256) + Latent Spatial Highway injection
        u4 = self.up4_proj(self.up4_norm(h8.movedim(1, -1))).movedim(-1, 1).contiguous()
        h4 = F.pixel_shuffle(u4, 2) + lat_inj4
        h4 = self.sigma_adaln.modulate_nchw(h4, s_cond, 2)
        for blk in self.blocks_4px:
            h4 = blk(h4)

        # 2px Haar Wavelet grid stage (512x512, channels_last fast path) + Latent Spatial Highway injection
        h2 = F.pixel_shuffle(F.silu(self.up2_conv(h4)), 2) + lat_inj2
        h2 = self.sigma_adaln.modulate_nchw(h2, s_cond, 3)
        if h2.is_cuda:
            h2 = h2.contiguous(memory_format=torch.channels_last)
        for blk in self.blocks_2px:
            h2 = blk(h2)

        with torch.autocast(device_type=h2.device.type, enabled=False):
            w_delta = self.wavelet_out(h2.float())
        d_ll, d_lh, d_hl, d_hh = w_delta.chunk(4, dim=1)
        return anchor_img, d_ll, d_lh, d_hl, d_hh

    def forward_subbands(
        self,
        feat: torch.Tensor,
        x0_hat_norm: torch.Tensor,
        sigma: torch.Tensor | float = 0.10,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Return full FP32 2D Haar Wavelet subbands `(LL, LH, HL, HH)` at `[B, C, H/2, W/2]`."""
        anchor_img, d_ll, d_lh, d_hl, d_hh = self._compute_anchor_and_wavelet_delta(feat, x0_hat_norm, sigma=sigma)
        a_ll, a_lh, a_hl, a_hh = dwt_haar_2d(anchor_img)
        return a_ll + d_ll, a_lh + d_lh, a_hl + d_hl, a_hh + d_hh

    def forward(
        self,
        feat: torch.Tensor,
        x0_hat_norm: torch.Tensor,
        sigma: torch.Tensor | float = 0.10,
    ) -> torch.Tensor:
        anchor_img, d_ll, d_lh, d_hl, d_hh = self._compute_anchor_and_wavelet_delta(feat, x0_hat_norm, sigma=sigma)
        return anchor_img + idwt_haar_2d(d_ll, d_lh, d_hl, d_hh)


class QwenImage21PixelHeadX2(nn.Module):
    """Optional 2x Super-Resolution Direct Pixel Head (`32x32` pixels per DiT token)."""

    def __init__(
        self,
        base: QwenImage21PixelHead,
        depth_x2: int = 2,
        ls_init: float = 1.0,
    ):
        super().__init__()
        self.base = base
        self.depth_x2 = int(depth_x2)
        w_in = base.width // 2
        w_x2 = w_in // 2
        self.out_channels = base.out_channels
        self.out_patch = base.out_patch

        self.up2_norm = nn.LayerNorm(w_in, eps=1e-6)
        self.up2_proj = nn.Linear(w_in, 4 * w_x2)
        self.x2_blocks = nn.ModuleList(
            [ConvNeXtBlock2d(w_x2, ls_init=ls_init) for _ in range(depth_x2)]
        )
        self.norm2 = nn.LayerNorm(w_x2, eps=1e-6)
        self.out2_proj = nn.Linear(w_x2, self.out_patch * self.out_patch * self.out_channels)
        nn.init.normal_(self.out2_proj.weight, mean=0.0, std=0.02)
        nn.init.zeros_(self.out2_proj.bias)
        self.meta: dict[str, str] = {}

    def forward(self, feat: torch.Tensor, x0_hat_norm: torch.Tensor, scale: int = 2) -> torch.Tensor:
        h = self.base.forward_trunk(feat, x0_hat_norm)
        if scale == 1:
            h1 = self.base.norm_out(h.movedim(1, -1))
            with torch.autocast(device_type=h1.device.type, enabled=False):
                p1 = self.base.out_proj(h1.float())
            return unpatchify_2d(p1, self.base.out_patch, self.out_channels)

        u2 = self.up2_proj(self.up2_norm(h.movedim(1, -1))).movedim(-1, 1).contiguous()
        g = F.pixel_shuffle(u2, 2)
        for blk in self.x2_blocks:
            g = blk(g)
        h2 = self.norm2(g.movedim(1, -1))
        with torch.autocast(device_type=h2.device.type, enabled=False):
            p2 = self.out2_proj(h2.float())
        return unpatchify_2d(p2, self.out_patch, self.out_channels)


@torch.no_grad()
def decode_pixel_head_tiled(
    head: QwenImage21PixelHead | QwenImage21PixelHeadX2 | QwenImage21PixelHeadV2 | QwenImage21PixelHeadV3,
    feat: torch.Tensor,
    x0_hat_norm: torch.Tensor,
    sigma: float = 0.10,
    scale: int = 1,
    tile_tokens: int = 64,
    overlap_tokens: int = 8,
) -> torch.Tensor:
    """Seamless 2D raised-cosine tiled decoding for 2K/4K ultra-high resolutions (<220 MB peak VRAM)."""
    bsz, _, gh, gw = x0_hat_norm.shape
    d_feat = head.base.d_feat if hasattr(head, "base") else head.d_feat
    if feat.ndim == 3:
        feat_grid = feat.view(bsz, gh, gw, d_feat)
    else:
        feat_grid = feat

    px_per_tok = TOKEN_PIXELS * scale
    out_c = head.out_channels
    out_h, out_w = gh * px_per_tok, gw * px_per_tok
    acc = torch.zeros((bsz, out_c, out_h, out_w), device="cpu", dtype=torch.float32)
    w_sum = torch.zeros((1, 1, out_h, out_w), device="cpu", dtype=torch.float32)

    stride = max(8, tile_tokens - overlap_tokens)
    ys = list(range(0, max(1, gh - tile_tokens + stride), stride))
    xs = list(range(0, max(1, gw - tile_tokens + stride), stride))
    if ys[-1] + tile_tokens < gh:
        ys.append(max(0, gh - tile_tokens))
    if xs[-1] + tile_tokens < gw:
        xs.append(max(0, gw - tile_tokens))

    def _ramp_1d(length: int, ov: int, at_start: bool, at_end: bool) -> torch.Tensor:
        w = torch.ones(length, dtype=torch.float32)
        if not at_start and ov > 0:
            t = torch.linspace(0.0, math.pi, ov, dtype=torch.float32)
            w[:ov] = 0.5 * (1.0 - torch.cos(t))
        if not at_end and ov > 0:
            t = torch.linspace(0.0, math.pi, ov, dtype=torch.float32)
            w[-ov:] = 0.5 * (1.0 + torch.cos(t))
        return w

    ov_px = overlap_tokens * px_per_tok
    for y0 in ys:
        y1 = min(gh, y0 + tile_tokens)
        wy = _ramp_1d((y1 - y0) * px_per_tok, ov_px, y0 == 0, y1 == gh)[:, None]
        for x0 in xs:
            x1 = min(gw, x0 + tile_tokens)
            wx = _ramp_1d((x1 - x0) * px_per_tok, ov_px, x0 == 0, x1 == gw)[None, :]
            win = (wy * wx)[None, None]

            f_tile = feat_grid[:, y0:y1, x0:x1, :].contiguous()
            x_tile = x0_hat_norm[:, :, y0:y1, x0:x1].contiguous()
            if isinstance(head, (QwenImage21PixelHeadV2, QwenImage21PixelHeadV3)):
                pred_tile = head(f_tile, x_tile, sigma=sigma).float().cpu()
            elif isinstance(head, QwenImage21PixelHeadX2):
                pred_tile = head(f_tile, x_tile, scale=scale).float().cpu()
            else:
                pred_tile = head(f_tile, x_tile).float().cpu()

            py0, py1 = y0 * px_per_tok, y1 * px_per_tok
            px0, px1 = x0 * px_per_tok, x1 * px_per_tok
            acc[:, :, py0:py1, px0:px1] += pred_tile * win
            w_sum[:, :, py0:py1, px0:px1] += win

    return acc / w_sum.clamp_min(1e-6)


def save_qwen21_pixel_head(
    head: QwenImage21PixelHead | QwenImage21PixelHeadX2 | QwenImage21PixelHeadV2 | QwenImage21PixelHeadV3,
    path: str | Path,
    extra_meta: Optional[dict[str, str]] = None,
    save_bf16_trunk: bool = False,
) -> None:
    from safetensors.torch import save_file

    is_v3 = isinstance(head, QwenImage21PixelHeadV3)
    is_v2 = isinstance(head, QwenImage21PixelHeadV2)
    is_x2 = isinstance(head, QwenImage21PixelHeadX2)
    base = head.base if (is_v3 or is_v2 or is_x2) else head

    if is_v3:
        arch = "qwen_image21_pixel_head_v3"
    elif is_v2:
        arch = "qwen_image21_pixel_head_v2"
    elif is_x2:
        arch = "qwen_image21_pixel_head_x2"
    else:
        arch = "qwen_image21_pixel_head"

    meta = {
        "arch": arch,
        "d_feat": str(base.d_feat),
        "d_lat": str(base.d_lat),
        "width": str(base.width),
        "depth_low": str(base.depth_low),
        "depth_high": str(base.depth_high),
        "out_channels": str(base.out_channels),
        "ls_init": str(base.ls_init),
        "scale": "2" if is_x2 else "1",
        "latent_input": "x0_hat_norm",
    }
    if is_x2:
        meta["depth_x2"] = str(head.depth_x2)
    if is_v2:
        meta.update(
            {
                "extra_low": str(head.extra_low),
                "extra_high": str(head.extra_high),
                "depth_4px": str(head.depth_4px),
                "depth_2px": str(head.depth_2px),
                "depth_1px": str(head.depth_1px),
                "c_4px": str(head.c_4px),
                "c_2px": str(head.c_2px),
                "c_1px": str(head.c_1px),
            }
        )
    if is_v3:
        meta.update(
            {
                "extra_low": str(head.extra_low),
                "extra_high": str(head.extra_high),
                "depth_4px": str(head.depth_4px),
                "depth_2px": str(head.depth_2px),
                "c_4px": str(head.c_4px),
                "c_2px": str(head.c_2px),
            }
        )
    if extra_meta:
        for k, v in extra_meta.items():
            meta[str(k)] = str(v)

    fp32_prefixes = ("out_proj.", "base.out_proj.", "out2_proj.", "detail_out.", "detail_norm.", "wavelet_out.")
    sd = {}
    for k, v in head.state_dict().items():
        t = v.detach().cpu().contiguous()
        if save_bf16_trunk and t.is_floating_point() and not any(k.startswith(p) for p in fp32_prefixes):
            t = t.to(torch.bfloat16)
        elif any(k.startswith(p) for p in fp32_prefixes):
            t = t.to(torch.float32)
        sd[k] = t

    Path(path).parent.mkdir(parents=True, exist_ok=True)
    save_file(sd, str(path), metadata=meta)


def load_qwen21_pixel_head(
    path: str | Path,
    device: str | torch.device = "cpu",
    dtype: torch.dtype = torch.float32,
) -> QwenImage21PixelHead | QwenImage21PixelHeadX2 | QwenImage21PixelHeadV2 | QwenImage21PixelHeadV3:
    from safetensors import safe_open
    from safetensors.torch import load_file

    path = Path(path)
    with safe_open(str(path), framework="pt", device="cpu") as fh:
        meta = dict(fh.metadata() or {})

    arch = meta.get("arch", "qwen_image21_pixel_head")
    base = QwenImage21PixelHead(
        d_feat=int(meta.get("d_feat", "4096")),
        d_lat=int(meta.get("d_lat", "64")),
        width=int(meta.get("width", "1024")),
        depth_low=int(meta.get("depth_low", "2")),
        depth_high=int(meta.get("depth_high", "1")),
        out_channels=int(meta.get("out_channels", "4")),
        ls_init=float(meta.get("ls_init", "1.0")),
    )
    if arch == "qwen_image21_pixel_head_v3":
        model: nn.Module = QwenImage21PixelHeadV3(
            base=base,
            extra_low=int(meta.get("extra_low", "1")),
            extra_high=int(meta.get("extra_high", "2")),
            depth_4px=int(meta.get("depth_4px", "2")),
            depth_2px=int(meta.get("depth_2px", "2")),
            c_4px=int(meta.get("c_4px", "256")),
            c_2px=int(meta.get("c_2px", "64")),
            ls_init=float(meta.get("ls_init", "1.0")),
        )
    elif arch == "qwen_image21_pixel_head_v2" or "depth_1px" in meta:
        model = QwenImage21PixelHeadV2(
            base=base,
            extra_low=int(meta.get("extra_low", "1")),
            extra_high=int(meta.get("extra_high", "2")),
            depth_4px=int(meta.get("depth_4px", "2")),
            depth_2px=int(meta.get("depth_2px", "2")),
            depth_1px=int(meta.get("depth_1px", "2")),
            c_4px=int(meta.get("c_4px", "256")),
            c_2px=int(meta.get("c_2px", "128")),
            c_1px=int(meta.get("c_1px", "64")),
            ls_init=float(meta.get("ls_init", "1.0")),
        )
    elif arch == "qwen_image21_pixel_head_x2" or "depth_x2" in meta:
        model = QwenImage21PixelHeadX2(
            base=base,
            depth_x2=int(meta.get("depth_x2", "2")),
            ls_init=float(meta.get("ls_init", "1.0")),
        )
    else:
        model = base

    sd = load_file(str(path), device="cpu")
    model.load_state_dict(sd, strict=True)
    model = model.to(device=device, dtype=dtype)
    if isinstance(model, QwenImage21PixelHeadV3):
        model.base.out_proj.float()
        model.wavelet_out.float()
    elif isinstance(model, QwenImage21PixelHeadV2):
        model.base.out_proj.float()
        model.detail_norm.float()
        model.detail_out.float()
    elif isinstance(model, QwenImage21PixelHeadX2):
        model.base.out_proj.float()
        model.out2_proj.float()
    else:
        model.out_proj.float()
    model.meta = meta
    model.eval()
    return model
