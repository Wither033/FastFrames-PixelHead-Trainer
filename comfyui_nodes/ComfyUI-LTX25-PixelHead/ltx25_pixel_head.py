"""Clean-room PyTorch implementation of 3D Causal Video Direct Pixel Head for LTX-2.5 (`LTXAVModel` / `LTXVModel`).

License: MIT
Architecture v2 (Overcomplete Spatiotemporal Shuffle, zero linear rank bottleneck) follows FastFrames (arXiv:2611.23226):
  - Frozen LTX-2.5 DiT (`LTXAVModel` / `LTXVModel`, inner_dim=4096, video_channels=128, vae_scale_factors=(8, 32, 32)).
  - Pre-hook taps video hidden states `f_vid` (`[B, T_lat*H_lat*W_lat, 4096]`, post-`norm_out` AdaLN, pre-`proj_out`)
    plus the exact denoised video latent `x0_hat_vid` (`[B, 128, T_lat, H_lat, W_lat]`).
  - Progressive 3-stage Spatiotemporal 3D PixelShuffle trunk (`(8t, 32px) -> (8t, 16px) -> (4t, 8px) -> (2t, 4px)`):
      1. `feat_norm(4096) -> Linear(4096 -> w32, bias=False)` + `Linear(128 -> w32, bias=True)` -> `entry_norm`.
      2. `depth_s32` CausalConvNeXt3d blocks at `(T_lat, H_lat, W_lat)` (32px spatial, 8f temporal).
      3. `(1, 2, 2)` Spatial PixelShuffle -> `depth_s16` CausalConvNeXt3d blocks at `(T_lat, 2*H_lat, 2*W_lat)` (16px, 8f).
      4. `(2, 2, 2)` Spatiotemporal PixelShuffle -> `depth_s8` CausalConvNeXt3d blocks at `(2*T_lat, 4*H_lat, 4*W_lat)` (8px, 4f).
      5. `(2, 2, 2)` Spatiotemporal PixelShuffle -> `depth_s4` CausalConvNeXt3d blocks at `(4*T_lat, 8*H_lat, 8*W_lat)` (4px, 2f).
      6. `norm_out` + overcomplete FP32 Linear (`w4=256 -> 2 * 4 * 4 * 3 = 96`, 2.67x overcomplete!) -> `[B, 3, 8*T_lat, 32*H_lat, 32*W_lat]`.
      7. Causal temporal frame alignment: drops the leading 7 pre-frames -> `8*T_lat - 7` frames
         (`T_lat=1 -> 1f`, `T_lat=3 -> 17f`, `T_lat=13 -> 97f`, matching `CausalDiffusionVAE` 1:1).
"""

from __future__ import annotations

import math
from pathlib import Path
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F


FINAL_TEMPORAL_PATCH = 2   # After two 2x temporal shuffles: 8 / 4 = 2 frames
FINAL_SPATIAL_PATCH = 4    # After three 2x spatial shuffles: 32 / 8 = 4 pixels


class CausalDepthwiseConv3d(nn.Module):
    """3D depthwise conv with left-only causal temporal padding and symmetric spatial padding."""

    def __init__(self, dim: int, kt: int = 3, kh: int = 3, kw: int = 3):
        super().__init__()
        self.pad_t = kt - 1
        self.pad_h = kh // 2
        self.pad_w = kw // 2
        self.conv = nn.Conv3d(dim, dim, kernel_size=(kt, kh, kw), padding=0, groups=dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [B, C, T, H, W]; replicate first temporal frame for causal start stability
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
        # x: [B, C, T, H, W]
        h = self.dw(x).movedim(1, -1)  # [B, T, H, W, C]
        h = self.fc2(F.gelu(self.fc1(self.norm(h))))
        h = (h * self.gamma).movedim(-1, 1).contiguous()
        return x + h


def spatial_pixel_shuffle_3d(x: torch.Tensor, norm: nn.LayerNorm, proj: nn.Linear, scale: int = 2) -> torch.Tensor:
    """Channel LayerNorm + Linear + exact per-frame `(1, scale, scale)` PixelShuffle on `[B, C, T, H, W]`."""
    b, _, t, h, w = x.shape
    u = proj(norm(x.movedim(1, -1)))  # [B, T, H, W, C_out * scale * scale]
    c_out = u.shape[-1] // (scale * scale)
    u = u.view(b, t, h, w, c_out, scale, scale)
    return u.permute(0, 4, 1, 2, 5, 3, 6).contiguous().view(b, c_out, t, h * scale, w * scale)


def spatiotemporal_pixel_shuffle_3d(
    x: torch.Tensor, norm: nn.LayerNorm, proj: nn.Linear, pt: int = 2, ps: int = 2
) -> torch.Tensor:
    """Channel LayerNorm + Linear + exact 3D `(pt, ps, ps)` PixelShuffle on `[B, C, T, H, W]`."""
    b, _, t, h, w = x.shape
    u = proj(norm(x.movedim(1, -1)))  # [B, T, H, W, C_out * pt * ps * ps]
    c_out = u.shape[-1] // (pt * ps * ps)
    u = u.view(b, t, h, w, c_out, pt, ps, ps)
    return u.permute(0, 4, 1, 5, 2, 6, 3, 7).contiguous().view(b, c_out, t * pt, h * ps, w * ps)


def unpatchify_3d_ltx(
    tokens: torch.Tensor, pt: int = FINAL_TEMPORAL_PATCH, ph: int = FINAL_SPATIAL_PATCH, pw: int = FINAL_SPATIAL_PATCH, out_channels: int = 3
) -> torch.Tensor:
    """Convert `[B, T_grid, H_grid, W_grid, pt * ph * pw * C]` -> `[B, C, T_grid * pt, H_grid * ph, W_grid * pw]`."""
    b, tg, gh, gw, _ = tokens.shape
    x = tokens.view(b, tg, gh, gw, pt, ph, pw, out_channels)
    x = x.permute(0, 7, 1, 4, 2, 5, 3, 6).contiguous()
    return x.view(b, out_channels, tg * pt, gh * ph, gw * pw)


class LTX25VideoPixelHead(nn.Module):
    """Overcomplete 3D Causal Video Pixel Head for LTX-2.5 (`8*T_lat - 7` frames, `32x32` spatial per token)."""

    def __init__(
        self,
        d_feat: int = 4096,
        d_lat: int = 128,
        width: int = 768,
        depth_s32: int = 2,
        depth_s16: int = 2,
        depth_s8: int = 2,
        depth_s4: int = 1,
        out_channels: int = 3,
        ls_init: float = 1.0,
    ):
        super().__init__()
        if width % 32 != 0:
            raise ValueError(f"width must be divisible by 32, got {width}")
        self.d_feat = int(d_feat)
        self.d_lat = int(d_lat)
        self.width = int(width)
        self.depth_s32 = int(depth_s32)
        self.depth_s16 = int(depth_s16)
        self.depth_s8 = int(depth_s8)
        self.depth_s4 = int(depth_s4)
        self.out_channels = int(out_channels)
        self.ls_init = float(ls_init)

        w32 = width
        w16 = max(128, ((width * 2) // 3 // 32) * 32)   # 512 when width=768
        w8 = max(128, (width // 2 // 32) * 32)          # 384 when width=768
        w4 = max(128, (width // 3 // 32) * 32)          # 256 when width=768 (>= 96 out_dim!)

        self.feat_norm = nn.LayerNorm(self.d_feat, eps=1e-6)
        self.feat_proj = nn.Linear(self.d_feat, w32, bias=False)
        self.lat_proj = nn.Linear(self.d_lat, w32, bias=True)
        self.entry_norm = nn.LayerNorm(w32, eps=1e-6)

        self.s32_blocks = nn.ModuleList([CausalConvNeXtBlock3d(w32, ls_init=ls_init) for _ in range(depth_s32)])
        self.up1_norm = nn.LayerNorm(w32, eps=1e-6)
        self.up1_proj = nn.Linear(w32, 4 * w16)  # (1, 2, 2) spatial shuffle

        self.s16_blocks = nn.ModuleList([CausalConvNeXtBlock3d(w16, ls_init=ls_init) for _ in range(depth_s16)])
        self.up2_norm = nn.LayerNorm(w16, eps=1e-6)
        self.up2_proj = nn.Linear(w16, 8 * w8)   # (2, 2, 2) spatiotemporal shuffle

        self.s8_blocks = nn.ModuleList([CausalConvNeXtBlock3d(w8, ls_init=ls_init) for _ in range(depth_s8)])
        self.up3_norm = nn.LayerNorm(w8, eps=1e-6)
        self.up3_proj = nn.Linear(w8, 8 * w4)    # (2, 2, 2) spatiotemporal shuffle

        self.s4_blocks = nn.ModuleList([CausalConvNeXtBlock3d(w4, ls_init=ls_init) for _ in range(depth_s4)])
        self.norm_out = nn.LayerNorm(w4, eps=1e-6)
        out_dim = FINAL_TEMPORAL_PATCH * FINAL_SPATIAL_PATCH * FINAL_SPATIAL_PATCH * self.out_channels  # 2*4*4*3 = 96
        self.out_proj = nn.Linear(w4, out_dim)
        nn.init.normal_(self.out_proj.weight, mean=0.0, std=0.02)
        nn.init.zeros_(self.out_proj.bias)

        self.meta: dict[str, str] = {}

    def causal_receptive_t_lat(self) -> int:
        """Exact causal temporal lookback in latent frames (`T_lat`) for zero-seam chunked decoding."""
        return (
            2 * self.depth_s32
            + 2 * self.depth_s16
            + math.ceil((2 * self.depth_s8) / 2)
            + math.ceil((2 * self.depth_s4) / 4)
        )

    def forward_raw(self, feat: torch.Tensor, x0_hat_vid: torch.Tensor) -> torch.Tensor:
        """Return uncropped `[B, 3, 8*T_lat, 32*H_lat, 32*W_lat]` before dropping leading 7 pre-frames."""
        b, _, t_lat, h_lat, w_lat = x0_hat_vid.shape
        if feat.ndim == 3:
            feat = feat.view(b, t_lat, h_lat, w_lat, self.d_feat)
        lat_tok = x0_hat_vid.movedim(1, -1)  # [B, T_lat, H_lat, W_lat, 128]

        h = self.entry_norm(self.feat_proj(self.feat_norm(feat)) + self.lat_proj(lat_tok))
        h = h.movedim(-1, 1).contiguous()  # [B, w32, T_lat, H_lat, W_lat]

        for blk in self.s32_blocks:
            h = blk(h)
        h = spatial_pixel_shuffle_3d(h, self.up1_norm, self.up1_proj, scale=2)
        for blk in self.s16_blocks:
            h = blk(h)
        h = spatiotemporal_pixel_shuffle_3d(h, self.up2_norm, self.up2_proj, pt=2, ps=2)
        for blk in self.s8_blocks:
            h = blk(h)
        h = spatiotemporal_pixel_shuffle_3d(h, self.up3_norm, self.up3_proj, pt=2, ps=2)
        for blk in self.s4_blocks:
            h = blk(h)

        h_tok = self.norm_out(h.movedim(1, -1))  # [B, 4*T_lat, 8*H_lat, 8*W_lat, w4]
        with torch.autocast(device_type=h_tok.device.type, enabled=False):
            patches = self.out_proj(h_tok.float())
        return unpatchify_3d_ltx(patches, FINAL_TEMPORAL_PATCH, FINAL_SPATIAL_PATCH, FINAL_SPATIAL_PATCH, self.out_channels)

    def forward(self, feat: torch.Tensor, x0_hat_vid: torch.Tensor, align_causal: bool = True) -> torch.Tensor:
        """Return `[B, 3, 8*T_lat - 7, 32*H_lat, 32*W_lat]` in `[-1, 1]`."""
        raw = self.forward_raw(feat, x0_hat_vid)
        if align_causal:
            return raw[:, :, 7:, :, :]
        return raw

    @torch.no_grad()
    def decode_chunked(
        self,
        feat: torch.Tensor,
        x0_hat_vid: torch.Tensor,
        chunk_t: int = 8,
        overlap_t: Optional[int] = None,
    ) -> torch.Tensor:
        """Low-VRAM bit-exact causal chunked decode for long 97~241+ frame LTX-2.5 videos."""
        b, _, t_lat, h_lat, w_lat = x0_hat_vid.shape
        if feat.ndim == 3:
            feat = feat.view(b, t_lat, h_lat, w_lat, self.d_feat)
        if t_lat <= chunk_t:
            return self.forward(feat, x0_hat_vid, align_causal=True)

        ov = self.causal_receptive_t_lat() if overlap_t is None else int(overlap_t)
        pieces = []
        pos = 0
        while pos < t_lat:
            end = min(t_lat, pos + chunk_t)
            ctx_start = max(0, pos - ov)
            f_sl = feat[:, ctx_start:end]
            x_sl = x0_hat_vid[:, :, ctx_start:end]
            raw_sl = self.forward_raw(f_sl, x_sl)  # [B, 3, 8*(end - ctx_start), H, W]
            if pos == 0:
                pieces.append(raw_sl[:, :, 7:].cpu())
            else:
                keep_lat = end - pos
                pieces.append(raw_sl[:, :, -keep_lat * 8 :].cpu())
            pos = end
        return torch.cat(pieces, dim=2)


def save_ltx25_pixel_head(
    head: LTX25VideoPixelHead,
    path: str | Path,
    extra_meta: Optional[dict[str, str]] = None,
) -> None:
    from safetensors.torch import save_file

    meta = {
        "arch": "ltx25_video_pixel_head_v2",
        "d_feat": str(head.d_feat),
        "d_lat": str(head.d_lat),
        "width": str(head.width),
        "depth_s32": str(head.depth_s32),
        "depth_s16": str(head.depth_s16),
        "depth_s8": str(head.depth_s8),
        "depth_s4": str(head.depth_s4),
        "out_channels": str(head.out_channels),
        "ls_init": str(head.ls_init),
        "latent_input": "x0_hat_vid",
    }
    if extra_meta:
        for k, v in extra_meta.items():
            meta[str(k)] = str(v)

    sd = {k: v.detach().cpu().contiguous() for k, v in head.state_dict().items()}
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    save_file(sd, str(path), metadata=meta)


def load_ltx25_pixel_head(
    path: str | Path,
    device: str | torch.device = "cpu",
    dtype: torch.dtype = torch.float32,
) -> LTX25VideoPixelHead:
    from safetensors import safe_open
    from safetensors.torch import load_file

    path = Path(path)
    with safe_open(str(path), framework="pt", device="cpu") as fh:
        meta = dict(fh.metadata() or {})

    model = LTX25VideoPixelHead(
        d_feat=int(meta.get("d_feat", "4096")),
        d_lat=int(meta.get("d_lat", "128")),
        width=int(meta.get("width", "768")),
        depth_s32=int(meta.get("depth_s32", "2")),
        depth_s16=int(meta.get("depth_s16", "2")),
        depth_s8=int(meta.get("depth_s8", "2")),
        depth_s4=int(meta.get("depth_s4", "1")),
        out_channels=int(meta.get("out_channels", "3")),
        ls_init=float(meta.get("ls_init", "1.0")),
    )
    sd = load_file(str(path), device="cpu")
    model.load_state_dict(sd, strict=True)
    model = model.to(device=device, dtype=dtype)
    model.out_proj.float()
    model.meta = meta
    model.eval()
    return model
