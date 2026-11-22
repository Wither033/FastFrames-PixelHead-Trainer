"""Clean-room PyTorch implementation of Direct Pixel Readout Head for FLUX.3-Image-Klein.

License: MIT
Architecture follows the public mathematical formulation of FastFrames (arXiv:2611.23226):
  - Frozen FLUX.3-Image-Klein DiT (`comfy.ldm.flux3.model.Flux3`, hidden_size=3072, in_channels=128, 16x spatial).
  - Pre-hook taps `f` (`[B, H_lat*W_lat, 3072]`, post-`modulated_norm`, pre-`final_layer["image"].linear`)
    plus the exact denoised latent `x0_hat_norm` (`[B, 128, H_lat, W_lat]`, sigma-invariant across FlowMap & Base).
  - Feature LayerNorm + bridge linear (`3072 -> width`, bias=False) + latent linear (`128 -> width`, bias=True)
    + entry LayerNorm at 16px token grid.
  - `depth_low` ConvNeXt2d blocks at 16px token grid -> 2x PixelShuffle -> `depth_high` ConvNeXt2d blocks at 8px grid.
  - Final LayerNorm + FP32 Linear (`width//2 -> 8*8*out_channels`) producing RGB in `[-1, 1]`.
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F


TOKEN_PIXELS = 16  # Flux2/Flux3 VAE spatial downscale is 16x, DiT patch is 1x1


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
        h = self.dw(x).movedim(1, -1)
        h = self.fc2(F.gelu(self.fc1(self.norm(h))))
        h = (h * self.gamma).movedim(-1, 1)
        return x + h


def unpatchify_2d(tokens: torch.Tensor, patch: int, out_channels: int) -> torch.Tensor:
    """Convert `[B, gh, gw, patch * patch * C]` -> `[B, C, gh * patch, gw * patch]`."""
    b, gh, gw, _ = tokens.shape
    x = tokens.view(b, gh, gw, patch, patch, out_channels)
    x = x.permute(0, 5, 1, 3, 2, 4).contiguous()
    return x.view(b, out_channels, gh * patch, gw * patch)


class Flux3PixelHead(nn.Module):
    """1x Direct Pixel Head for FLUX.3-Image-Klein (`16x16` RGB pixels per DiT token)."""

    def __init__(
        self,
        d_feat: int = 3072,
        d_lat: int = 128,
        width: int = 1024,
        depth_low: int = 2,
        depth_high: int = 1,
        out_channels: int = 3,
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
        self.up_proj = nn.Linear(width, 4 * w_high)
        self.high_blocks = nn.ModuleList(
            [ConvNeXtBlock2d(w_high, ls_init=ls_init) for _ in range(depth_high)]
        )
        self.norm_out = nn.LayerNorm(w_high, eps=1e-6)
        self.out_proj = nn.Linear(w_high, self.out_patch * self.out_patch * self.out_channels)
        nn.init.normal_(self.out_proj.weight, mean=0.0, std=0.02)
        nn.init.zeros_(self.out_proj.bias)

        self.meta: dict[str, str] = {}

    def forward_trunk(self, feat: torch.Tensor, x0_hat_norm: torch.Tensor) -> torch.Tensor:
        """Run shared trunk up to `high_blocks` (`[B, width//2, 2*gh, 2*gw]`)."""
        b, _, gh, gw = x0_hat_norm.shape
        if feat.ndim == 3:
            feat = feat.view(b, gh, gw, self.d_feat)
        lat_tok = x0_hat_norm.movedim(1, -1)  # [B, gh, gw, d_lat]

        h = self.entry_norm(self.feat_proj(self.feat_norm(feat)) + self.lat_proj(lat_tok))
        h = h.movedim(-1, 1).contiguous()
        for blk in self.low_blocks:
            h = blk(h)

        u = self.up_proj(self.up_norm(h.movedim(1, -1))).movedim(-1, 1).contiguous()
        h = F.pixel_shuffle(u, 2)
        for blk in self.high_blocks:
            h = blk(h)
        return h

    def forward(self, feat: torch.Tensor, x0_hat_norm: torch.Tensor) -> torch.Tensor:
        """Return `[B, out_channels, gh*16, gw*16]` in `[-1, 1]`."""
        h = self.forward_trunk(feat, x0_hat_norm)
        h_tok = self.norm_out(h.movedim(1, -1))
        with torch.autocast(device_type=h_tok.device.type, enabled=False):
            patches = self.out_proj(h_tok.float())
        return unpatchify_2d(patches, self.out_patch, self.out_channels)


class Flux3PixelHeadX2(nn.Module):
    """Optional 2x Super-Resolution Direct Pixel Head (`32x32` pixels per DiT token)."""

    def __init__(
        self,
        base: Flux3PixelHead,
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


def save_flux3_pixel_head(
    head: Flux3PixelHead | Flux3PixelHeadX2,
    path: str | Path,
    extra_meta: Optional[dict[str, str]] = None,
) -> None:
    from safetensors.torch import save_file

    is_x2 = isinstance(head, Flux3PixelHeadX2)
    base = head.base if is_x2 else head
    meta = {
        "arch": "flux3_pixel_head_x2" if is_x2 else "flux3_pixel_head",
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
    if extra_meta:
        for k, v in extra_meta.items():
            meta[str(k)] = str(v)

    sd = {k: v.detach().cpu().contiguous() for k, v in head.state_dict().items()}
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    save_file(sd, str(path), metadata=meta)


def load_flux3_pixel_head(
    path: str | Path,
    device: str | torch.device = "cpu",
    dtype: torch.dtype = torch.float32,
) -> Flux3PixelHead | Flux3PixelHeadX2:
    from safetensors import safe_open
    from safetensors.torch import load_file

    path = Path(path)
    with safe_open(str(path), framework="pt", device="cpu") as fh:
        meta = dict(fh.metadata() or {})

    arch = meta.get("arch", "flux3_pixel_head")
    base = Flux3PixelHead(
        d_feat=int(meta.get("d_feat", "3072")),
        d_lat=int(meta.get("d_lat", "128")),
        width=int(meta.get("width", "1024")),
        depth_low=int(meta.get("depth_low", "2")),
        depth_high=int(meta.get("depth_high", "1")),
        out_channels=int(meta.get("out_channels", "3")),
        ls_init=float(meta.get("ls_init", "1.0")),
    )
    if arch == "flux3_pixel_head_x2" or "depth_x2" in meta:
        model: nn.Module = Flux3PixelHeadX2(
            base=base,
            depth_x2=int(meta.get("depth_x2", "2")),
            ls_init=float(meta.get("ls_init", "1.0")),
        )
    else:
        model = base

    sd = load_file(str(path), device="cpu")
    model.load_state_dict(sd, strict=True)
    model = model.to(device=device, dtype=dtype)
    if isinstance(model, Flux3PixelHeadX2):
        model.base.out_proj.float()
        model.out2_proj.float()
    else:
        model.out_proj.float()
    model.meta = meta
    model.eval()
    return model
