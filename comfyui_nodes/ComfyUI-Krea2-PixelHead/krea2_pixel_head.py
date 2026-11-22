"""Clean-room PyTorch implementation of Direct Pixel Readout Head for Krea-2 (`SingleStreamDiT`).

License: MIT
Architecture follows the public mathematical formulation of FastFrames (arXiv:2611.23226):
  - Frozen Krea-2 DiT (`comfy.ldm.krea2.model.SingleStreamDiT`, features=6144, patch=2, channels=16, 8x VAE -> 16px/token).
  - Pre-hook taps target image hidden states `f` (`[B, gh*gw, 6144]`, post-AdaLN norm, pre-`last.linear`)
    plus the exact denoised normalized latent `x0_hat_norm` (`[B, 16, H_lat, W_lat]` or `[B, 16, 1, H_lat, W_lat]`),
    rearranged into `2x2` spatial patches (`d_lat = 16 * 2 * 2 = 64`).
  - Feature LayerNorm + bias-free bridge linear (`6144 -> width=1024`) + latent linear (`64 -> width=1024`)
    + entry LayerNorm at 16px token grid.
  - `depth_low` ConvNeXt2d blocks at 16px token grid -> 2x PixelShuffle -> `depth_high` ConvNeXt2d blocks at 8px grid.
  - Final LayerNorm + FP32 Linear (`width//2 -> 8*8*3`) producing RGB in `[-1, 1]`.
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F


TOKEN_PIXELS = 16  # Krea-2 VAE is 8x spatial, DiT patch is 2x2 -> 16x16 pixels per token


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


def patchify_krea2_latent(x0_hat_norm: torch.Tensor, patch: int = 2) -> tuple[torch.Tensor, int, int]:
    """Convert `[B, 16, (1,) H_lat, W_lat]` -> `([B, gh, gw, 64], orig_px_h, orig_px_w)`."""
    if x0_hat_norm.ndim == 5:
        x0_hat_norm = x0_hat_norm[:, :, 0]
    b, c, h_lat, w_lat = x0_hat_norm.shape
    orig_px_h, orig_px_w = h_lat * 8, w_lat * 8
    pad_h = h_lat % patch
    pad_w = w_lat % patch
    if pad_h or pad_w:
        x0_hat_norm = F.pad(x0_hat_norm, (0, pad_w, 0, pad_h))
    gh, gw = x0_hat_norm.shape[2] // patch, x0_hat_norm.shape[3] // patch
    lat_tok = (
        x0_hat_norm.view(b, c, gh, patch, gw, patch)
        .permute(0, 2, 4, 1, 3, 5)
        .contiguous()
        .view(b, gh, gw, c * patch * patch)
    )
    return lat_tok, orig_px_h, orig_px_w


class Krea2PixelHead(nn.Module):
    """1x Bridged Direct Pixel Head for Krea-2 (`16x16` RGB pixels per DiT token)."""

    def __init__(
        self,
        d_feat: int = 6144,
        d_lat: int = 64,
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

    def forward_trunk(self, feat: torch.Tensor, x0_hat_norm: torch.Tensor) -> tuple[torch.Tensor, int, int]:
        """Accepts `x0_hat_norm` as either `[B, 16, (1,) H_lat, W_lat]` or pre-patched `[B, 64, gh, gw]`."""
        if x0_hat_norm.ndim == 4 and x0_hat_norm.shape[1] == self.d_lat:
            b, _, gh, gw = x0_hat_norm.shape
            lat_tok = x0_hat_norm.movedim(1, -1)
            orig_px_h, orig_px_w = gh * 16, gw * 16
        else:
            lat_tok, orig_px_h, orig_px_w = patchify_krea2_latent(x0_hat_norm)
            b, gh, gw, _ = lat_tok.shape

        if feat.ndim == 3:
            feat = feat.view(b, gh, gw, self.d_feat)

        h = self.entry_norm(self.feat_proj(self.feat_norm(feat)) + self.lat_proj(lat_tok))
        h = h.movedim(-1, 1).contiguous()
        for blk in self.low_blocks:
            h = blk(h)

        u = self.up_proj(self.up_norm(h.movedim(1, -1))).movedim(-1, 1).contiguous()
        h = F.pixel_shuffle(u, 2)
        for blk in self.high_blocks:
            h = blk(h)
        return h, orig_px_h, orig_px_w

    def forward(self, feat: torch.Tensor, x0_hat_norm: torch.Tensor) -> torch.Tensor:
        """Return `[B, out_channels, H_px, W_px]` in `[-1, 1]`."""
        h, orig_px_h, orig_px_w = self.forward_trunk(feat, x0_hat_norm)
        h_tok = self.norm_out(h.movedim(1, -1))
        with torch.autocast(device_type=h_tok.device.type, enabled=False):
            patches = self.out_proj(h_tok.float())
        out = unpatchify_2d(patches, self.out_patch, self.out_channels)
        return out[:, :, :orig_px_h, :orig_px_w]


class Krea2PixelHeadX2(nn.Module):
    """Optional 2x Super-Resolution Direct Pixel Head for Krea-2 (`32x32` pixels per DiT token)."""

    def __init__(
        self,
        base: Krea2PixelHead,
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
        h, orig_px_h, orig_px_w = self.base.forward_trunk(feat, x0_hat_norm)
        if scale == 1:
            h1 = self.base.norm_out(h.movedim(1, -1))
            with torch.autocast(device_type=h1.device.type, enabled=False):
                p1 = self.base.out_proj(h1.float())
            out1 = unpatchify_2d(p1, self.base.out_patch, self.out_channels)
            return out1[:, :, :orig_px_h, :orig_px_w]

        u2 = self.up2_proj(self.up2_norm(h.movedim(1, -1))).movedim(-1, 1).contiguous()
        g = F.pixel_shuffle(u2, 2)
        for blk in self.x2_blocks:
            g = blk(g)
        h2 = self.norm2(g.movedim(1, -1))
        with torch.autocast(device_type=h2.device.type, enabled=False):
            p2 = self.out2_proj(h2.float())
        out2 = unpatchify_2d(p2, self.out_patch, self.out_channels)
        return out2[:, :, : orig_px_h * 2, : orig_px_w * 2]


def save_krea2_pixel_head(
    head: Krea2PixelHead | Krea2PixelHeadX2,
    path: str | Path,
    extra_meta: Optional[dict[str, str]] = None,
) -> None:
    from safetensors.torch import save_file

    is_x2 = isinstance(head, Krea2PixelHeadX2)
    base = head.base if is_x2 else head
    meta = {
        "arch": "krea2_pixel_head_x2" if is_x2 else "krea2_pixel_head",
        "d_feat": str(base.d_feat),
        "d_lat": str(base.d_lat),
        "width": str(base.width),
        "depth_low": str(base.depth_low),
        "depth_high": str(base.depth_high),
        "out_channels": str(base.out_channels),
        "ls_init": str(base.ls_init),
        "scale": "2" if is_x2 else "1",
        "latent_input": "x0_hat_norm_2x2",
    }
    if is_x2:
        meta["depth_x2"] = str(head.depth_x2)
    if extra_meta:
        for k, v in extra_meta.items():
            meta[str(k)] = str(v)

    sd = {k: v.detach().cpu().contiguous() for k, v in head.state_dict().items()}
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    save_file(sd, str(path), metadata=meta)


def load_krea2_pixel_head(
    path: str | Path,
    device: str | torch.device = "cpu",
    dtype: torch.dtype = torch.float32,
) -> Krea2PixelHead | Krea2PixelHeadX2:
    from safetensors import safe_open
    from safetensors.torch import load_file

    path = Path(path)
    with safe_open(str(path), framework="pt", device="cpu") as fh:
        meta = dict(fh.metadata() or {})

    arch = meta.get("arch", "krea2_pixel_head")
    base = Krea2PixelHead(
        d_feat=int(meta.get("d_feat", "6144")),
        d_lat=int(meta.get("d_lat", "64")),
        width=int(meta.get("width", "1024")),
        depth_low=int(meta.get("depth_low", "2")),
        depth_high=int(meta.get("depth_high", "1")),
        out_channels=int(meta.get("out_channels", "3")),
        ls_init=float(meta.get("ls_init", "1.0")),
    )
    if arch == "krea2_pixel_head_x2" or "depth_x2" in meta:
        model: nn.Module = Krea2PixelHeadX2(
            base=base,
            depth_x2=int(meta.get("depth_x2", "2")),
            ls_init=float(meta.get("ls_init", "1.0")),
        )
    else:
        model = base

    sd = load_file(str(path), device="cpu")
    model.load_state_dict(sd, strict=True)
    model = model.to(device=device, dtype=dtype)
    if isinstance(model, Krea2PixelHeadX2):
        model.base.out_proj.float()
        model.out2_proj.float()
    else:
        model.out_proj.float()
    model.meta = meta
    model.eval()
    return model
