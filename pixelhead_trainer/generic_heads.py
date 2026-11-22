"""Generic parameterizable 2D & 3D Direct Pixel Readout Heads for adapting to any new DiT.

License: MIT
"""

from __future__ import annotations

import math
from pathlib import Path
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from .blocks import (
    CausalConvNeXtBlock3d,
    ConvNeXtBlock2d,
    spatial_pixel_shuffle_3d,
    spatiotemporal_pixel_shuffle_3d,
    unpatchify_2d,
    unpatchify_3d,
)


class BridgedPixelHead2D(nn.Module):
    """Universal 1x Bridged Direct Pixel Head for any 2D Diffusion Transformer (DiT).

    Design principles:
      1. `feat_norm` + bias-free `Linear(d_feat -> width)` bridge controls parameter size (~25M)
         even when `d_feat` is 4096 or 6144.
      2. Fuses normalized pre-linear velocity features `f` with denoised latent patches `x0_hat_norm`
         at the token grid (`+6 dB` PSNR over feature-only readout, and sigma-invariant across
         4-8 step distilled Turbo and 20-50 step Base schedules).
      3. Intermediate `2x PixelShuffle` ensures the final FP32 linear projection (`width//2 -> out_dim`)
         remains overcomplete (`512 -> 192` for `token_pixels=16`, RGB).
      4. Mandatory FP32 final `out_proj` (`torch.autocast(..., enabled=False)`) eliminates 1-2 LSB
         BF16 color banding on smooth gradients and dark shadows.
    """

    def __init__(
        self,
        d_feat: int,
        d_lat: int,
        token_pixels: int = 16,
        width: int = 1024,
        depth_low: int = 2,
        depth_high: int = 1,
        out_channels: int = 3,
        ls_init: float = 1.0,
    ):
        super().__init__()
        if width % 2 != 0:
            raise ValueError(f"width must be even, got {width}")
        if token_pixels % 2 != 0:
            raise ValueError(f"token_pixels must be divisible by 2, got {token_pixels}")

        self.d_feat = int(d_feat)
        self.d_lat = int(d_lat)
        self.token_pixels = int(token_pixels)
        self.width = int(width)
        self.depth_low = int(depth_low)
        self.depth_high = int(depth_high)
        self.out_channels = int(out_channels)
        self.ls_init = float(ls_init)

        w_high = width // 2
        self.out_patch = self.token_pixels // 2

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

    def forward_trunk(self, feat: torch.Tensor, x0_hat_patched: torch.Tensor) -> torch.Tensor:
        """Run trunk up to `high_blocks` (`[B, width//2, 2*gh, 2*gw]`).

        Args:
            feat: `[B, gh * gw, d_feat]` or `[B, gh, gw, d_feat]` pre-linear hidden states.
            x0_hat_patched: `[B, d_lat, gh, gw]` token-aligned denoised latent.
        """
        b, _, gh, gw = x0_hat_patched.shape
        if feat.ndim == 3:
            feat = feat.view(b, gh, gw, self.d_feat)
        lat_tok = x0_hat_patched.movedim(1, -1)

        h = self.entry_norm(self.feat_proj(self.feat_norm(feat)) + self.lat_proj(lat_tok))
        h = h.movedim(-1, 1).contiguous()
        for blk in self.low_blocks:
            h = blk(h)

        u = self.up_proj(self.up_norm(h.movedim(1, -1))).movedim(-1, 1).contiguous()
        h = F.pixel_shuffle(u, 2)
        for blk in self.high_blocks:
            h = blk(h)
        return h

    def forward(self, feat: torch.Tensor, x0_hat_patched: torch.Tensor) -> torch.Tensor:
        """Return `[B, out_channels, gh * token_pixels, gw * token_pixels]` in `[-1, 1]`."""
        h = self.forward_trunk(feat, x0_hat_patched)
        h_tok = self.norm_out(h.movedim(1, -1))
        with torch.autocast(device_type=h_tok.device.type, enabled=False):
            patches = self.out_proj(h_tok.float())
        return unpatchify_2d(patches, self.out_patch, self.out_channels)


class BridgedPixelHead2DX2(nn.Module):
    """Stage-2 `2x` Super-Resolution extension (`2 * token_pixels` per DiT token) with frozen `1x` trunk."""

    def __init__(self, base: BridgedPixelHead2D, depth_x2: int = 2, ls_init: float = 1.0):
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

    def forward(self, feat: torch.Tensor, x0_hat_patched: torch.Tensor, scale: int = 2) -> torch.Tensor:
        h = self.base.forward_trunk(feat, x0_hat_patched)
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


class CausalVideoPixelHead3D(nn.Module):
    """Overcomplete 3-Stage Spatiotemporal PixelShuffle Causal 3D Video Head (`(8t, 32px)` -> `(2t, 4px)`).

    Designed for uniform-stride causal video DiTs (e.g. LTX-2.5 `(8, 32, 32)`, Wan-2.1/2.2 `(4, 16, 16)`
    when configured accordingly). Eliminates the fatal undercomplete linear bottleneck (`256 -> 1536`)
    by progressively shuffling down to `(pt=2, ph=4, pw=4)` (`w4=256 -> 96`, `2.67x` overcomplete!).
    """

    def __init__(
        self,
        d_feat: int = 4096,
        d_lat: int = 128,
        width: int = 768,
        causal_pre_frames: int = 7,
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
        self.causal_pre_frames = int(causal_pre_frames)
        self.depth_s32 = int(depth_s32)
        self.depth_s16 = int(depth_s16)
        self.depth_s8 = int(depth_s8)
        self.depth_s4 = int(depth_s4)
        self.out_channels = int(out_channels)
        self.ls_init = float(ls_init)

        w32 = width
        w16 = max(128, ((width * 2) // 3 // 32) * 32)
        w8 = max(128, (width // 2 // 32) * 32)
        w4 = max(128, (width // 3 // 32) * 32)

        self.feat_norm = nn.LayerNorm(self.d_feat, eps=1e-6)
        self.feat_proj = nn.Linear(self.d_feat, w32, bias=False)
        self.lat_proj = nn.Linear(self.d_lat, w32, bias=True)
        self.entry_norm = nn.LayerNorm(w32, eps=1e-6)

        self.s32_blocks = nn.ModuleList([CausalConvNeXtBlock3d(w32, ls_init=ls_init) for _ in range(depth_s32)])
        self.up1_norm = nn.LayerNorm(w32, eps=1e-6)
        self.up1_proj = nn.Linear(w32, 4 * w16)

        self.s16_blocks = nn.ModuleList([CausalConvNeXtBlock3d(w16, ls_init=ls_init) for _ in range(depth_s16)])
        self.up2_norm = nn.LayerNorm(w16, eps=1e-6)
        self.up2_proj = nn.Linear(w16, 8 * w8)

        self.s8_blocks = nn.ModuleList([CausalConvNeXtBlock3d(w8, ls_init=ls_init) for _ in range(depth_s8)])
        self.up3_norm = nn.LayerNorm(w8, eps=1e-6)
        self.up3_proj = nn.Linear(w8, 8 * w4)

        self.s4_blocks = nn.ModuleList([CausalConvNeXtBlock3d(w4, ls_init=ls_init) for _ in range(depth_s4)])
        self.norm_out = nn.LayerNorm(w4, eps=1e-6)
        self.out_proj = nn.Linear(w4, 2 * 4 * 4 * self.out_channels)
        nn.init.normal_(self.out_proj.weight, mean=0.0, std=0.02)
        nn.init.zeros_(self.out_proj.bias)

    def causal_receptive_t_lat(self) -> int:
        return (
            2 * self.depth_s32
            + 2 * self.depth_s16
            + math.ceil((2 * self.depth_s8) / 2)
            + math.ceil((2 * self.depth_s4) / 4)
        )

    def forward_raw(self, feat: torch.Tensor, x0_hat_vid: torch.Tensor) -> torch.Tensor:
        b, _, t_lat, h_lat, w_lat = x0_hat_vid.shape
        if feat.ndim == 3:
            feat = feat.view(b, t_lat, h_lat, w_lat, self.d_feat)
        lat_tok = x0_hat_vid.movedim(1, -1)

        h = self.entry_norm(self.feat_proj(self.feat_norm(feat)) + self.lat_proj(lat_tok))
        h = h.movedim(-1, 1).contiguous()

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

        h_tok = self.norm_out(h.movedim(1, -1))
        with torch.autocast(device_type=h_tok.device.type, enabled=False):
            patches = self.out_proj(h_tok.float())
        return unpatchify_3d(patches, 2, 4, 4, self.out_channels)

    def forward(self, feat: torch.Tensor, x0_hat_vid: torch.Tensor, align_causal: bool = True) -> torch.Tensor:
        raw = self.forward_raw(feat, x0_hat_vid)
        if align_causal and self.causal_pre_frames > 0:
            return raw[:, :, self.causal_pre_frames :, :, :]
        return raw

    @torch.no_grad()
    def decode_chunked(
        self,
        feat: torch.Tensor,
        x0_hat_vid: torch.Tensor,
        chunk_t: int = 8,
        overlap_t: Optional[int] = None,
    ) -> torch.Tensor:
        """Bit-exact causal chunked decode with bounded peak VRAM on arbitrary-length videos."""
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
            raw_sl = self.forward_raw(feat[:, ctx_start:end], x0_hat_vid[:, :, ctx_start:end])
            if pos == 0:
                pieces.append(raw_sl[:, :, self.causal_pre_frames :].cpu())
            else:
                keep_lat = end - pos
                pieces.append(raw_sl[:, :, -keep_lat * 8 :].cpu())
            pos = end
        return torch.cat(pieces, dim=2)


def save_generic_pixel_head_2d(
    head: BridgedPixelHead2D | BridgedPixelHead2DX2,
    path: str | Path,
    extra_meta: Optional[dict[str, str]] = None,
) -> None:
    from safetensors.torch import save_file

    is_x2 = isinstance(head, BridgedPixelHead2DX2)
    base = head.base if is_x2 else head
    meta = {
        "arch": "generic_pixel_head_2d_x2" if is_x2 else "generic_pixel_head_2d",
        "d_feat": str(base.d_feat),
        "d_lat": str(base.d_lat),
        "token_pixels": str(base.token_pixels),
        "width": str(base.width),
        "depth_low": str(base.depth_low),
        "depth_high": str(base.depth_high),
        "out_channels": str(base.out_channels),
        "ls_init": str(base.ls_init),
        "scale": "2" if is_x2 else "1",
    }
    if is_x2:
        meta["depth_x2"] = str(head.depth_x2)
    if extra_meta:
        for k, v in extra_meta.items():
            meta[str(k)] = str(v)

    sd = {k: v.detach().cpu().contiguous() for k, v in head.state_dict().items()}
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    save_file(sd, str(path), metadata=meta)


def load_generic_pixel_head_2d(
    path: str | Path,
    device: str | torch.device = "cpu",
    dtype: torch.dtype = torch.float32,
) -> BridgedPixelHead2D | BridgedPixelHead2DX2:
    from safetensors import safe_open
    from safetensors.torch import load_file

    path = Path(path)
    with safe_open(str(path), framework="pt", device="cpu") as fh:
        meta = dict(fh.metadata() or {})

    base = BridgedPixelHead2D(
        d_feat=int(meta["d_feat"]),
        d_lat=int(meta["d_lat"]),
        token_pixels=int(meta.get("token_pixels", "16")),
        width=int(meta.get("width", "1024")),
        depth_low=int(meta.get("depth_low", "2")),
        depth_high=int(meta.get("depth_high", "1")),
        out_channels=int(meta.get("out_channels", "3")),
        ls_init=float(meta.get("ls_init", "1.0")),
    )
    if meta.get("arch") == "generic_pixel_head_2d_x2" or "depth_x2" in meta:
        model: nn.Module = BridgedPixelHead2DX2(
            base=base,
            depth_x2=int(meta.get("depth_x2", "2")),
            ls_init=float(meta.get("ls_init", "1.0")),
        )
    else:
        model = base

    sd = load_file(str(path), device="cpu")
    model.load_state_dict(sd, strict=True)
    model = model.to(device=device, dtype=dtype)
    if isinstance(model, BridgedPixelHead2DX2):
        model.base.out_proj.float()
        model.out2_proj.float()
    else:
        model.out_proj.float()
    model.meta = meta
    model.eval()
    return model


def export_bf16_trunk_fp32_out(src_fp32_path: str | Path, dst_bf16_path: str | Path) -> None:
    """Convert an FP32 `.safetensors` PixelHead into half-size BF16 trunk + FP32 output projection."""
    from safetensors import safe_open
    from safetensors.torch import save_file

    keep_fp32_prefixes = ("out_proj.", "out2_proj.", "base.out_proj.")
    with safe_open(str(src_fp32_path), framework="pt", device="cpu") as fh:
        meta = dict(fh.metadata() or {})
        sd_out = {}
        for k in fh.keys():
            t = fh.get_tensor(k)
            if k.startswith(keep_fp32_prefixes):
                sd_out[k] = t.float().contiguous()
            else:
                sd_out[k] = t.to(torch.bfloat16).contiguous()
    meta["storage_precision"] = "bf16_trunk_fp32_out"
    Path(dst_bf16_path).parent.mkdir(parents=True, exist_ok=True)
    save_file(sd_out, str(dst_bf16_path), metadata=meta)
