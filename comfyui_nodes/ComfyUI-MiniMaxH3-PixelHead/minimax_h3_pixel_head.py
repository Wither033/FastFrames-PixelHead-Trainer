"""Clean-room PyTorch implementation of 3D Non-Uniform Video Direct Pixel Head for MiniMax-H3.

License: MIT
Architecture v2 (Overcomplete 3-Stage Spatial Shuffle + Unified Frame-Timeline Causal 3D Refinement)
follows FastFrames (arXiv:2611.23226), tailored specifically to MiniMax-H3 (`MiniMaxH3Model` + `MiniMaxH3VideoVAE`):
  - Frozen MiniMax-H3 DiT (`hidden_size=5376`, `latents_dim=24`, `patch_size=(1, 2, 2)` -> `d_lat = 96`,
    VAE `16x` spatial -> `32x32` pixels per DiT token; non-uniform `FRAME_PER_TOKEN = (1, 4, 4, 4, 4)` temporal!).
  - Pre-hook on `diffusion_model.final_layer` computes the exact AdaLN-modulated video token sequence
    `f_vid = mod(video_seg)` (`[B, T_lat * gh * gw, 5376]`), working identically on BOTH `n>1` PDD 8-Step Turbo
    checkpoints (`8-Step PDD Turbo`) AND standard `n=1` Base/Ref2VA checkpoints!
  - Progressive 3-stage Spatial PixelShuffle + Unified Frame-Level Causal 3D ConvNeXt:
      1. `feat_norm(5376) -> Linear(5376 -> w32, bias=False)` + `Linear(96 -> w32, bias=True)` -> `entry_norm`.
      2. `depth_s32` CausalConvNeXt3d blocks at `(T_lat, gh, gw)` (32px grid).
      3. Spatial 2x PixelShuffle -> `depth_s16` CausalConvNeXt3d blocks at `(T_lat, 2*gh, 2*gw)` (16px grid).
      4. Spatial 2x PixelShuffle -> `depth_s8` CausalConvNeXt3d blocks at `(T_lat, 4*gh, 4*gw)` (8px grid).
      5. Spatial 2x PixelShuffle -> `(T_lat, 8*gh, 8*gw)` (4px grid).
      6. Non-uniform `(1, 4, 4, 4, 4)` temporal unpack (`t1_proj` for `global_t % 5 == 0`, `t4_proj` for `!= 0`)
         into the unified pixel-frame timeline `[B, w4, F_expected, 8*gh, 8*gw]`, followed by `depth_s4`
         CausalConvNeXt3d blocks across adjacent frames (eliminating `1f <-> 4f` boundary seams!) and a single
         shared `5.33x` overcomplete FP32 output head (`w4=256 -> 1 * 4 * 4 * 3 = 48`)!
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F


FRAME_PER_TOKEN = (1, 4, 4, 4, 4)
FINAL_SPATIAL_PATCH = 4  # After three 2x PixelShuffles: 32 / 8 = 4px


def h3_expected_frames(t_lat: int, t_offset: int = 0) -> int:
    return sum(FRAME_PER_TOKEN[(t_offset + k) % 5] for k in range(t_lat))


class CausalDepthwiseConv3d(nn.Module):
    """3D depthwise conv with left-only causal temporal padding and symmetric spatial padding."""

    def __init__(self, dim: int, kt: int = 3, kh: int = 3, kw: int = 3):
        super().__init__()
        self.pad_t = kt - 1
        self.pad_h = kh // 2
        self.pad_w = kw // 2
        self.conv = nn.Conv3d(dim, dim, kernel_size=(kt, kh, kw), padding=0, groups=dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
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
        h = self.dw(x).movedim(1, -1)
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


def patchify_h3_video_latent(x0_hat_vid: torch.Tensor) -> tuple[torch.Tensor, int, int]:
    """Convert `[B, 24, T_lat, H_lat, W_lat]` -> `([B, T_lat, gh, gw, 96], orig_px_h, orig_px_w)`."""
    b, c, t_lat, h_lat, w_lat = x0_hat_vid.shape
    orig_px_h, orig_px_w = h_lat * 16, w_lat * 16
    pad_h = h_lat % 2
    pad_w = w_lat % 2
    if pad_h or pad_w:
        x0_hat_vid = F.pad(x0_hat_vid, (0, pad_w, 0, pad_h, 0, 0), mode="replicate")
    gh, gw = x0_hat_vid.shape[3] // 2, x0_hat_vid.shape[4] // 2
    lat_tok = (
        x0_hat_vid.view(b, c, t_lat, gh, 2, gw, 2)
        .permute(0, 2, 3, 5, 1, 4, 6)
        .contiguous()
        .view(b, t_lat, gh, gw, c * 4)
    )
    return lat_tok, orig_px_h, orig_px_w


def unpatchify_h3_frames(
    patches: torch.Tensor, ph: int = FINAL_SPATIAL_PATCH, pw: int = FINAL_SPATIAL_PATCH, out_channels: int = 3
) -> torch.Tensor:
    """Convert `[B, F, H8, W8, ph * pw * C]` -> `[B, C, F, H8 * ph, W8 * pw]`."""
    b, f_cnt, h8, w8, _ = patches.shape
    x = patches.view(b, f_cnt, h8, w8, ph, pw, out_channels)
    x = x.permute(0, 6, 1, 2, 4, 3, 5).contiguous()
    return x.view(b, out_channels, f_cnt, h8 * ph, w8 * pw)


class MiniMaxH3VideoPixelHead(nn.Module):
    """Overcomplete 3D Non-Uniform Video Pixel Head for MiniMax-H3 (`(1,4,4,4,4)` frames, `32x32` spatial/token)."""

    def __init__(
        self,
        d_feat: int = 5376,
        d_lat: int = 96,
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
        w4 = max(128, (width // 3 // 32) * 32)          # 256 when width=768 (>> 48 out_dim!)
        self.w4 = w4

        self.feat_norm = nn.LayerNorm(self.d_feat, eps=1e-6)
        self.feat_proj = nn.Linear(self.d_feat, w32, bias=False)
        self.lat_proj = nn.Linear(self.d_lat, w32, bias=True)
        self.entry_norm = nn.LayerNorm(w32, eps=1e-6)

        self.s32_blocks = nn.ModuleList([CausalConvNeXtBlock3d(w32, ls_init=ls_init) for _ in range(depth_s32)])
        self.up1_norm = nn.LayerNorm(w32, eps=1e-6)
        self.up1_proj = nn.Linear(w32, 4 * w16)

        self.s16_blocks = nn.ModuleList([CausalConvNeXtBlock3d(w16, ls_init=ls_init) for _ in range(depth_s16)])
        self.up2_norm = nn.LayerNorm(w16, eps=1e-6)
        self.up2_proj = nn.Linear(w16, 4 * w8)

        self.s8_blocks = nn.ModuleList([CausalConvNeXtBlock3d(w8, ls_init=ls_init) for _ in range(depth_s8)])
        self.up3_norm = nn.LayerNorm(w8, eps=1e-6)
        self.up3_proj = nn.Linear(w8, 4 * w4)

        # Non-uniform (1, 4, 4, 4, 4) temporal expansion onto unified frame grid at 4px spatial resolution
        self.t_norm = nn.LayerNorm(w4, eps=1e-6)
        self.t1_proj = nn.Linear(w4, 1 * w4)
        self.t4_proj = nn.Linear(w4, 4 * w4)

        # Unified frame-level causal 3D refinement across all F_expected frames (zero 1f<->4f seam!)
        self.s4_blocks = nn.ModuleList([CausalConvNeXtBlock3d(w4, ls_init=ls_init) for _ in range(depth_s4)])
        self.norm_out = nn.LayerNorm(w4, eps=1e-6)

        out_dim = FINAL_SPATIAL_PATCH * FINAL_SPATIAL_PATCH * self.out_channels  # 4*4*3 = 48
        self.out_proj = nn.Linear(w4, out_dim)
        nn.init.normal_(self.out_proj.weight, mean=0.0, std=0.02)
        nn.init.zeros_(self.out_proj.bias)

        self.meta: dict[str, str] = {}

    def causal_receptive_t_lat(self) -> int:
        """Exact causal temporal lookback in latent frames (`T_lat`) for zero-seam chunked decoding."""
        return 2 * self.depth_s32 + 2 * self.depth_s16 + 2 * self.depth_s8 + 2 * self.depth_s4

    def _expand_nonuniform_time(self, h_s4: torch.Tensor, t_offset: int) -> torch.Tensor:
        """Expand `[B, w4, T_lat, H8, W8]` -> `[B, w4, F_expected, H8, W8]` according to `(1, 4, 4, 4, 4)`."""
        b, c4, t_lat, h8, w8 = h_s4.shape
        u = self.t_norm(h_s4.movedim(1, -1))  # [B, T_lat, H8, W8, w4]

        idx_1f = [i for i in range(t_lat) if (t_offset + i) % 5 == 0]
        idx_4f = [i for i in range(t_lat) if (t_offset + i) % 5 != 0]

        slots: list[Optional[torch.Tensor]] = [None] * t_lat
        if idx_1f:
            # [B, n1, H8, W8, w4] -> [B, w4, n1, H8, W8]
            y1 = self.t1_proj(u[:, idx_1f]).permute(0, 4, 1, 2, 3).contiguous()
            for k, ti in enumerate(idx_1f):
                slots[ti] = y1[:, :, k : k + 1]
        if idx_4f:
            # [B, n4, H8, W8, w4 * 4] -> [B, n4, H8, W8, w4, 4] -> [B, w4, n4, 4, H8, W8]
            y4 = self.t4_proj(u[:, idx_4f]).view(b, len(idx_4f), h8, w8, c4, 4).permute(0, 4, 1, 5, 2, 3).contiguous()
            for k, ti in enumerate(idx_4f):
                slots[ti] = y4[:, :, k]  # [B, w4, 4, H8, W8]

        return torch.cat([s for s in slots if s is not None], dim=2)

    def forward(
        self,
        feat: torch.Tensor,
        x0_hat_vid: torch.Tensor,
        t_offset: int = 0,
    ) -> torch.Tensor:
        """Decode `(feat, x0_hat_vid)` into `[B, 3, F_expected, H_px, W_px]` in `[-1, 1]`."""
        if x0_hat_vid.ndim == 5 and x0_hat_vid.shape[1] == self.d_lat:
            b, _, t_lat, gh, gw = x0_hat_vid.shape
            lat_tok = x0_hat_vid.movedim(1, -1)
            orig_px_h, orig_px_w = gh * 32, gw * 32
        else:
            lat_tok, orig_px_h, orig_px_w = patchify_h3_video_latent(x0_hat_vid)
            b, t_lat, gh, gw, _ = lat_tok.shape

        if feat.ndim == 3:
            feat = feat.view(b, t_lat, gh, gw, self.d_feat)

        h = self.entry_norm(self.feat_proj(self.feat_norm(feat)) + self.lat_proj(lat_tok))
        h = h.movedim(-1, 1).contiguous()  # [B, w32, T_lat, gh, gw]

        for blk in self.s32_blocks:
            h = blk(h)
        h = spatial_pixel_shuffle_3d(h, self.up1_norm, self.up1_proj, scale=2)
        for blk in self.s16_blocks:
            h = blk(h)
        h = spatial_pixel_shuffle_3d(h, self.up2_norm, self.up2_proj, scale=2)
        for blk in self.s8_blocks:
            h = blk(h)
        h = spatial_pixel_shuffle_3d(h, self.up3_norm, self.up3_proj, scale=2)  # [B, w4, T_lat, 8*gh, 8*gw]

        h_frames = self._expand_nonuniform_time(h, t_offset=t_offset)  # [B, w4, F_expected, 8*gh, 8*gw]
        for blk in self.s4_blocks:
            h_frames = blk(h_frames)

        h_tok = self.norm_out(h_frames.movedim(1, -1))  # [B, F_expected, 8*gh, 8*gw, w4]
        with torch.autocast(device_type=h_tok.device.type, enabled=False):
            patches = self.out_proj(h_tok.float())
        vid = unpatchify_h3_frames(patches, FINAL_SPATIAL_PATCH, FINAL_SPATIAL_PATCH, self.out_channels)
        return vid[:, :, :, :orig_px_h, :orig_px_w]

    @torch.no_grad()
    def decode_chunked(
        self,
        feat: torch.Tensor,
        x0_hat_vid: torch.Tensor,
        chunk_t: int = 10,
        overlap_t: Optional[int] = None,
    ) -> torch.Tensor:
        """Low-VRAM bit-exact causal chunked decode for 124~362+ frame MiniMax-H3 videos."""
        lat_tok, orig_px_h, orig_px_w = patchify_h3_video_latent(x0_hat_vid)
        b, t_lat, gh, gw, _ = lat_tok.shape
        if feat.ndim == 3:
            feat = feat.view(b, t_lat, gh, gw, self.d_feat)
        lat_5d = lat_tok.movedim(-1, 1).contiguous()  # [B, 96, T_lat, gh, gw]

        if t_lat <= chunk_t:
            out = self.forward(feat, lat_5d, t_offset=0)
            return out[:, :, :, :orig_px_h, :orig_px_w]

        ov = self.causal_receptive_t_lat() if overlap_t is None else int(overlap_t)
        pieces = []
        pos = 0
        while pos < t_lat:
            end = min(t_lat, pos + chunk_t)
            ctx_start = max(0, pos - ov)
            f_sl = feat[:, ctx_start:end]
            x_sl = lat_5d[:, :, ctx_start:end]
            dec_sl = self.forward(f_sl, x_sl, t_offset=ctx_start)[:, :, :, :orig_px_h, :orig_px_w]
            if pos == 0:
                pieces.append(dec_sl.cpu())
            else:
                drop_frames = h3_expected_frames(pos - ctx_start, t_offset=ctx_start)
                pieces.append(dec_sl[:, :, drop_frames:].cpu())
            pos = end
        return torch.cat(pieces, dim=2)


def save_minimax_h3_pixel_head(
    head: MiniMaxH3VideoPixelHead,
    path: str | Path,
    extra_meta: Optional[dict[str, str]] = None,
) -> None:
    from safetensors.torch import save_file

    meta = {
        "arch": "minimax_h3_video_pixel_head_v2",
        "d_feat": str(head.d_feat),
        "d_lat": str(head.d_lat),
        "width": str(head.width),
        "depth_s32": str(head.depth_s32),
        "depth_s16": str(head.depth_s16),
        "depth_s8": str(head.depth_s8),
        "depth_s4": str(head.depth_s4),
        "out_channels": str(head.out_channels),
        "ls_init": str(head.ls_init),
        "frame_per_token": "1,4,4,4,4",
        "latent_input": "x0_hat_vid_1x2x2",
    }
    if extra_meta:
        for k, v in extra_meta.items():
            meta[str(k)] = str(v)

    sd = {k: v.detach().cpu().contiguous() for k, v in head.state_dict().items()}
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    save_file(sd, str(path), metadata=meta)


def load_minimax_h3_pixel_head(
    path: str | Path,
    device: str | torch.device = "cpu",
    dtype: torch.dtype = torch.float32,
) -> MiniMaxH3VideoPixelHead:
    from safetensors import safe_open
    from safetensors.torch import load_file

    path = Path(path)
    with safe_open(str(path), framework="pt", device="cpu") as fh:
        meta = dict(fh.metadata() or {})

    model = MiniMaxH3VideoPixelHead(
        d_feat=int(meta.get("d_feat", "5376")),
        d_lat=int(meta.get("d_lat", "96")),
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
