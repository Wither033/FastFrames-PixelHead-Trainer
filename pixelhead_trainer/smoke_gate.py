"""Pre-flight verification gates (Spatial Crop Equivariance, Causal Chunking, and P0 Smoke Climb).

License: MIT
Running these deterministic 1-second checks BEFORE launching a 30-minute training run immediately
catches axis permutation typos, non-square H/W transposes, or misaligned feature hooks!
"""

from __future__ import annotations

import math
import torch
import torch.nn as nn

from .losses import pixel_head_loss_2d


@torch.no_grad()
def verify_spatial_crop_equivariance_2d(
    head: nn.Module,
    d_feat: int,
    d_lat: int,
    token_pixels: int = 16,
    gh: int = 12,
    gw: int = 16,
    crop_h: int = 8,
    crop_w: int = 10,
    margin_tok: int = 3,
    atol: float = 1e-5,
) -> float:
    """Verify on non-square `(gh != gw)` input that cropping tokens matches cropping output pixels!

    If any `view` / `permute` inside `PixelShuffle` or `unpatchify` scrambles H/W or channel axes,
    the interior region of `head(crop(f), crop(x))` will diverge wildly (`max_diff > 0.1`) from
    `crop_px(head(f, x))`. With exact depthwise receptive field margin stripped, `max_diff < 1e-5`.
    """
    was_training = head.training
    head.eval()
    dev = next(head.parameters()).device
    gen = torch.Generator(device="cpu").manual_seed(20261123)
    f_full = torch.randn(1, gh, gw, d_feat, generator=gen).to(dev)
    x_full = torch.randn(1, d_lat, gh, gw, generator=gen).to(dev)

    y0, x0 = 2, 3
    out_full = head(f_full, x_full)
    out_crop = head(
        f_full[:, y0 : y0 + crop_h, x0 : x0 + crop_w],
        x_full[:, :, y0 : y0 + crop_h, x0 : x0 + crop_w],
    )

    m_px = margin_tok * token_pixels
    s_px = token_pixels
    ref_interior = out_full[
        :,
        :,
        (y0 * s_px + m_px) : ((y0 + crop_h) * s_px - m_px),
        (x0 * s_px + m_px) : ((x0 + crop_w) * s_px - m_px),
    ]
    crop_interior = out_crop[:, :, m_px : crop_h * s_px - m_px, m_px : crop_w * s_px - m_px]
    max_diff = float((ref_interior - crop_interior).abs().max().item())
    if was_training:
        head.train()
    if max_diff > atol:
        raise AssertionError(
            f"2D Spatial Crop Equivariance FAILED: max_abs_diff={max_diff:.6e} > {atol:.6e}. "
            "Check PixelShuffle / unpatchify axis permutation!"
        )
    return max_diff


@torch.no_grad()
def verify_spatial_crop_equivariance_3d(
    head: nn.Module,
    d_feat: int,
    d_lat: int,
    token_pixels: int = 32,
    t_lat: int = 3,
    gh: int = 12,
    gw: int = 16,
    crop_h: int = 10,
    crop_w: int = 12,
    margin_tok: int = 4,
    atol: float = 1e-5,
) -> float:
    """Verify non-square `(T_lat, gh, gw)` 3D spatial crop equivariance (`gh != gw != T_lat`)."""
    was_training = head.training
    head.eval()
    dev = next(head.parameters()).device
    gen = torch.Generator(device="cpu").manual_seed(20261123)
    f_full = torch.randn(1, t_lat, gh, gw, d_feat, generator=gen).to(dev)
    x_full = torch.randn(1, d_lat, t_lat, gh, gw, generator=gen).to(dev)

    y0, x0 = 1, 2
    out_full = head(f_full, x_full)
    out_crop = head(
        f_full[:, :, y0 : y0 + crop_h, x0 : x0 + crop_w],
        x_full[:, :, :, y0 : y0 + crop_h, x0 : x0 + crop_w],
    )

    m_px = margin_tok * token_pixels
    s_px = token_pixels
    ref_interior = out_full[
        :,
        :,
        :,
        (y0 * s_px + m_px) : ((y0 + crop_h) * s_px - m_px),
        (x0 * s_px + m_px) : ((x0 + crop_w) * s_px - m_px),
    ]
    crop_interior = out_crop[:, :, :, m_px : crop_h * s_px - m_px, m_px : crop_w * s_px - m_px]
    max_diff = float((ref_interior - crop_interior).abs().max().item())
    if was_training:
        head.train()
    if max_diff > atol:
        raise AssertionError(
            f"3D Spatial Crop Equivariance FAILED: max_abs_diff={max_diff:.6e} > {atol:.6e}. "
            "Check 3D PixelShuffle / unpatchify axis permutation!"
        )
    return max_diff


@torch.no_grad()
def verify_chunked_causal_decode_3d(
    head: nn.Module,
    d_feat: int,
    d_lat: int,
    t_lat: int = 11,
    gh: int = 4,
    gw: int = 6,
    chunk_t: int = 4,
    atol: float = 1e-5,
) -> float:
    """Verify `decode_chunked` matches full-sequence causal `forward` within floating-point precision."""
    was_training = head.training
    head.eval()
    dev = next(head.parameters()).device
    gen = torch.Generator(device="cpu").manual_seed(99)
    f_seq = torch.randn(1, t_lat, gh, gw, d_feat, generator=gen).to(dev)
    x_seq = torch.randn(1, d_lat, t_lat, gh, gw, generator=gen).to(dev)

    full_out = head(f_seq, x_seq).cpu()
    chunk_out = head.decode_chunked(f_seq, x_seq, chunk_t=chunk_t).cpu()
    assert full_out.shape == chunk_out.shape, f"Shape mismatch: {full_out.shape} vs {chunk_out.shape}"
    max_diff = float((full_out - chunk_out).abs().max().item())
    if was_training:
        head.train()
    if max_diff > atol:
        raise AssertionError(f"Causal chunked decode mismatch: max_abs_diff={max_diff:.6e} > {atol:.6e}")
    return max_diff


def run_p0_smoke_gate_2d(
    head_factory,
    sample_pairs: list[tuple[torch.Tensor, torch.Tensor, torch.Tensor]],
    steps: int = 90,
    lr: float = 8e-4,
    min_final_db: float = 18.5,
    min_gain_db: float = 6.0,
) -> tuple[float, float]:
    """Run a 5-second 90-step overfitting smoke gate on 2 real full-frame `(f, x0_hat, tgt)` samples."""
    dev = sample_pairs[0][0].device
    smoke_head = head_factory().to(dev)
    opt = torch.optim.AdamW(smoke_head.parameters(), lr=lr, betas=(0.9, 0.99), weight_decay=0.0)
    init_p, final_p = [], []
    for st in range(1, steps + 1):
        f_in, x_in, tgt = sample_pairs[st % len(sample_pairs)]
        opt.zero_grad(set_to_none=True)
        use_amp = dev.type == "cuda"
        with torch.autocast(device_type=dev.type, dtype=torch.bfloat16, enabled=use_amp):
            pred = smoke_head(f_in, x_in)
        loss, psnr = pixel_head_loss_2d(pred, tgt)
        if not math.isfinite(float(loss.item())):
            raise RuntimeError(f"NaN/Inf loss at P0 step {st}!")
        loss.backward()
        torch.nn.utils.clip_grad_norm_(smoke_head.parameters(), 1.0)
        opt.step()
        if st <= 4:
            init_p.append(psnr)
        if st > steps - 10:
            final_p.append(psnr)

    p_start = sum(init_p) / len(init_p)
    p_end = sum(final_p) / len(final_p)
    del smoke_head, opt
    if p_end < min_final_db or (p_end - p_start) < min_gain_db:
        raise AssertionError(
            f"P0 Smoke Gate FAILED: PSNR {p_start:.2f} dB -> {p_end:.2f} dB "
            f"(expected >= {min_final_db:.1f} dB and gain >= +{min_gain_db:.1f} dB). "
            "Check feature sequence slicing (`txtlen`/`ref_tokens`) and latent normalization!"
        )
    return p_start, p_end
