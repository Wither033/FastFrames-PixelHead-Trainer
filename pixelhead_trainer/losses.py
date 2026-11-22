"""2D Image & 3D Video PixelHead reconstruction losses and PSNR metrics.

License: MIT
"""

from __future__ import annotations

import math
import torch


def compute_psnr_db(pred: torch.Tensor, tgt: torch.Tensor, value_range: str = "minus1_to_1") -> float:
    """Compute Peak Signal-to-Noise Ratio (dB) between `pred` and `tgt`.

    Args:
        pred: Predicted image/video tensor.
        tgt: Teacher VAE reference tensor of identical shape.
        value_range: `"minus1_to_1"` (training range `[-1, 1]`) or `"zero_to_1"` (eval range `[0, 1]`).
    """
    if value_range == "minus1_to_1":
        p01 = ((pred.detach().float() + 1.0) * 0.5).clamp(0.0, 1.0)
        t01 = ((tgt.detach().float() + 1.0) * 0.5).clamp(0.0, 1.0)
    else:
        p01 = pred.detach().float().clamp(0.0, 1.0)
        t01 = tgt.detach().float().clamp(0.0, 1.0)
    mse = max(1e-10, float(((p01 - t01) ** 2).mean().item()))
    return -10.0 * math.log10(mse)


def strip_spatial_margin(img_or_vid: torch.Tensor, margin_px: tuple[int, int, int, int]) -> torch.Tensor:
    """Strip `(top, bottom, left, right)` border pixels so cropped ConvNeXt padding never biases loss."""
    mt, mb, ml, mr = margin_px
    h, w = img_or_vid.shape[-2], img_or_vid.shape[-1]
    return img_or_vid[..., mt : h - mb, ml : w - mr]


def pixel_head_loss_2d(
    pred: torch.Tensor,
    tgt: torch.Tensor,
    w_l1: float = 0.5,
    w_mse: float = 5.0,
    w_grad: float = 0.05,
) -> tuple[torch.Tensor, float]:
    """2D Direct Pixel Head loss on `[-1, 1]` tensors `[B, C, H, W]`.

    Combines:
      - `MSE`: anchors global color/luminance and suppresses outlier artifacts (drives PSNR dB).
      - `L1`: preserves sharp step edges without quadratic over-smoothing.
      - `Spatial finite-difference gradients` (`gh + gw`): enforces crisp stroke/hair transitions.
    """
    pred_f = pred.float()
    tgt_f = tgt.float()
    l1 = (pred_f - tgt_f).abs().mean()
    mse = ((pred_f - tgt_f) ** 2).mean()
    gh = (pred_f.diff(dim=-2) - tgt_f.diff(dim=-2)).abs().mean()
    gw = (pred_f.diff(dim=-1) - tgt_f.diff(dim=-1)).abs().mean()
    total = w_l1 * l1 + w_mse * mse + w_grad * (gh + gw)
    psnr_db = -10.0 * math.log10(max(1e-10, float(mse.detach().item()) * 0.25))
    return total, psnr_db


def video_pixel_head_loss_3d(
    pred: torch.Tensor,
    tgt: torch.Tensor,
    w_l1: float = 0.5,
    w_mse: float = 5.0,
    w_grad_spatial: float = 0.05,
    w_grad_temporal: float = 0.15,
    motion_boost: float = 2.0,
) -> tuple[torch.Tensor, float]:
    """3D Causal Video Pixel Head loss on `[-1, 1]` tensors `[B, C, F, H, W]`.

    Adds two video-critical terms on top of 2D spatial reconstruction:
      1. `Temporal finite-difference loss` (`pred.diff(dim=2) - tgt.diff(dim=2)`): directly eliminates
         inter-frame flicker and `1f <-> 4f` boundary seams.
      2. `Motion-weighted L1`: upweights dynamic foreground pixels where `|tgt[t] - tgt[t-1]|` is large,
         preventing moving limbs/water/particles from blurring into static backgrounds.
    """
    pred_f = pred.float()
    tgt_f = tgt.float()
    diff = pred_f - tgt_f
    l1 = diff.abs().mean()
    mse = (diff ** 2).mean()
    gh = (pred_f.diff(dim=-2) - tgt_f.diff(dim=-2)).abs().mean()
    gw = (pred_f.diff(dim=-1) - tgt_f.diff(dim=-1)).abs().mean()

    total = w_l1 * l1 + w_mse * mse + w_grad_spatial * (gh + gw)
    if pred_f.shape[2] > 1:
        dt_pred = pred_f.diff(dim=2)
        dt_tgt = tgt_f.diff(dim=2)
        gt = (dt_pred - dt_tgt).abs().mean()
        motion_weight = 1.0 + motion_boost * dt_tgt.abs().detach().clamp(0.0, 1.0)
        motion_l1 = (diff[:, :, 1:].abs() * motion_weight).mean()
        total = total + w_grad_temporal * gt + 0.25 * motion_l1

    psnr_db = -10.0 * math.log10(max(1e-10, float(mse.detach().item()) * 0.25))
    return total, psnr_db
