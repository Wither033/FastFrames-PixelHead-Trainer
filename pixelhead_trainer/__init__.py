"""Clean-room PyTorch toolkit for training Zero-VAE Direct Pixel Readout Heads (2D & 3D Video).

License: MIT
Mathematical formulation inspired by FastFrames (arXiv:2611.23226) and single-step
Replay Buffer distillation practices.
"""

from .blocks import (
    CausalConvNeXtBlock3d,
    CausalDepthwiseConv3d,
    ConvNeXtBlock2d,
    spatial_pixel_shuffle_3d,
    spatiotemporal_pixel_shuffle_3d,
    unpatchify_2d,
    unpatchify_3d,
)
from .generic_heads import (
    BridgedPixelHead2D,
    BridgedPixelHead2DX2,
    CausalVideoPixelHead3D,
    export_bf16_trunk_fp32_out,
    load_generic_pixel_head_2d,
    save_generic_pixel_head_2d,
)
from .losses import (
    compute_psnr_db,
    pixel_head_loss_2d,
    strip_spatial_margin,
    video_pixel_head_loss_3d,
)
from .smoke_gate import (
    run_p0_smoke_gate_2d,
    verify_chunked_causal_decode_3d,
    verify_spatial_crop_equivariance_2d,
    verify_spatial_crop_equivariance_3d,
)

__all__ = [
    "ConvNeXtBlock2d",
    "CausalDepthwiseConv3d",
    "CausalConvNeXtBlock3d",
    "unpatchify_2d",
    "unpatchify_3d",
    "spatial_pixel_shuffle_3d",
    "spatiotemporal_pixel_shuffle_3d",
    "BridgedPixelHead2D",
    "BridgedPixelHead2DX2",
    "CausalVideoPixelHead3D",
    "save_generic_pixel_head_2d",
    "load_generic_pixel_head_2d",
    "export_bf16_trunk_fp32_out",
    "compute_psnr_db",
    "strip_spatial_margin",
    "pixel_head_loss_2d",
    "video_pixel_head_loss_3d",
    "verify_spatial_crop_equivariance_2d",
    "verify_spatial_crop_equivariance_3d",
    "verify_chunked_causal_decode_3d",
    "run_p0_smoke_gate_2d",
]
