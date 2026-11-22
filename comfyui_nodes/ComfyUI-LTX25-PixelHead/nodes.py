"""ComfyUI Nodes for LTX-2.5 3D Causal Video Direct Pixel Head (Zero-Video-VAE Decoding).

License: MIT
"""

from __future__ import annotations

import os
import time
import logging

import folder_paths
import comfy.model_management
import torch

from .ltx25_pixel_head import load_ltx25_pixel_head
from .tap import attach_ltx25_tap

logger = logging.getLogger(__name__)

PIXEL_HEAD_DIR = os.path.join(folder_paths.models_dir, "pixel_heads")
os.makedirs(PIXEL_HEAD_DIR, exist_ok=True)
if "pixel_heads" not in folder_paths.folder_names_and_paths:
    folder_paths.add_model_folder_path("pixel_heads", PIXEL_HEAD_DIR)


def _list_candidate_files() -> list[str]:
    seen = set()
    items = []
    for folder_key in ("pixel_heads", "vae"):
        if folder_key in folder_paths.folder_names_and_paths:
            for fn in folder_paths.get_filename_list(folder_key):
                if fn.endswith(".safetensors") and fn not in seen:
                    seen.add(fn)
                    items.append(fn)
    return sorted(items) if items else ["ltx_2.5_video_pixel_head.safetensors"]


def _resolve_head_path(name: str) -> str:
    for folder_key in ("pixel_heads", "vae"):
        if folder_key in folder_paths.folder_names_and_paths:
            p = folder_paths.get_full_path(folder_key, name)
            if p and os.path.exists(p):
                return p
    raise FileNotFoundError(
        f"未找到 LTX-2.5 Pixel Head 权重 `{name}`，请将其放入 `ComfyUI/models/pixel_heads/` 或 `ComfyUI/models/vae/`。"
    )


class LTX25PixelHeadLoader:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "head_name": (_list_candidate_files(),),
                "precision": (["bf16_trunk_fp32_out", "fp32"], {"default": "bf16_trunk_fp32_out"}),
            }
        }

    RETURN_TYPES = ("LTX25_PIXEL_HEAD",)
    RETURN_NAMES = ("pixel_head",)
    FUNCTION = "load_head"
    CATEGORY = "advanced/ltx25"
    TITLE = "Load LTX-2.5 Video Pixel Head"

    def load_head(self, head_name: str, precision: str):
        path = _resolve_head_path(head_name)
        dtype = torch.bfloat16 if precision == "bf16_trunk_fp32_out" else torch.float32
        head = load_ltx25_pixel_head(path, device="cpu", dtype=dtype)
        return (head,)


class LTX25PixelHeadTap:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "model": ("MODEL",),
            }
        }

    RETURN_TYPES = ("MODEL", "LTX25_TAP")
    RETURN_NAMES = ("model", "pixel_tap")
    FUNCTION = "install_tap"
    CATEGORY = "advanced/ltx25"
    TITLE = "LTX-2.5 Video Pixel Head Tap"

    def install_tap(self, model):
        patched_model, tap_state = attach_ltx25_tap(model)
        return (patched_model, tap_state)


class LTX25VideoPixelDecode:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "samples": ("LATENT",),
                "pixel_head": ("LTX25_PIXEL_HEAD",),
                "pixel_tap": ("LTX25_TAP",),
                "temporal_chunk_latents": ("INT", {"default": 8, "min": 2, "max": 64, "step": 1}),
            }
        }

    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("images",)
    FUNCTION = "decode"
    CATEGORY = "advanced/ltx25"
    TITLE = "LTX-2.5 Video Pixel Decode (Zero Video VAE)"

    @torch.no_grad()
    def decode(self, samples, pixel_head, pixel_tap, temporal_chunk_latents: int = 8):
        _ = samples["samples"]
        f_cpu, x0_vid_cpu, sigma = pixel_tap.resolve_final_features()

        device = comfy.model_management.get_torch_device()
        head_dev = pixel_head.to(device)
        trunk_dtype = next(head_dev.parameters()).dtype
        f_dev = f_cpu.to(device=device, dtype=trunk_dtype)
        x_dev = x0_vid_cpu.to(device=device, dtype=trunk_dtype)

        if torch.cuda.is_available() and device.type == "cuda":
            torch.cuda.synchronize(device)
        t0 = time.perf_counter()

        # [B, 3, F, H, W] in [-1, 1] (overlap_t=None uses exact causal receptive field)
        out_bcthw = head_dev.decode_chunked(f_dev, x_dev, chunk_t=int(temporal_chunk_latents))

        if torch.cuda.is_available() and device.type == "cuda":
            torch.cuda.synchronize(device)
        dt_ms = (time.perf_counter() - t0) * 1000.0

        # ComfyUI video IMAGE format is [B*F, H, W, 3] in [0, 1]
        b, c, f_cnt, h, w = out_bcthw.shape
        frames = (
            ((out_bcthw.float() + 1.0) * 0.5)
            .clamp_(0.0, 1.0)
            .permute(0, 2, 3, 4, 1)
            .reshape(b * f_cnt, h, w, c)
            .contiguous()
            .cpu()
        )
        pixel_head.to("cpu")

        logger.info(
            f"[LTX25VideoPixelDecode] Decoded {f_cnt} frames @ {w}x{h} (last_sigma={sigma:.4f}) in {dt_ms:.1f} ms (Zero Video VAE)"
        )
        return (frames,)


NODE_CLASS_MAPPINGS = {
    "LTX25PixelHeadLoader": LTX25PixelHeadLoader,
    "LTX25PixelHeadTap": LTX25PixelHeadTap,
    "LTX25VideoPixelDecode": LTX25VideoPixelDecode,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "LTX25PixelHeadLoader": "Load LTX-2.5 Video Pixel Head",
    "LTX25PixelHeadTap": "LTX-2.5 Video Pixel Head Tap",
    "LTX25VideoPixelDecode": "LTX-2.5 Video Pixel Decode (Zero Video VAE)",
}
