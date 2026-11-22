"""ComfyUI Nodes for FLUX.3-Image-Klein Direct Pixel Head (Zero-VAE RGB Decoding).

License: MIT
"""

from __future__ import annotations

import os
import time
import logging

import folder_paths
import comfy.model_management
import torch

from .flux3_pixel_head import (
    Flux3PixelHeadX2,
    load_flux3_pixel_head,
)
from .tap import attach_flux3_tap

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
    return sorted(items) if items else ["flux3_klein_pixel_head.safetensors"]


def _resolve_head_path(name: str) -> str:
    for folder_key in ("pixel_heads", "vae"):
        if folder_key in folder_paths.folder_names_and_paths:
            p = folder_paths.get_full_path(folder_key, name)
            if p and os.path.exists(p):
                return p
    raise FileNotFoundError(
        f"未找到 Pixel Head 权重 `{name}`，请将其放入 `ComfyUI/models/pixel_heads/` 或 `ComfyUI/models/vae/`。"
    )


class Flux3PixelHeadLoader:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "head_name": (_list_candidate_files(),),
                "precision": (["bf16_trunk_fp32_out", "fp32"], {"default": "bf16_trunk_fp32_out"}),
            }
        }

    RETURN_TYPES = ("FLUX3_PIXEL_HEAD",)
    RETURN_NAMES = ("pixel_head",)
    FUNCTION = "load_head"
    CATEGORY = "advanced/flux3"
    TITLE = "Load FLUX.3 Klein Pixel Head"

    def load_head(self, head_name: str, precision: str):
        path = _resolve_head_path(head_name)
        dtype = torch.bfloat16 if precision == "bf16_trunk_fp32_out" else torch.float32
        head = load_flux3_pixel_head(path, device="cpu", dtype=dtype)
        return (head,)


class Flux3PixelHeadTap:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "model": ("MODEL",),
            }
        }

    RETURN_TYPES = ("MODEL", "FLUX3_TAP")
    RETURN_NAMES = ("model", "pixel_tap")
    FUNCTION = "install_tap"
    CATEGORY = "advanced/flux3"
    TITLE = "FLUX.3 Klein Pixel Head Tap"

    def install_tap(self, model):
        patched_model, tap_state = attach_flux3_tap(model)
        return (patched_model, tap_state)


class Flux3PixelDecode:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "samples": ("LATENT",),
                "pixel_head": ("FLUX3_PIXEL_HEAD",),
                "pixel_tap": ("FLUX3_TAP",),
                "output_scale": (["auto", "1x", "2x"], {"default": "auto"}),
            }
        }

    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("image",)
    FUNCTION = "decode"
    CATEGORY = "advanced/flux3"
    TITLE = "FLUX.3 Klein Pixel Decode (Zero VAE)"

    @torch.no_grad()
    def decode(self, samples, pixel_head, pixel_tap, output_scale: str = "auto"):
        _ = samples["samples"]
        f_cpu, xt_cpu, sigma = pixel_tap.resolve_final_features()

        device = comfy.model_management.get_torch_device()
        is_x2 = isinstance(pixel_head, Flux3PixelHeadX2)
        if output_scale == "auto":
            scale = 2 if is_x2 else 1
        elif output_scale == "2x":
            if not is_x2:
                raise ValueError("当前加载的是 1x Pixel Head 权重，若需 2x 直出请加载带 `_x2` 的二阶权重。")
            scale = 2
        else:
            scale = 1

        head_dev = pixel_head.to(device)
        trunk_dtype = next(head_dev.parameters()).dtype
        f_dev = f_cpu.to(device=device, dtype=trunk_dtype)
        xt_dev = xt_cpu.to(device=device, dtype=trunk_dtype)

        if torch.cuda.is_available() and device.type == "cuda":
            torch.cuda.synchronize(device)
        t0 = time.perf_counter()

        if is_x2:
            out_nchw = head_dev(f_dev, xt_dev, scale=scale)
        else:
            out_nchw = head_dev(f_dev, xt_dev)

        if torch.cuda.is_available() and device.type == "cuda":
            torch.cuda.synchronize(device)
        dt_ms = (time.perf_counter() - t0) * 1000.0

        rgb = ((out_nchw[:, :3].float() + 1.0) * 0.5).clamp_(0.0, 1.0).movedim(1, -1).contiguous().cpu()
        pixel_head.to("cpu")

        logger.info(
            f"[Flux3PixelDecode] Decoded {tuple(rgb.shape[1:3])} (scale={scale}x, last_sigma={sigma:.4f}) in {dt_ms:.2f} ms (Zero VAE)"
        )
        return (rgb,)


NODE_CLASS_MAPPINGS = {
    "Flux3PixelHeadLoader": Flux3PixelHeadLoader,
    "Flux3PixelHeadTap": Flux3PixelHeadTap,
    "Flux3PixelDecode": Flux3PixelDecode,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "Flux3PixelHeadLoader": "Load FLUX.3 Klein Pixel Head",
    "Flux3PixelHeadTap": "FLUX.3 Klein Pixel Head Tap",
    "Flux3PixelDecode": "FLUX.3 Klein Pixel Decode (Zero VAE)",
}
