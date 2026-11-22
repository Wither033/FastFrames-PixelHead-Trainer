"""ComfyUI Nodes for Krea-2 Direct Pixel Head (Zero-VAE RGB Decoding).

License: MIT
"""

from __future__ import annotations

import os
import time
import logging

import folder_paths
import comfy.model_management
import torch

from .krea2_pixel_head import (
    Krea2PixelHeadX2,
    load_krea2_pixel_head,
)
from .tap import attach_krea2_tap

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
    return sorted(items) if items else ["krea2_pixel_head_bf16.safetensors"]


def _resolve_head_path(name: str) -> str:
    for folder_key in ("pixel_heads", "vae"):
        if folder_key in folder_paths.folder_names_and_paths:
            p = folder_paths.get_full_path(folder_key, name)
            if p and os.path.exists(p):
                return p
    raise FileNotFoundError(
        f"未找到 Pixel Head 权重 `{name}`，请将其放入 `ComfyUI/models/pixel_heads/` 或 `ComfyUI/models/vae/`。"
    )


class Krea2PixelHeadLoader:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "head_name": (_list_candidate_files(),),
                "precision": (["bf16_trunk_fp32_out", "fp32"], {"default": "bf16_trunk_fp32_out"}),
            }
        }

    RETURN_TYPES = ("KREA2_PIXEL_HEAD",)
    RETURN_NAMES = ("pixel_head",)
    FUNCTION = "load_head"
    CATEGORY = "advanced/krea2"
    TITLE = "Load Krea-2 Pixel Head"

    def load_head(self, head_name: str, precision: str):
        path = _resolve_head_path(head_name)
        dtype = torch.bfloat16 if precision == "bf16_trunk_fp32_out" else torch.float32
        head = load_krea2_pixel_head(path, device="cpu", dtype=dtype)
        return (head,)


class Krea2PixelHeadTap:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "model": ("MODEL",),
            }
        }

    RETURN_TYPES = ("MODEL", "KREA2_TAP")
    RETURN_NAMES = ("model", "pixel_tap")
    FUNCTION = "install_tap"
    CATEGORY = "advanced/krea2"
    TITLE = "Krea-2 Pixel Head Tap"

    def install_tap(self, model):
        patched_model, tap_state = attach_krea2_tap(model)
        return (patched_model, tap_state)


class Krea2PixelDecode:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "samples": ("LATENT",),
                "pixel_head": ("KREA2_PIXEL_HEAD",),
                "pixel_tap": ("KREA2_TAP",),
                "output_scale": (["auto", "1x", "2x"], {"default": "auto"}),
            }
        }

    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("image",)
    FUNCTION = "decode"
    CATEGORY = "advanced/krea2"
    TITLE = "Krea-2 Pixel Decode (Zero VAE)"

    @torch.no_grad()
    def decode(self, samples, pixel_head, pixel_tap, output_scale: str = "auto"):
        _ = samples["samples"]
        f_cpu, xt_cpu, sigma = pixel_tap.resolve_final_features()

        device = comfy.model_management.get_torch_device()
        is_x2 = isinstance(pixel_head, Krea2PixelHeadX2)
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
            f"[Krea2PixelDecode] Decoded {tuple(rgb.shape[1:3])} (scale={scale}x, last_sigma={sigma:.4f}) in {dt_ms:.2f} ms (Zero VAE)"
        )
        return (rgb,)


NODE_CLASS_MAPPINGS = {
    "Krea2PixelHeadLoader": Krea2PixelHeadLoader,
    "Krea2PixelHeadTap": Krea2PixelHeadTap,
    "Krea2PixelDecode": Krea2PixelDecode,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "Krea2PixelHeadLoader": "Load Krea-2 Pixel Head",
    "Krea2PixelHeadTap": "Krea-2 Pixel Head Tap",
    "Krea2PixelDecode": "Krea-2 Pixel Decode (Zero VAE)",
}
