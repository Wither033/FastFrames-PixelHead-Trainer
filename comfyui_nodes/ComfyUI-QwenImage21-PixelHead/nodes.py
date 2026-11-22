"""ComfyUI Nodes for Qwen-Image 2.1 Direct Pixel Head (Zero-VAE RGB & RGBA Decoding).

License: MIT
"""

from __future__ import annotations

import logging
import os
import time

import comfy.model_management
import folder_paths
import torch

from .qwen21_pixel_head import (
    QwenImage21PixelHeadX2,
    load_qwen21_pixel_head,
)
from .tap import attach_qwen21_tap

logger = logging.getLogger(__name__)

# Register dedicated `models/pixel_heads` directory AND allow loading from `models/vae` for convenience
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
    return sorted(items) if items else ["qwen_image_2.1_pixel_head_bf16.safetensors", "qwen_image_2.1_pixel_head.safetensors"]


def _resolve_head_path(name: str) -> str:
    for folder_key in ("pixel_heads", "vae"):
        if folder_key in folder_paths.folder_names_and_paths:
            p = folder_paths.get_full_path(folder_key, name)
            if p and os.path.exists(p):
                return p
    raise FileNotFoundError(
        f"Pixel Head weight `{name}` not found. Place it in `ComfyUI/models/pixel_heads/` or `ComfyUI/models/vae/`. / "
        f"未找到 Pixel Head 权重 `{name}`，请放入 `ComfyUI/models/pixel_heads/` 或 `ComfyUI/models/vae/`。"
    )


class QwenImage21PixelHeadLoader:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "head_name": (_list_candidate_files(),),
                "precision": (["bf16_trunk_fp32_out", "fp32"], {"default": "bf16_trunk_fp32_out"}),
            }
        }

    RETURN_TYPES = ("QWEN21_PIXEL_HEAD",)
    RETURN_NAMES = ("pixel_head",)
    FUNCTION = "load_head"
    CATEGORY = "advanced/qwen_image21"
    TITLE = "Load Qwen-Image 2.1 Pixel Head"
    DESCRIPTION = (
        "Load Qwen-Image 2.1 Direct RGBA Pixel Head (~49 MB BF16 / ~97 MB FP32) from models/pixel_heads or models/vae.\n"
        "从 models/pixel_heads 或 models/vae 加载 Qwen-Image 2.1 免 VAE 直出像素头权重。"
    )

    def load_head(self, head_name: str, precision: str):
        path = _resolve_head_path(head_name)
        dtype = torch.bfloat16 if precision == "bf16_trunk_fp32_out" else torch.float32
        head = load_qwen21_pixel_head(path, device="cpu", dtype=dtype)
        return (head,)


class QwenImage21PixelHeadTap:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "model": ("MODEL",),
            }
        }

    RETURN_TYPES = ("MODEL", "QWEN21_TAP")
    RETURN_NAMES = ("model", "pixel_tap")
    FUNCTION = "install_tap"
    CATEGORY = "advanced/qwen_image21"
    TITLE = "Qwen-Image 2.1 Pixel Head Tap"
    DESCRIPTION = (
        "Non-invasive hook on Qwen-Image 2.1 MODEL right before KSampler. Supports T2I, <image1> Omni Edit, Cache & CFG.\n"
        "串联在进入 KSampler 前的 MODEL 连线上，无侵入截获末步 4096 维特征（兼容文生图、多图编辑、KV 缓存与 CFG）。"
    )

    def install_tap(self, model):
        patched_model, tap_state = attach_qwen21_tap(model)
        return (patched_model, tap_state)


class QwenImage21PixelDecode:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "samples": ("LATENT",),
                "pixel_head": ("QWEN21_PIXEL_HEAD",),
                "pixel_tap": ("QWEN21_TAP",),
                "output_scale": (["auto", "1x", "2x"], {"default": "auto"}),
            }
        }

    RETURN_TYPES = ("IMAGE", "MASK", "IMAGE")
    RETURN_NAMES = ("image_rgb", "alpha_mask", "image_rgba")
    FUNCTION = "decode"
    CATEGORY = "advanced/qwen_image21"
    TITLE = "Qwen-Image 2.1 Pixel Decode (Zero VAE)"
    DESCRIPTION = (
        "Replaces VAEDecode: decodes Qwen-Image 2.1 features directly to RGB + Alpha in ~5 ms with zero VAE VRAM.\n"
        "替代 VAEDecode：无需加载 644MB VAE，约 5 毫秒极速直出高清 RGB、Alpha 蒙版与 4 通道 RGBA 图像。"
    )

    @torch.no_grad()
    def decode(self, samples, pixel_head, pixel_tap, output_scale: str = "auto"):
        # `samples` dependency guarantees ComfyUI executes KSampler BEFORE this decode node!
        _ = samples["samples"]
        f_cpu, x0_hat_cpu, sigma = pixel_tap.resolve_final_features()

        device = comfy.model_management.get_torch_device()
        is_x2 = isinstance(pixel_head, QwenImage21PixelHeadX2)
        if output_scale == "auto":
            scale = 2 if is_x2 else 1
        elif output_scale == "2x":
            if not is_x2:
                raise ValueError(
                    "Loaded Pixel Head is 1x; load an `_x2` checkpoint for 2x output. / 当前为 1x 像素头权重，2x 直出请加载 `_x2` 权重。"
                )
            scale = 2
        else:
            scale = 1

        head_dev = pixel_head.to(device)
        trunk_dtype = head_dev.base.feat_proj.weight.dtype if is_x2 else head_dev.feat_proj.weight.dtype
        f_dev = f_cpu.to(device=device, dtype=trunk_dtype)
        x_dev = x0_hat_cpu.to(device=device, dtype=trunk_dtype)

        if torch.cuda.is_available() and device.type == "cuda":
            torch.cuda.synchronize(device)
        t0 = time.perf_counter()

        if is_x2:
            out_nchw = head_dev(f_dev, x_dev, scale=scale)
        else:
            out_nchw = head_dev(f_dev, x_dev)

        if torch.cuda.is_available() and device.type == "cuda":
            torch.cuda.synchronize(device)
        dt_ms = (time.perf_counter() - t0) * 1000.0

        # Convert [-1, 1] NCHW -> [0, 1] NHWC ComfyUI IMAGE format
        out_nhwc = ((out_nchw.float() + 1.0) * 0.5).clamp_(0.0, 1.0).movedim(1, -1).cpu()
        # Return head to CPU so idle VRAM stays zero
        pixel_head.to("cpu")

        if out_nhwc.shape[-1] == 4:
            rgb = out_nhwc[..., :3].contiguous()
            alpha = out_nhwc[..., 3].contiguous()
            rgba = out_nhwc.contiguous()
        else:
            rgb = out_nhwc[..., :3].contiguous()
            alpha = torch.ones(rgb.shape[:-1], dtype=torch.float32)
            rgba = torch.cat([rgb, alpha.unsqueeze(-1)], dim=-1)

        logger.info(
            f"[Qwen21PixelDecode] Decoded {tuple(rgb.shape[1:3])} (scale={scale}x, last_sigma={sigma:.4f}) in {dt_ms:.2f} ms (Zero VAE)"
        )
        return (rgb, alpha, rgba)


NODE_CLASS_MAPPINGS = {
    "QwenImage21PixelHeadLoader": QwenImage21PixelHeadLoader,
    "QwenImage21PixelHeadTap": QwenImage21PixelHeadTap,
    "QwenImage21PixelDecode": QwenImage21PixelDecode,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "QwenImage21PixelHeadLoader": "Load Qwen-Image 2.1 Pixel Head",
    "QwenImage21PixelHeadTap": "Qwen-Image 2.1 Pixel Head Tap",
    "QwenImage21PixelDecode": "Qwen-Image 2.1 Pixel Decode (Zero VAE)",
}
