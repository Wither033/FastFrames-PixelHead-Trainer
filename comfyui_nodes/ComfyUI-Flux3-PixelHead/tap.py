"""Non-invasive ComfyUI ModelPatcher tap for FLUX.3-Image-Klein (`comfy.ldm.flux3.model.Flux3`).

License: MIT
Captures the final denoising step's pre-linear hidden states `f` (`[B, H*W, 3072]`) from
`diffusion_model.final_layer["image"].linear` and the exact CFG-combined denoised latent
`x0_hat_norm` (`[B, 128, H_lat, W_lat]`), supporting:
  - Distilled FlowMap (`flux3-image-klein`, 4~16 steps, CFG = 1.0)
  - Base CFG (`flux3-image-klein-base`, 25~50 steps, CFG > 1.0 via `f_u + cfg * (f_c - f_u)`)
  - Text-to-Image (T2I), Image-to-Image (I2I), and Multi-Reference Image Editing (MI2I)
    (`final_layer["image"]` already receives `x[:, :H * W]`, stripping reference & text tokens automatically!)
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

import comfy.patcher_extension
import torch


TAP_WRAPPER_KEY = "flux3_pixel_head_tap"


@dataclass
class StepRecord:
    sigma: float
    forwards: list[dict] = field(default_factory=list)
    cond_scale: Optional[float] = None
    denoised: Optional[torch.Tensor] = None


class Flux3TapState:
    """Stores only the latest actual DiT forward step's features on CPU."""

    def __init__(self):
        self.current_step: Optional[StepRecord] = None
        self.last_valid_step: Optional[StepRecord] = None

    def open_step(self, sigma: float) -> None:
        self.current_step = StepRecord(sigma=sigma)

    def close_step(self) -> None:
        if self.current_step is not None and len(self.current_step.forwards) > 0:
            self.last_valid_step = self.current_step
        self.current_step = None

    def resolve_final_features(self) -> tuple[torch.Tensor, torch.Tensor, float]:
        """Return `(f_combined [B, H*W, 3072], x0_hat_norm [B, 128, H_lat, W_lat], sigma)`."""
        step = self.last_valid_step
        if step is None or not step.forwards:
            raise RuntimeError(
                "Flux3PixelDecode: 未捕获到任何前向特征！请确认已将 `FLUX.3 Pixel Head Tap` "
                "串联在进入 KSampler 的 MODEL 连线上，且本次运行实际执行了采样步骤。"
            )

        by_tag: dict[int, list[tuple[torch.Tensor, torch.Tensor]]] = {0: [], 1: []}
        for rec in step.forwards:
            cou = rec["cond_or_uncond"]
            n_chunks = len(cou)
            f_chunks = rec["f"].chunk(n_chunks, dim=0)
            d_chunks = rec["denoised"].chunk(n_chunks, dim=0)
            for tag, fc, dc in zip(cou, f_chunks, d_chunks):
                by_tag.setdefault(int(tag), []).append((fc, dc))

        if not by_tag.get(0):
            raise RuntimeError("Flux3PixelDecode: 最后一步未找到正向条件 (cond=0) 特征记录。")

        f_cond = torch.cat([p[0] for p in by_tag[0]], dim=0).float()
        d_cond = torch.cat([p[1] for p in by_tag[0]], dim=0).float()

        if by_tag.get(1) and step.cond_scale is not None and abs(step.cond_scale - 1.0) > 1e-5:
            f_uncond = torch.cat([p[0] for p in by_tag[1]], dim=0).float()
            d_uncond = torch.cat([p[1] for p in by_tag[1]], dim=0).float()
            scale = float(step.cond_scale)
            f_final = f_uncond + scale * (f_cond - f_uncond)
            x0_hat_fallback = d_uncond + scale * (d_cond - d_uncond)
        else:
            f_final = f_cond
            x0_hat_fallback = d_cond

        x0_hat_norm = step.denoised.float() if step.denoised is not None else x0_hat_fallback
        return f_final, x0_hat_norm, float(step.sigma)


def attach_flux3_tap(model_patcher) -> tuple[object, Flux3TapState]:
    """Clone `model_patcher` and install lightweight step + `final_layer['image'].linear` pre-hooks."""
    cloned = model_patcher.clone()
    state = Flux3TapState()

    cloned.remove_wrappers_with_key(comfy.patcher_extension.WrappersMP.CALC_COND_BATCH, TAP_WRAPPER_KEY)
    cloned.remove_wrappers_with_key(comfy.patcher_extension.WrappersMP.APPLY_MODEL, TAP_WRAPPER_KEY)

    def calc_cond_batch_wrapper(executor, model, conds, x_in, timestep, model_options):
        sig = float(timestep.flatten()[0].detach().cpu().item())
        state.open_step(sig)
        try:
            return executor(model, conds, x_in, timestep, model_options)
        finally:
            state.close_step()

    def apply_model_wrapper(executor, x, t, c_concat=None, c_crossattn=None, control=None, transformer_options={}, **kwargs):
        step = state.current_step
        cou = (transformer_options or {}).get("cond_or_uncond")
        if step is None or not cou:
            return executor(x, t, c_concat, c_crossattn, control, transformer_options, **kwargs)

        base_model = executor.class_obj
        diff_model = getattr(base_model, "diffusion_model", None)
        final_layer = getattr(diff_model, "final_layer", None)
        img_head = final_layer["image"] if isinstance(final_layer, (dict, torch.nn.ModuleDict)) and "image" in final_layer else None
        linear_mod = getattr(img_head, "linear", None)
        if linear_mod is None:
            raise RuntimeError(
                f"Flux3PixelHeadTap: 当前模型 ({type(diff_model).__name__}) 不包含 `final_layer['image'].linear`，请确认底模为 FLUX.3-Image-Klein。"
            )

        rec = {
            "denoised": None,
            "f": None,
            "cond_or_uncond": list(cou),
        }

        def _pre_linear_hook(_mod, inputs):
            if rec["f"] is None and len(inputs) > 0:
                rec["f"] = inputs[0].detach().to(device="cpu", dtype=torch.bfloat16)

        handle = linear_mod.register_forward_pre_hook(_pre_linear_hook)
        try:
            out = executor(x, t, c_concat, c_crossattn, control, transformer_options, **kwargs)
        finally:
            handle.remove()

        if rec["f"] is not None:
            rec["denoised"] = out.detach().to(device="cpu", dtype=torch.float32)
            step.forwards.append(rec)
        return out

    def post_cfg_hook(args):
        target_step = state.current_step if state.current_step is not None else state.last_valid_step
        if target_step is not None:
            if "cond_scale" in args:
                target_step.cond_scale = float(args["cond_scale"])
            if "denoised" in args and args["denoised"] is not None:
                target_step.denoised = args["denoised"].detach().to(device="cpu", dtype=torch.float32)
        return args["denoised"]

    cloned.add_wrapper_with_key(
        comfy.patcher_extension.WrappersMP.CALC_COND_BATCH,
        TAP_WRAPPER_KEY,
        calc_cond_batch_wrapper,
    )
    cloned.add_wrapper_with_key(
        comfy.patcher_extension.WrappersMP.APPLY_MODEL,
        TAP_WRAPPER_KEY,
        apply_model_wrapper,
    )
    cloned.set_model_sampler_post_cfg_function(post_cfg_hook)
    cloned.model_options[TAP_WRAPPER_KEY] = state
    return cloned, state
