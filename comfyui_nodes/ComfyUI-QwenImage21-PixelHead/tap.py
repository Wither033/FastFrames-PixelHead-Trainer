"""Non-invasive ComfyUI ModelPatcher tap for Qwen-Image 2.1 (`QwenImage21Transformer2DModel`).

License: MIT
Captures the final denoising step's pre-`proj_out` normalized hidden states `f` (`[B, H*W, 4096]`)
and the exact CFG-combined denoised normalized latent `x0_hat_norm` (`[B, 64, H_lat, W_lat]`), supporting:
  - CFG = 1.0 (8-step Turbo / distilled) AND CFG > 1.0 (`f_u + cfg * (f_c - f_u)`)
  - `QwenImage21Cache` KV prefix caching ON or OFF
  - Both Text-to-Image (T2I) and Multi-Image Reference Editing (`<image1>`, `<image2>`, ...)
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

import comfy.patcher_extension
import torch


TAP_WRAPPER_KEY = "qwen21_pixel_head_tap"


@dataclass
class StepRecord:
    sigma: float
    forwards: list[dict] = field(default_factory=list)
    cond_scale: Optional[float] = None
    denoised: Optional[torch.Tensor] = None


class Qwen21TapState:
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
        """Return `(f_combined [B, H*W, 4096], x0_hat_norm [B, 64, H_lat, W_lat], sigma)`."""
        step = self.last_valid_step
        if step is None or not step.forwards:
            raise RuntimeError(
                "QwenImage21PixelDecode: No DiT features captured! Connect `Qwen-Image 2.1 Pixel Head Tap` "
                "on the MODEL wire before KSampler. / 未捕获到前向特征！请将 `Qwen-Image 2.1 Pixel Head Tap` 串联在进入采样器的 MODEL 连线上。"
            )

        # Group recorded chunks by ComfyUI's cond_or_uncond tag (0 = positive/cond, 1 = negative/uncond)
        by_tag: dict[int, list[tuple[torch.Tensor, torch.Tensor]]] = {0: [], 1: []}
        for rec in step.forwards:
            cou = rec["cond_or_uncond"]
            n_chunks = len(cou)
            f_chunks = rec["f"].chunk(n_chunks, dim=0)
            d_chunks = rec["denoised"].chunk(n_chunks, dim=0)
            for tag, fc, dc in zip(cou, f_chunks, d_chunks):
                by_tag.setdefault(int(tag), []).append((fc, dc))

        if not by_tag.get(0):
            raise RuntimeError(
                "QwenImage21PixelDecode: Missing positive condition (cond=0) features at final step. / 最后一步缺少正向条件特征记录。"
            )

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


def attach_qwen21_tap(model_patcher) -> tuple[object, Qwen21TapState]:
    """Clone `model_patcher` and install lightweight step + `proj_out` pre-hooks."""
    cloned = model_patcher.clone()
    state = Qwen21TapState()

    # Remove any prior tap hooks if re-attached
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
        proj_out = getattr(diff_model, "proj_out", None)
        if proj_out is None:
            raise RuntimeError(
                f"QwenImage21PixelHeadTap: Model ({type(diff_model).__name__}) has no `proj_out`; expected Qwen-Image 2.1. / "
                f"当前模型 ({type(diff_model).__name__}) 不包含 `proj_out`，请确认底模为 Qwen-Image 2.1。"
            )

        rec = {
            "denoised": None,
            "f": None,
            "cond_or_uncond": list(cou),
        }

        def _pre_proj_hook(_mod, inputs):
            if rec["f"] is None and len(inputs) > 0:
                rec["f"] = inputs[0].detach().to(device="cpu", dtype=torch.bfloat16)

        handle = proj_out.register_forward_pre_hook(_pre_proj_hook)
        try:
            # BaseModel.apply_model returns calculate_denoised(sigma, model_output, x) = x0_hat_norm
            out = executor(x, t, c_concat, c_crossattn, control, transformer_options, **kwargs)
        finally:
            handle.remove()

        if rec["f"] is not None:
            rec["denoised"] = out.detach().to(device="cpu", dtype=torch.float32)
            step.forwards.append(rec)
        return out

    def post_cfg_hook(args):
        # In comfy/samplers.py, `cfg_function` runs right AFTER `calc_cond_batch` returns,
        # so `state.last_valid_step` holds the step that just finished.
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
