"""Non-invasive ComfyUI ModelPatcher tap for Krea-2 (`comfy.ldm.krea2.model.SingleStreamDiT`).

License: MIT
In `SingleStreamDiT._forward` (`comfy/ldm/krea2/model.py` lines 355-376):
  - `combined = torch.cat((context, img), dim=1)` where `img` holds `[target_img (img_tokens), optional_ref_img (ref_tokens)]`.
  - `self.last.linear` runs on the full `combined` sequence before `[:, txtlen : txtlen + img_tokens, :]`.
  - Inside our `last.linear` pre-hook, we read `gh = (H_lat + 1) // 2`, `gw = (W_lat + 1) // 2`, `img_tokens = gh * gw`,
    and `ref_total = sum(transformer_options.get("reference_image_num_tokens", []))` (note `SingleStreamDiT._forward`
    copies `transformer_options` at line 303, so if `ref_latents` was passed via kwargs, we compute exact `ref_total`
    directly from `kwargs.get("ref_latents")` as well!).
    Thus `txtlen = seq_len - img_tokens - ref_total` and `inputs[0][:, txtlen : txtlen + img_tokens, :]` extracts
    100% pure target image modulated features `[B, gh*gw, 6144]` for BOTH T2I and Style/Reference Editing!
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

import comfy.patcher_extension
import torch


TAP_WRAPPER_KEY = "krea2_pixel_head_tap"


def count_krea2_ref_tokens(ref_latents, patch: int = 2) -> int:
    if not ref_latents:
        return 0
    total = 0
    for ref in ref_latents:
        if ref is None:
            continue
        if ref.ndim == 5:
            _, _, rt, rh, rw = ref.shape
            total += rt * ((rh + patch - 1) // patch) * ((rw + patch - 1) // patch)
        else:
            rh, rw = ref.shape[-2], ref.shape[-1]
            total += ((rh + patch - 1) // patch) * ((rw + patch - 1) // patch)
    return total


@dataclass
class StepRecord:
    sigma: float
    forwards: list[dict] = field(default_factory=list)
    cond_scale: Optional[float] = None
    denoised: Optional[torch.Tensor] = None


class Krea2TapState:
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
        """Return `(f_combined [B, gh*gw, 6144], x0_hat_norm [B, 16, (1,) H_lat, W_lat], sigma)`."""
        step = self.last_valid_step
        if step is None or not step.forwards:
            raise RuntimeError(
                "Krea2PixelDecode: 未捕获到任何前向特征！请确认已将 `Krea-2 Pixel Head Tap` "
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
            raise RuntimeError("Krea2PixelDecode: 最后一步未找到正向条件 (cond=0) 特征记录。")

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


def attach_krea2_tap(model_patcher) -> tuple[object, Krea2TapState]:
    """Clone `model_patcher` and install lightweight step + `last.linear` pre-hooks."""
    cloned = model_patcher.clone()
    state = Krea2TapState()

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
        last_mod = getattr(diff_model, "last", None)
        linear_mod = getattr(last_mod, "linear", None)
        if linear_mod is None:
            raise RuntimeError(
                f"Krea2PixelHeadTap: 当前模型 ({type(diff_model).__name__}) 不包含 `last.linear`，请确认底模为 Krea-2。"
            )

        patch = int(getattr(diff_model, "patch", 2))
        h_lat, w_lat = x.shape[-2], x.shape[-1]
        gh = (h_lat + patch - 1) // patch
        gw = (w_lat + patch - 1) // patch
        img_tokens = gh * gw

        ref_method = kwargs.get("ref_latents_method", getattr(diff_model, "default_ref_method", None))
        ref_total = count_krea2_ref_tokens(kwargs.get("ref_latents"), patch) if ref_method is not None else 0

        rec = {
            "denoised": None,
            "f": None,
            "cond_or_uncond": list(cou),
        }

        def _pre_last_linear_hook(_mod, inputs):
            if rec["f"] is None and len(inputs) > 0:
                comb = inputs[0]
                seq_len = comb.shape[1]
                txtlen = seq_len - img_tokens - ref_total
                if txtlen < 0:
                    txtlen = seq_len - img_tokens
                f_img = comb[:, txtlen : txtlen + img_tokens, :]
                rec["f"] = f_img.detach().to(device="cpu", dtype=torch.bfloat16)

        handle = linear_mod.register_forward_pre_hook(_pre_last_linear_hook)
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
