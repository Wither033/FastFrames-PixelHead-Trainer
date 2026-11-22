"""Non-invasive ComfyUI ModelPatcher tap for MiniMax-H3 (`comfy.ldm.minimax.model.MiniMaxH3Model`).

License: MIT
Why we hook `diffusion_model.final_layer` instead of `final_layer.video_out`:
  - In `comfy/ldm/minimax/model.py` lines 322-336, PDD 8-Step Turbo checkpoints (`minimax_h3_8stepTURBO...`, where `n > 1`)
    call `_pdd_head(self.video_out, mod(video_seg), ...)` using `F.linear` directly on `video_out.weight`, bypassing
    `video_out.forward`!
  - By registering a `register_forward_pre_hook` on `diffusion_model.final_layer` itself (`inputs = (h, t_emb, video_seg, ...)`),
    we compute the exact 3-line `mod(video_seg)`:
      `shift, scale = fl.adaln_proj(t_emb)`
      `f_vid = fl.norm(h[va:vb]) * (1.0 + scale[row]) + shift[row]`
    This works 100% identically across BOTH `8-Step PDD Turbo` 8-Step PDD Turbo (`n > 1`) AND standard `n = 1` Base / Ref2VA
    checkpoints, while automatically stripping text, keyframe, reference image, and audio tokens via `video_seg = (va, vb, row)`!
"""

from __future__ import annotations

from dataclasses import dataclass, field
import math
from typing import Optional

import comfy.patcher_extension
import torch


TAP_WRAPPER_KEY = "minimax_h3_pixel_head_tap"


def _extract_h3_video_latent(tensor: torch.Tensor, vid_shape: Optional[tuple[int, ...]]) -> torch.Tensor:
    if getattr(tensor, "is_nested", False):
        return tensor.unbind()[0]
    if tensor.ndim == 5:
        return tensor
    if vid_shape is not None:
        vid_elems = int(math.prod(vid_shape[1:]))
        b = tensor.shape[0]
        return tensor.reshape(b, -1)[:, :vid_elems].reshape(b, *vid_shape[1:])
    raise ValueError(f"Cannot unpack MiniMax-H3 video latent from shape {tuple(tensor.shape)} without latent_shapes")


@dataclass
class StepRecord:
    sigma: float
    vid_shape: Optional[tuple[int, ...]] = None
    forwards: list[dict] = field(default_factory=list)
    cond_scale: Optional[float] = None
    denoised_vid: Optional[torch.Tensor] = None


class MiniMaxH3TapState:
    """Stores only the latest actual MiniMax-H3 DiT forward step's video features on CPU."""

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
        """Return `(f_vid [B, T_lat*gh*gw, 5376], x0_hat_vid [B, 24, T_lat, H_lat, W_lat], sigma)`."""
        step = self.last_valid_step
        if step is None or not step.forwards:
            raise RuntimeError(
                "MiniMaxH3VideoPixelDecode: 未捕获到任何前向特征！请确认已将 `MiniMax-H3 Pixel Head Tap` "
                "串联在进入 BasicGuider / KSampler 的 MODEL 连线上（位于 MiniMaxH3SigmaShift 之后）。"
            )

        by_tag: dict[int, list[tuple[torch.Tensor, torch.Tensor]]] = {0: [], 1: []}
        for rec in step.forwards:
            cou = rec["cond_or_uncond"]
            n_chunks = len(cou)
            f_chunks = rec["f"].chunk(n_chunks, dim=0)
            d_chunks = rec["denoised_vid"].chunk(n_chunks, dim=0)
            for tag, fc, dc in zip(cou, f_chunks, d_chunks):
                by_tag.setdefault(int(tag), []).append((fc, dc))

        if not by_tag.get(0):
            raise RuntimeError("MiniMaxH3VideoPixelDecode: 最后一步未找到正向条件 (cond=0) 特征记录。")

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

        x0_hat_vid = step.denoised_vid.float() if step.denoised_vid is not None else x0_hat_fallback
        return f_final, x0_hat_vid, float(step.sigma)


def attach_minimax_h3_tap(model_patcher) -> tuple[object, MiniMaxH3TapState]:
    """Clone `model_patcher` and install lightweight step + `final_layer` pre-hooks for MiniMax-H3."""
    cloned = model_patcher.clone()
    state = MiniMaxH3TapState()

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
        if final_layer is None or not hasattr(final_layer, "adaln_proj"):
            raise RuntimeError(
                f"MiniMaxH3PixelHeadTap: 当前模型 ({type(diff_model).__name__}) 不包含 MiniMax-H3 `final_layer`，请确认底模为 MiniMax-H3 (Base & 8-Step PDD Turbo)。"
            )

        latent_shapes = kwargs.get("latent_shapes", getattr(base_model, "latent_shapes", None))
        if latent_shapes is not None and len(latent_shapes) > 0:
            vid_shape = tuple(int(s) for s in latent_shapes[0])
        elif x.ndim == 5:
            vid_shape = tuple(int(s) for s in x.shape)
        else:
            vid_shape = None
        step.vid_shape = vid_shape

        b_call = int(x.shape[0])
        rec = {
            "denoised_vid": None,
            "f": None,
            "cond_or_uncond": list(cou),
        }

        def _pre_final_hook(fl, inputs):
            if rec["f"] is None and len(inputs) >= 3:
                with torch.no_grad():
                    h, t_emb, video_seg = inputs[0], inputs[1], inputs[2]
                    va, vb, row = video_seg
                    shift, scale = fl.adaln_proj(t_emb)
                    sc = scale[row].to(scale.dtype)
                    sh = shift[row].to(shift.dtype)
                    f_vid = fl.norm(h[va:vb]) * (1.0 + sc) + sh
                    rec["f"] = f_vid.reshape(b_call, -1, f_vid.shape[-1]).detach().to(device="cpu", dtype=torch.bfloat16)

        handle = final_layer.register_forward_pre_hook(_pre_final_hook)
        try:
            out = executor(x, t, c_concat, c_crossattn, control, transformer_options, **kwargs)
        finally:
            handle.remove()

        if rec["f"] is not None:
            d_vid = _extract_h3_video_latent(out, vid_shape)
            rec["denoised_vid"] = d_vid.detach().to(device="cpu", dtype=torch.float32)
            step.forwards.append(rec)
        return out

    def post_cfg_hook(args):
        target_step = state.current_step if state.current_step is not None else state.last_valid_step
        if target_step is not None:
            if "cond_scale" in args and isinstance(args["cond_scale"], (int, float)):
                target_step.cond_scale = float(args["cond_scale"])
            if "denoised" in args and args["denoised"] is not None:
                try:
                    d_vid = _extract_h3_video_latent(args["denoised"], target_step.vid_shape)
                    target_step.denoised_vid = d_vid.detach().to(device="cpu", dtype=torch.float32)
                except Exception:
                    pass
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
