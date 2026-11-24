# Adapting Direct Pixel Readout Heads to New DiT Backbones (Notes & Debugging Log)

This document summarizes the five steps we use when adapting a direct pixel readout head to a new 2D image or 3D video DiT in ComfyUI, followed by seven lessons and issues we ran into while implementing `Qwen-Image 2.1`, `Krea-2`, `FLUX.3-Klein`, `LTX-2.5`, and `MiniMax-H3`.

---

## Part 1: Five-Step Inspection Checklist

Open `comfy/ldm/<model_name>/model.py` and check its main `forward` / `_forward` path:

### 1. Locate the Post-AdaLN, Pre-Linear Tensor `f`
- Most DiTs run an adaptive normalization (`norm_out` or `final_layer.adaln_modulation`) right before the final `nn.Linear` (`proj_out` or `linear`).
- Attach the `register_forward_pre_hook` **after** the final normalization/modulation and **before** the final `nn.Linear`.
- Check whether distilled checkpoints bypass `nn.Linear.forward`: in `comfy/ldm/minimax/model.py`, 8-step PDD checkpoints (`n > 1`) call `F.linear(x, self.video_out.weight)` inside `_pdd_head`, so a hook on `video_out` never runs. Hooking `final_layer` itself and evaluating the 3-line `mod(video_seg)` works for both Base (`n = 1`) and PDD (`n > 1`).

### 2. Strip Non-Target Tokens
Check what else is concatenated into the sequence entering that linear layer:
- **Already sliced**: `Qwen-Image 2.1` slices `hidden_states[:, prefix_len:]` before `proj_out`; `FLUX.3-Klein` passes `x[:, :H*W]` into `final_layer["image"]`.
- **Single-stream concatenation**: `Krea-2` feeds `[context (txtlen), target_img (img_tokens), optional_ref_img (ref_tokens)]` into `last.linear`. Compute `img_tokens = gh * gw` and `ref_total` from `ref_latents`, then slice `comb[:, txtlen : txtlen + img_tokens, :]`.
- **Audio + Video `NestedTensor`**: `LTX-2.5` packs `(video_latent, audio_latent)` together. Unbind index `0` for `x0_hat_vid` and hook only the video `proj_out` so `LTXVAudioVAEDecode` continues to work normally.

### 3. Check `(d_feat, d_lat)` and Output Overcompleteness
1. `d_feat`: input width of the hooked `nn.Linear` (`3072`, `4096`, `5376`, `6144`).
2. `d_lat`: `latent_channels * patch_t * patch_h * patch_w` (`Krea-2` has `16` channels and `patch = 2`, giving `d_lat = 64`; `MiniMax-H3` has `24` channels and `patch = (1, 2, 2)`, giving `d_lat = 96`).
3. Output pixel volume per DiT token: `(vae_t * patch_t, vae_h * patch_h, vae_w * patch_w)`.
4. Keep the final channel width `w_final >= 2.5 × out_dim` (`out_dim = final_pt * final_ph * final_pw * out_channels`):
   - 2D `16×16` token → one `2×` shuffle → `8×8` patch (`out_dim = 192`, `w_final = 512`).
   - 3D `8t × 32×32` token (`LTX-2.5`) → three shuffles `(1,2,2) -> (2,2,2) -> (2,2,2)` → `2t × 4×4` patch (`out_dim = 96`, `w_final = 256`).
   - 3D `(1,4,4,4,4)t × 32×32` token (`MiniMax-H3`) → three spatial shuffles to `4×4` + temporal unpack onto a unified frame grid → `1t × 4×4` patch (`out_dim = 48`, `w_final = 256`).

### 4. Record Actual Terminal `sigma_last` Values
Inspect the `sigmas` tensor produced by ComfyUI's scheduler (`BasicScheduler` / `Flux2Scheduler` / `ManualSigmas`) and record the **last non-zero sigma**:
- Fast distilled schedule (e.g. `0.31090` for 8-step `Krea-2` at `shift = 1.15`; `0.421875` for `LTX-2.5`; `[0.444, 0.800]` for 8-step `MiniMax-H3` at `video_shift = 12.0`).
- Multi-step Base schedule (typically `[0.04, 0.20]`).
- Sample 50% from each range during training.

### 5. Run Pre-Flight Checks Before Full Training
Run `verify_spatial_crop_equivariance_2d/3d` and a 90-step P0 smoke check (`pixelhead_trainer.smoke_gate`) before starting a full training run.

---

## Part 2: Seven Issues & Architecture Lessons During Development

### 1. Non-`None` Return Value in a `lambda` Pre-Hook (`EinopsError: 2282 != 2304`)
- **What happened**: `mod.register_forward_pre_hook(lambda _m, inp: cap.setdefault("f", inp[0][:, -img_tokens:, :].detach()))` failed inside `SingleStreamDiT` unpatchify.
- **Cause**: In PyTorch, returning anything other than `None` from `register_forward_pre_hook` replaces the module's input. `dict.setdefault(...)` returned the `2304`-token slice and overwrote `last.linear`'s `2326`-token `[txt + img]` input.
- **Fix**: Use an explicit `def _pre_hook(_mod, inputs): ...` function with no `return` statement.

### 2. Axis Permutation Error in 3D `PixelShuffle` (`9.28 dB` Striped Output)
- **What happened**: An early `v1` 3D video head lowered loss on square `6×6` crops during training, but produced `9.28 dB` striped output on `512×320` non-square frames.
- **Cause**: Merging `(B, T)` into `(B*T)` and calling `u.permute(0, 2, 4, 1, 3)` swapped height `H` with the spatial upscale factor.
- **Fix**: Keep `T` explicit, view into 7D `(B, T, H, W, C_out, sh, sw)`, permute `(0, 4, 1, 2, 5, 3, 6)`, and test with `T != H != W` in `verify_spatial_crop_equivariance_3d`.

### 3. Undercomplete Output Projection on `32×32` Video Tokens (`23.86 dB` Ceiling)
- **What happened**: Predicting `8×8×4×3 = 768` or `8×8×8×3 = 1536` values linearly from a `256`-channel trunk plateaued around `23.86 dB`.
- **Cause**: A `256 → 1536` linear layer restricts every `8×8×8` spatiotemporal patch to a `256`-dim linear subspace.
- **Fix**: Adding a third `PixelShuffle` stage to a `4px` grid changed `out_proj` to `256 → 96` / `256 → 48` (overcomplete) while keeping total parameter count around `23.9M`.

### 4. Border Under-Sampling in Small Random Crops
- **What happened**: Sampling random `6×6` crops inside an `8×12` grid during a 90-step smoke check reached `21+ dB` on interior pixels but `16.86 dB` on the full frame.
- **Cause**: Outer columns (`x = 0` and `x = 11`) were included in only `1/7` of uniform random placements (~6 updates in 90 steps).
- **Fix**: Run smoke gates on full frames, and mix full-frame updates with margin-stripped window crops during training.

### 5. ComfyUI `post_cfg_function` Hook Timing
- **What happened**: In `comfy/samplers.py`, `cfg_function` runs *after* `calc_cond_batch` exits its `try/finally` block.
- **Cause**: Clearing `state.current_step = None` in `close_step()` caused `post_cfg_hook` to miss `cond_scale` and `denoised` if it only checked `state.current_step`.
- **Fix**: Write to `state.current_step if state.current_step is not None else state.last_valid_step`.

### 6. High-Frequency Smoothing on `32×32` 3D Video Heads Under Pure Pixel `MSE` (`MiniMax-H3 v0.2.0`)
- **What happened**: On `MiniMax-H3` (`v0.2.0` preview), overall held-out PSNR reached `31.03 dB` (`8.5×` faster, `45.7 MiB` vs `2.68 GB` ViT3D VAE), but natural brushwork, facial details, and fast foreground occlusion (`25.91 dB`, `LPIPS = 0.29–0.37`) looked noticeably smoothed.
- **Causes**:
  1. Synthetic flat `kinetic_cel_typo` clips (`25%` of the calibration set) scored `36.73 dB` (`LPIPS = 0.0547`), pulling the overall average up while natural subsets sat at `25.9–30.1 dB`. Always evaluate per subset and check `LPIPS` alongside PSNR.
  2. Each `MiniMax-H3` DiT token covers `4 × 32 × 32 = 4,096` pixels (`16×` more than `16×16` 2D image tokens) and denoises from $\sigma_{\text{last}} = 0.6316$ (`video_shift = 12.0`). Without perceptual (`LPIPS-VGG`) or frequency-domain losses, `5.0 × MSE` pushes a `24M`-param trunk to output smooth conditional averages over fine textures.
  3. `x0_hat_vid` (`[B, 24, T_lat, H_lat, W_lat]`) already sits on the `16px` spatial grid; folding it `2×2` down to the `32px` entry layer discards native `16px` spatial localization.
- **Planned changes**: Inject `x0_hat_vid` directly at the `16px` (`s16`) stage, add `LPIPS-VGG` + 2D FFT amplitude loss in a second sharpening stage, increase trunk width (`width >= 960`), and train on higher-resolution natural clips without synthetic flat filler.


### 7. Recovering `1–2px` Micro-Detail Without Full-Resolution Norm Bottlenecks (`Qwen-Image 2.1 v0.1 → v0.2 → v0.3`)
- **What happened**: `v0.1` (`8×8` linear unpatchify, `7.77 ms` on L4 `1024²`) softened `2px` ticket text and portrait pores (`LPIPS = 0.1272`). `v0.2` added a full `16px → 8px → 4px → 2px → 1px` pyramid with `GroupNorm + depthwise` blocks and `LPIPS + FFT` supervision, cutting `LPIPS` to `0.0808` (`+0.40 dB` PSNR), but latency rose `14.8×` to `115.05 ms` (`904 MiB` peak activation VRAM) and `1px` copperplate cross-hatching barely moved (`26.82 → 26.85 dB`).
- **Causes**:
  1. `GroupNorm` + depthwise separable convolutions at `1024×1024` are memory-bandwidth bound on consumer/cloud GPUs.
  2. Injecting `x0_hat_norm` only via a `1×1` per-token linear projection forces deeper layers to re-infer cross-token diagonal stroke continuity from scratch.
  3. Standard pixel-space GANs trade off `0.5–1.5 dB` of PSNR via low-frequency color/luminance shifts.
- **Fix (`v0.3`)**:
  1. Stop neural layers at `512×512` (`2px`), use norm-free `3×3` `WaveletConvBlock2d` blocks to predict 2D Haar wavelet subbands `(LL, LH, HL, HH)`, and reconstruct `1024×1024` via analytical `0`-parameter `idwt_haar_2d` (`37.51 ms` on L4 — `3.07×` faster than `v0.2`, `504 MiB` VRAM).
  2. Add `LatentSpatialHighway` (`3×3` convs on `64`-ch `x0_hat_norm` injected at `8px`, `4px`, and `2px`).
  3. Train a Wavelet PatchGAN on `torch.cat([LL.detach(), LH, HL, HH], dim=1)` so adversarial gradients sharpen edges without shifting `LL` luminance, reaching **`33.81 dB` PSNR** (`39.17 dB` on portraits, `+0.56 dB` on etchings) and **`0.0736` LPIPS**.
