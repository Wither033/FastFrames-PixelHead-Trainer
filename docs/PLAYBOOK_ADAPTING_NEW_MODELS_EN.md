# Deep-Dive Playbook: Adapting Direct Pixel Readout Heads to Any New DiT in 10 Minutes (with Real Debugging Post-Mortems)

This guide documents the exact step-by-step workflow we use to build a Zero-VAE Direct Pixel Head for any new 2D image or 3D video Diffusion Transformer in ComfyUI, followed by five real engineering bugs we hit and solved while building the `Qwen-Image 2.1`, `Krea-2`, `FLUX.3-Klein`, `LTX-2.5`, and `MiniMax-H3` heads.

---

## Part 1: The 5-Step DiT Surgery Checklist

When a new diffusion model lands in `comfy/ldm/<model_name>/model.py`, open its main `forward` / `_forward` method and answer five questions:

### Step 1 — Find the Post-AdaLN, Pre-Linear Feature Tensor `f`
Look at the very end of the Transformer trunk:
- Most DiTs apply an adaptive LayerNorm (`norm_out` or `final_layer.adaln_modulation`) followed immediately by a final `nn.Linear` (`proj_out` or `linear`).
- **Rule**: Always hook **after** the final LayerNorm/AdaLN modulation and **immediately before** the final `nn.Linear`.
- **Exception watch**: Check whether distilled checkpoints bypass `nn.Linear.forward`! For example, in `comfy/ldm/minimax/model.py`, 8-step PDD checkpoints (`n > 1`) call `F.linear(x, self.video_out.weight)` directly inside `_pdd_head`, so a hook on `video_out` never fires! Hooking `final_layer` itself and running the 3-line `mod(video_seg)` works identically on both Base (`n=1`) and PDD (`n>1`).

### Step 2 — Strip Non-Target Sequence Tokens
Check what tokens enter that final linear layer:
- **Already clean**: `Qwen-Image 2.1` slices `hidden_states[:, prefix_len:]` *before* `proj_out`; `FLUX.3-Klein` passes `x[:, :H*W]` into `final_layer["image"]`.
- **Concatenated single-stream**: `Krea-2` passes `combined = [context (txtlen), target_img (img_tokens), optional_ref_img (ref_tokens)]` into `last.linear`. Compute `img_tokens = gh * gw` and `ref_total` from `ref_latents`, then slice `comb[:, txtlen : txtlen + img_tokens, :]`.
- **Joint Audio-Video**: `LTX-2.5` uses a `NestedTensor` `(video_latent, audio_latent)` — unbind index `0` for `x0_hat_vid` and hook only the video `proj_out` so native audio decoding (`LTXVAudioVAEDecode`) keeps working untouched!

### Step 3 — Compute `(d_feat, d_lat)` and Check Output Overcompleteness
1. `d_feat` = input features of the hooked `nn.Linear` (e.g., `3072`, `4096`, `5376`, `6144`).
2. `d_lat` = `latent_channels * patch_t * patch_h * patch_w` (e.g., `Krea-2` has `16` channels and `patch=2`, so `d_lat = 16 * 2 * 2 = 64`; `MiniMax-H3` has `24` channels and `patch=(1,2,2)`, so `d_lat = 96`).
3. Pixel footprint per DiT token = `(vae_t * patch_t, vae_h * patch_h, vae_w * patch_w)`.
4. **Overcompleteness Rule**: Choose progressive `2×` `PixelShuffle` stages so that at the final grid resolution, the channel width `w_final` is **at least `2.5×` larger** than `out_dim = final_pt * final_ph * final_pw * out_channels`:
   - 2D `16×16` token → 1 stage `2×` shuffle → `8×8` patch (`out_dim = 8*8*3 = 192`, `w_final = 512` → **`2.67×` overcomplete**).
   - 3D `8t × 32×32` token (`LTX-2.5`) → 3 stages (`(1,2,2) -> (2,2,2) -> (2,2,2)`) → `2t × 4×4` patch (`out_dim = 2*4*4*3 = 96`, `w_final = 256` → **`2.67×` overcomplete**).
   - 3D `(1,4,4,4,4)t × 32×32` token (`MiniMax-H3`) → 3 spatial shuffles to `4×4` + temporal unpack onto unified frame grid → `1t × 4×4` patch (`out_dim = 48`, `w_final = 256` → **`5.33×` overcomplete**).

### Step 4 — Calibrate Dual-Regime Terminal `sigma_last`
Print the exact scheduler sigmas (`BasicScheduler` / `Flux2Scheduler` / `ManualSigmas`) used at inference:
- Record the **penultimate non-zero sigma** of the fast distilled mode (e.g., `0.31090` for 8-step `Krea-2` at `shift=1.15`; `0.421875` for `LTX-2.5`; `[0.444, 0.800]` for 8-step `MiniMax-H3` at `video_shift=12.0`).
- Record the multi-step Base terminal range (typically `[0.04, 0.20]`).
- Sample 50% from each regime during training!

### Step 5 — Run CPU Equivariance + 90-Step P0 Smoke Gate Before Full Training
Use `pixelhead_trainer.smoke_gate` (`verify_spatial_crop_equivariance_2d/3d` + `run_p0_smoke_gate_2d`) before starting a 3000-rollout run.

---

## Part 2: Five Real Debugging Post-Mortems

### Bug 1: The One-Line `lambda` Pre-Hook Trap (`EinopsError: 2282 != 2304`)
- **Symptom**: Writing `mod.register_forward_pre_hook(lambda _m, inp: cap.setdefault("f", inp[0][:, -img_tokens:, :].detach()))` crashed `SingleStreamDiT` inside `unpatchify`!
- **Root Cause**: In PyTorch, if a `forward_pre_hook` returns **anything other than `None`**, PyTorch replaces the module's input with that return value! `dict.setdefault(...)` returned the sliced `2304`-token tensor, overwriting the `2326`-token `[txt + img]` input to `last.linear`.
- **Rule**: Always write an explicit `def _pre_hook(_mod, inputs): ...` with no `return` statement (implicitly returning `None`).

### Bug 2: 3D `PixelShuffle` Axis-Scrambling Typo (`9.28 dB` Garbage Output)
- **Symptom**: An early `v1` 3D video head converged smoothly on square `6×6` patches in loss, yet produced `9.28 dB` scrambled stripes on full `512×320` non-square frames.
- **Root Cause**: Flattening `(B, T)` into `(B*T)` and calling a 5D `u.permute(0, 2, 4, 1, 3)` transposed spatial height `H` with `scale` and channel sub-blocks!
- **Fix**: Never fold `T` into `B` during 3D shuffle. Explicitly view into 7D `(B, T, H, W, C_out, sh, sw)` and permute `(0, 4, 1, 2, 5, 3, 6)` (see `pixelhead_trainer/blocks.py`), and always run `verify_spatial_crop_equivariance_3d` with `T != H != W`!

### Bug 3: Undercomplete Output Bottleneck on `32×32` Video Tokens (`23.86 dB` Ceiling)
- **Symptom**: Even after fixing axis order, predicting `8×8×4×3 = 768` or `8×8×8×3 = 1536` values from a `256`-channel trunk plateaued around `23.86 dB` with soft edges.
- **Root Cause**: Linearly projecting `256` dimensions into `1536` dimensions restricts every `8×8×8` spatiotemporal cube to a `256`-dimensional linear subspace!
- **Fix**: Adding a 3rd `PixelShuffle` stage to a `4px` spatial grid made the projection `256 -> 96` / `256 -> 48` (overcomplete) while actually *shrinking* parameter count to `~23.9M`.

### Bug 4: Corner Gradient Starvation in Random Small-Crop Smoke Gates
- **Symptom**: Sampling random `6×6` crops inside an `8×12` video grid during a 90-step P0 smoke check passed interior pixels (`21+ dB`) but failed full-frame evaluation (`16.86 dB`).
- **Root Cause**: Outer columns (`x=0` and `x=11`) were covered by only `1/7` of uniform random `6×6` placements — receiving just `~6` gradient updates out of `90`!
- **Fix**: Use full-frame supervision (or `50%` full-frame + `50%` margin-stripped window crops) so frame borders receive dense gradients from step 1.

### Bug 5: ComfyUI `post_cfg_function` Execution Timing
- **Symptom**: Inside `comfy/samplers.py`, `cfg_function` executes *after* `calc_cond_batch` exits its `try/finally` block!
- **Root Cause**: If `close_step()` clears `current_step = None`, a `post_cfg_hook` writing only to `state.current_step` silently drops `cond_scale` and `denoised`!
- **Fix**: In `post_cfg_hook`, always target `state.current_step if state.current_step is not None else state.last_valid_step` (implemented across all 5 `tap.py` modules).

### Bug 6: High-Compression 3D Video `MSE` Posterior-Mean Blur & Synthetic Metric Inflation (`MiniMax-H3 v0.2.0` Case Study)
- **Symptom**: On `MiniMax-H3` (`v0.2.0` Preview, [Wither033/ComfyUI-MiniMaxH3-PixelHead](https://github.com/Wither033/ComfyUI-MiniMaxH3-PixelHead)), aggregate held-out PSNR reached `31.03 dB` (`8.5×` faster, `45.7 MiB` vs `2.68 GB` ViT3D VAE), yet visual inspection revealed noticeable "plastic/soft-focus" smoothing on oil brushstrokes, eyes, and fast parallax occlusion (`25.91 dB`, `LPIPS = 0.29–0.37`).
- **Root Causes**:
  1. **Metric masking**: `25%` synthetic flat 2D cel/typography clips scored `36.73 dB` (`LPIPS = 0.0547`), masking `25.9–30.1 dB` natural-texture blur in the overall mean! Always report **per-regime PSNR + LPIPS** separately.
  2. **`16×` heavier token decompression + high $\sigma_{\text{last}}$ without perceptual loss**: 1 `MiniMax-H3` DiT token spans `4 × 32 × 32 = 4,096` pixels (`16×` more than `16×16` 2D image tokens) and denoises from $\sigma_{\text{last}} = 0.6316$ (`video_shift=12.0`). Pure pixel `L1 + 5.0 × MSE` without perceptual `LPIPS-VGG` / FFT supervision drives a `24M`-param head to predict smooth conditional expectations over uncertain micro-textures.
  3. **Premature `16px` latent folding**: `x0_hat_vid` (`[B, 24, T_lat, H_lat, W_lat]`) already lives natively at the `16px` grid (`2*gh × 2*gw`); folding it down to `32px` at entry discards native spatial sharpness instead of injecting it directly into the `16px` (`s16`) stage.
- **Fix (`v3 Sharp` Roadmap)**: Inject native-resolution `x0_hat_vid` directly at the `16px` stage, add multi-scale `LPIPS-VGG` + 2D FFT high-frequency amplitude loss during Stage-2 sharpening, widen 3D trunks (`width >= 960`), and calibrate purely on high-resolution natural footage.
