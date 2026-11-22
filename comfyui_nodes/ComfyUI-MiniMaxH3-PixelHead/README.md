# ComfyUI-MiniMaxH3-PixelHead

Zero-Video-VAE 3D Non-Uniform Direct Pixel Readout Head (`v2` Overcomplete 3-Stage Spatial Shuffle + Unified Frame-Timeline 3D Refinement) for **MiniMax-H3 (Base & 8-Step PDD Turbo)** (`MiniMaxH3Model`) in ComfyUI.

Taps `diffusion_model.final_layer` (`5376`-dim `mod(video_seg)` tokens — compatible with both `8-Step PDD Turbo` 8-Step PDD Turbo and standard Base/Ref2VA checkpoints) + denoised `24`-ch `(1,2,2)` video latent (`d_lat=96`), using three `2x` spatial PixelShuffle stages (`32px -> 16px -> 8px -> 4px`), non-uniform `(1, 4, 4, 4, 4)` temporal expansion onto the unified frame timeline, frame-level causal 3D ConvNeXt refinement, and a `5.33x` overcomplete FP32 head (`256 -> 48`) to replace the multi-GB 36-layer ViT3D `MiniMaxH3VideoVAE`.

## License
MIT License (100% clean-room implementation).
