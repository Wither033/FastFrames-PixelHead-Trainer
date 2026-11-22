# ComfyUI-MiniMaxH3-PixelHead *(Experimental Preview `v0.2.0` — Not Recommended for Production Yet)*

> [!WARNING]
> **⚠Experimental Preview (`v0.2.0`) — Not Recommended for Production Use Yet (`v3 Sharp` Optimization Upcoming)**
>
> Standalone repository & weights:
> - **GitHub**: https://github.com/Wither033/ComfyUI-MiniMaxH3-PixelHead
> - **Hugging Face**: https://huggingface.co/pocis/MiniMax-H3-Video-PixelHead
>
> While global color, motion, `(1, 4, 4, 4, 4)` `17k + 5` frame alignment, and flat 2D cel/typography (`36.73 dB`) work cleanly (`45.7 MiB` BF16 vs `2.68 GB` ViT3D VAE, `8.5×` faster), **complex natural micro-textures and fast foreground parallax occlusion (`25.91 dB`, `LPIPS = 0.29–0.37`) currently exhibit noticeable high-frequency over-smoothing ("plastic blur")** due to pure pixel-space `MSE` training at $\sigma_{\text{last}} = 0.6316$ without perceptual `LPIPS` sharpening. Please wait for the upcoming `v3 Sharp` release before using in production video workflows!

## Architecture Summary (`v2` Overcomplete)
Taps `diffusion_model.final_layer` (`5376`-dim `mod(video_seg)` tokens — compatible with both 8-Step PDD Turbo and standard Base/Ref2VA checkpoints) + denoised `24`-ch `(1,2,2)` video latent (`d_lat=96`), using three `2x` spatial PixelShuffle stages (`32px -> 16px -> 8px -> 4px`), non-uniform `(1, 4, 4, 4, 4)` temporal expansion onto the unified frame timeline, frame-level causal 3D ConvNeXt refinement, and a `5.33x` overcomplete FP32 head (`256 -> 48`).

## License
MIT License (100% clean-room implementation).
