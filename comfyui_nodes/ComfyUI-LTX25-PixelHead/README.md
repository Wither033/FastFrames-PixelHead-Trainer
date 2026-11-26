# ComfyUI-LTX25-PixelHead

Zero-Video-VAE 3D Causal Direct Pixel Readout Head (`v2` Overcomplete Spatiotemporal Shuffle) for **LTX-2.5** (`LTXAVModel` / `LTXVModel`) in ComfyUI.

Taps `diffusion_model.proj_out` (`4096`-dim modulated video tokens) + denoised `128`-ch video latent `x0_hat_vid` at the final step, using a 3-stage 3D PixelShuffle cascade (`(8t, 32px) -> (8t, 16px) -> (4t, 8px) -> (2t, 4px)`) with a `2.67x` overcomplete FP32 head (`256 -> 96`) and causal `8*T_lat - 7` alignment to replace the heavy 5-stage `CausalDiffusionVAE` (`ltx-2.5-video-vae-bf16.safetensors`), while keeping audio decoding (`LTXVAudioVAEDecode`) intact.

## License
MIT License — adapted from [Seunghyun Ji (`sorryhyun`)](https://github.com/sorryhyun)'s [`anima_lora`](https://github.com/sorryhyun/anima_lora) / [`sorryhyun/anima_pixel_decoder`](https://huggingface.co/sorryhyun/anima-pixel-decoder) (`Copyright (c) 2026 Seunghyun Ji`) and [*FastFrames* (`arXiv:2611.23226`)](https://arxiv.org/abs/2611.23226).
