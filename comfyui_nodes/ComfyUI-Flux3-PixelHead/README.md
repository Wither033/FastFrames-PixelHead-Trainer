# ComfyUI-Flux3-PixelHead

Zero-VAE Direct RGB Pixel Readout Head (`1x` & `2x`) for **FLUX.3-Image-Klein** (`flux3-image-klein` FlowMap Distilled & `flux3-image-klein-base`) in ComfyUI.

Taps `diffusion_model.final_layer["image"].linear` (`3072`-dim modulated image tokens) + denoised `128`-ch latent `x0_hat_norm` at the final sampling step, decoding directly to RGB in **~4–6 ms** without loading `flux2-vae.safetensors` (`312 MiB`).

## License
MIT License — adapted from [Seunghyun Ji (`sorryhyun`)](https://github.com/sorryhyun)'s [`anima_lora`](https://github.com/sorryhyun/anima_lora) / [`sorryhyun/anima_pixel_decoder`](https://huggingface.co/sorryhyun/anima-pixel-decoder) (`Copyright (c) 2026 Seunghyun Ji`) and [*FastFrames* (`arXiv:2611.23226`)](https://arxiv.org/abs/2611.23226).
