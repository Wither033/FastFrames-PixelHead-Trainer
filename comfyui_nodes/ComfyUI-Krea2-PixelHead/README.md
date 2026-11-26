# ComfyUI-Krea2-PixelHead

Zero-VAE Bridged Direct RGB Pixel Readout Head (`1x` & `2x`) for **Krea-2** (`krea2_turbo` & `krea2_raw`, `SingleStreamDiT`) in ComfyUI.

Taps `diffusion_model.last.linear` (`6144`-dim modulated target image tokens, automatically stripping text & style-reference tokens) + denoised `16`-ch `2x2` patched latent `x0_hat_norm` (`d_lat=64`), decoding directly to RGB in **~5 ms** without loading `qwen_image_vae.safetensors`.

## License
MIT License — adapted from [Seunghyun Ji (`sorryhyun`)](https://github.com/sorryhyun)'s [`anima_lora`](https://github.com/sorryhyun/anima_lora) / [`sorryhyun/anima_pixel_decoder`](https://huggingface.co/sorryhyun/anima-pixel-decoder) (`Copyright (c) 2026 Seunghyun Ji`) and [*FastFrames* (`arXiv:2611.23226`)](https://arxiv.org/abs/2611.23226).
