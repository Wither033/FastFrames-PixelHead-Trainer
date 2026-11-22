# ComfyUI-QwenImage21-PixelHead

Zero-VAE Direct Pixel Readout Head (`RGB` + `Alpha / RGBA`, `1×` & `2×`) for **Qwen-Image 2.1** (`Qwen/Qwen-Image-2.1`, `Qwen-Image-2.1-Omni`, and `8-Step` / `Turbo` variants) in ComfyUI.

Inspired by *FastFrames: Video Generation at Interactive Latency* (`arXiv:2611.23226`), this node pack taps the pre-`proj_out` contextualized `4096`-dim DiT features and `64`-ch `x_t` latent at the final denoising step, decoding directly to pixels in **~4–8 ms** without loading the `644 MB` 64-channel Qwen-Image 2.1 VAE decoder.

## Nodes

1. **`Load Qwen-Image 2.1 Pixel Head`** (`QwenImage21PixelHeadLoader`)
   - Loads `qwen_image_2.1_pixel_head.safetensors` (`1×`) or `qwen_image_2.1_pixel_head_x2.safetensors` (`2×`) from `ComfyUI/models/pixel_heads/` or `ComfyUI/models/vae/`.
2. **`Qwen-Image 2.1 Pixel Head Tap`** (`QwenImage21PixelHeadTap`)
   - Insert on the `MODEL` wire right before `KSampler` / `SamplerCustomAdvanced`.
   - Compatible with `QwenImage21Cache` (`auto` / `gpu` / `cpu` / `off`), `CFG = 1.0` (8-step distilled) and `CFG > 1.0` (Base), as well as both **Text-to-Image (T2I)** and **Multi-Image Reference Editing (`<image1>`, `<image2>`)**.
3. **`Qwen-Image 2.1 Pixel Decode (Zero VAE)`** (`QwenImage21PixelDecode`)
   - Replaces `VAEDecode`. Outputs `image_rgb` (`[B, H, W, 3]`), `alpha_mask` (`[B, H, W]`), and `image_rgba` (`[B, H, W, 4]`).

## License

MIT License (100% clean-room implementation).
