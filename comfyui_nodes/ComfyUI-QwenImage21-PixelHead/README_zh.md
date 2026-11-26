# ComfyUI-QwenImage21-PixelHead

用于 **Qwen-Image 2.1** 的 ComfyUI 免 VAE 像素直出节点（现已发布 **`v0.3` 小波高保真版** 与 `v0.1` 快速版 `1x / 2x`，支持 4 通道 RGBA），用来替代采样末尾 `644.2 MiB` 的 64 通道 WanVAE 解码器。

[English](README.md) | [简体中文](README_zh.md)

[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)
[![Hugging Face](https://img.shields.io/badge/Hugging%20Face-pocis%2FQwen--Image--2.1--PixelHead-yellow)](https://huggingface.co/pocis/Qwen-Image-2.1-PixelHead)
[![Comfy Registry](https://img.shields.io/badge/Comfy%20Registry-comfyui--qwenimage21--pixelhead-blueviolet)](https://registry.comfy.org/publishers/wither033/nodes/comfyui-qwenimage21-pixelhead)

---

## `v0.3` 更新说明（2026-11-24 · 2D Haar 小波直出 + 多尺度潜空间空间桥 + 高频小波对抗精修）

`v0.3`（`qwen_image_2.1_pixel_head_v03_bf16.safetensors`，`79.3 MiB`）针对 `v0.2` 全图 `1024×1024` 归一化层耗时偏高及微缩排线仍有提升空间的问题，进行了三项结构改进，在解码速度、峰值显存和三类盲测指标上同时超过 `v0.2`：

- **2D Haar 小波子带预测 + 零参数解析逆变换（`iDWT`）**：不再在 `1024×1024` 全分辨率网格跑高开销的 `GroupNorm + 逐通道卷积`，改为在 `512×512` 网格用无归一化融合残差块直接预测 `(LL, LH, HL, HH)` 四个正交小波子带并解析重构 `1024×1024` 像素。在 NVIDIA L4 上解码耗时从 `v0.2` 的 `115.05 ms` 降至 **`37.51 ms`（比 `v0.2` 快 `3.07×`，比 `671.57 ms` 的官方 `644.2 MiB` VAE 快 `17.9×`）**，运行期激活显存下降 **`44.2%`**（`904 MiB → 504 MiB`）；
- **多尺度潜空间 `3×3` 空间直通桥（`LatentSpatialHighway`）**：对去噪后的 `64` 通道潜变量 `x0_hat_norm` 直接做带邻域感受野的 `3×3` 空间卷积，跨级注入 `8px`、`4px` 与 `2px` 小波层，改善跨 Token 边界的斜排微缩字（条码票据局部达 `36.2 dB`）与 `1px` 铜版画交叉排线（`fine_art` 子集比 `v0.2` 提升 `+0.56 dB`）；
- **锁死低频 `LL.detach()` 的纯高频小波 PatchGAN**：判别器仅作用于高频边缘子带 `(LH, HL, HH)` 并截断对低频底色 `LL` 的梯度，避免常规 GAN 偏色掉分，使 `1024×1024` 留出集总体 `PSNR` 升至 **`33.81 dB`**（比 `v0.2` 高 `+0.79 dB`，比 `v0.1` 高 `+1.19 dB`；近景人像子集均值达 **`39.17 dB`**，单张局部 `41.2 dB`），`LPIPS-VGG` 降至 **`0.0736`**（比 `v0.1` 下降 `42.1%`，比 `v0.2` 再降 `8.9%`）；
- **升级提示**：加载 `v0.3` 权重前请先更新本插件至 **`v1.3.0+`**（`git pull` 或通过 ComfyUI Manager 更新），同一个加载节点可自动识别 `v0.3`、`v0.2` 与 `v0.1`（`1x / 2x`）权重。

![1:1 像素四方同屏对比：官方 644MB VAE ｜ v0.1 ｜ v0.2 ｜ v0.3 小波版](assets/v03_hero_1to1.png)

*（完整 6 组 `1:1` 原尺寸四方盲测大图见 [`assets/v03_4way_1to1_comparison.png`](assets/v03_4way_1to1_comparison.png)）*

---

## 主要特性

- **免加载 `644.2 MiB` 大 VAE 省显存**：避免高分辨率 `VAEDecode` 产生 `1.9～7.5 GB` 峰值显存，算完自动将权重退回内存，并内置 `2K/4K` 余弦平滑分块解码开关（`tile_mode`）。
- **权重分档说明**：
  - **`v0.3` 小波高保真版（`79.3 MiB` BF16 · 默认推荐）**：L4 解码 `37.51 ms`（RTX 4060 Ti 约 `47.9 ms`），总体 `33.81 dB` PSNR（近景人像 `39.17 dB`）、`0.0736` LPIPS-VGG；
  - **`v0.1` 快速预览版（`48.8 MiB` `1x` / `52.1 MiB` `1x+2x`）**：L4 解码 `7.77 ms`（A100 约 `3.4 ms`），适合大批量抽卡试构图，并提供一步 `2x`（`1024 → 2048`）快速放大分支；
  - **`v0.2` 全分辨率金字塔版（`78.5 MiB` BF16）**：保留在 Hugging Face 仓库供对比复现（L4 解码 `115.05 ms`，`33.02 dB`）。
- **单权重兼容常用模式**：通过 `SigmaAdaLN` 同时兼容 8 步加速 LoRA（`CFG = 1.0`）与 20～35 步常规采样（`CFG = 3.5～4.5`），支持文生图、`<image1>` 多参考图编辑、官方 `QwenImage21Cache` 缓存以及 4 通道 RGBA 输出。

---

## 实测对比数据（NVIDIA L4 · `1024×1024` 多步留出盲测集：`8步 Turbo` / `20步 CFG=3.5` / `24步 CFG=4.5`）

以 `Comfy-Org/Qwen-Image-2.1` 官方 `qwen_image_2.1_vae_bf16.safetensors` 为基准，在 36 张留出的 `1024×1024` 真机多步生成样本上实测：

| 解码器 | BF16 体积 | `1024²` 耗时 (L4) | 相对速度 | 峰值激活显存 | 总体 PSNR (↑) | 总体 LPIPS-VGG (↓) |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **官方 Qwen-Image 2.1 VAE** | `644.2 MiB` | `671.57 ms` | `1.0×` | `~1.9–3.2 GB` | *(基准)* | `0.0000` |
| **`PixelHead v0.3`（Haar 小波 + 高频 GAN）** | **`79.3 MiB`** | **`37.51 ms`** | **`17.9×`** | **`504 MiB`** | **`33.81 dB`** | **`0.0736` (`-42.1%`)** |
| `PixelHead v0.2`（1px 全分辨率金字塔） | `78.5 MiB` | `115.05 ms` | `5.8×` | `904 MiB` | `33.02 dB` | `0.0808` |
| `PixelHead v0.1`（8×8 线性块展开） | `48.8 MiB` | `7.77 ms` | `86.4×` | `< 200 MiB` | `32.62 dB` | `0.1272` |

### 分场景细项对比（`PSNR ↑ / LPIPS-VGG ↓`）

| 测试子集（`1024×1024` 留出集） | `v0.1`（8×8 线性） | `v0.2`（1px 金字塔） | **`v0.3`（Haar 小波 + 空间桥）** | `v0.3` 相比 `v0.2` 提升 |
| :--- | :---: | :---: | :---: | :--- |
| **`portrait`（近景人像肤质/胡须发丝）** | `37.29 dB` / `0.1211` | `38.25 dB` / `0.0630` | **`39.17 dB` / `0.0572`** | **`+0.92 dB` PSNR，`-9.2%` LPIPS** |
| **`small_text`（斜排车票条码与路牌小字）** | `32.50 dB` / `0.1235` | `32.80 dB` / `0.0792` | **`33.61 dB` / `0.0727`** | **`+0.81 dB` PSNR，`-8.2%` LPIPS** |
| **`fine_art`（铜版画细密交叉排线与木刻）** | `26.82 dB` / `0.1490` | `26.85 dB` / `0.1107` | **`27.41 dB` / `0.0987`** | **`+0.56 dB` PSNR，`-10.8%` LPIPS** |

---

## 安装与权重下载

### 1. 安装或更新插件（使用 `v0.3` 需更新至 `v1.3.0+`）
在 ComfyUI Manager 中更新 `ComfyUI-QwenImage21-PixelHead`，或在插件目录执行：

```bash
cd ComfyUI/custom_nodes/ComfyUI-QwenImage21-PixelHead
git pull
```

*（首次安装请在 `ComfyUI/custom_nodes` 下执行 `git clone https://github.com/Wither033/ComfyUI-QwenImage21-PixelHead.git`。）*

### 2. 下载权重
从 [pocis/Qwen-Image-2.1-PixelHead](https://huggingface.co/pocis/Qwen-Image-2.1-PixelHead) 下载模型，放入 `ComfyUI/models/pixel_heads/`（放入 `ComfyUI/models/vae/` 同样可识别）：

- **`qwen_image_2.1_pixel_head_v03_bf16.safetensors`（`79.3 MiB` · 默认推荐）**：`v0.3` Haar 小波高保真 `1x` 权重，兼顾 `37.5 ms` 低延迟与人像、小字、版画排线细节。
- `qwen_image_2.1_pixel_head_bf16.safetensors`（`48.8 MiB`）：`v0.1` 快速抽卡 `1x` 权重（L4 `7.77 ms`）。
- `qwen_image_2.1_pixel_head_x2_bf16.safetensors`（`52.1 MiB`）：`v0.1` 含 `1x + 2x` 快速放大分支权重。
- `qwen_image_2.1_pixel_head_v02_bf16.safetensors`（`78.5 MiB`）：上一版 `v0.2` 权重（仓库内同时提供各版本的完整 `FP32` 权重）。

---

## 节点连接方式

[`example_workflows/`](example_workflows/) 目录下附带了默认指向 `v0.3` 权重的中英文工作流：
- 中文：[`01_QwenImage21_PixelHead_免VAE极速直出与对比_ZH.json`](example_workflows/01_QwenImage21_PixelHead_免VAE极速直出与对比_ZH.json)
- 英文：[`01_QwenImage21_PixelHead_ZeroVAE_EN.json`](example_workflows/01_QwenImage21_PixelHead_ZeroVAE_EN.json)

1. 把 `Qwen-Image 2.1 Pixel Head Tap` 串在进入 `KSampler` 之前的 `MODEL` 连线上；
2. 用 `Qwen-Image 2.1 Pixel Decode (Zero VAE)` 替换 `VAEDecode`，接入 `LATENT`、`pixel_head` 和 `pixel_tap`。

---

## 同系列仓库

| 适配模型 | ComfyUI 节点仓库 | Hugging Face 权重 | 说明 |
| :--- | :--- | :--- | :--- |
| **Qwen-Image 2.1** *(本仓库)* | [Wither033/ComfyUI-QwenImage21-PixelHead](https://github.com/Wither033/ComfyUI-QwenImage21-PixelHead) | [pocis/Qwen-Image-2.1-PixelHead](https://huggingface.co/pocis/Qwen-Image-2.1-PixelHead) | **`v0.3` 小波版 (`79.3 MiB`, `37.5 ms`, `LPIPS 0.0736`)** + `v0.1` 快速版 (`48.8 MiB`)，4 通道 RGBA |
| **Krea-2** | [Wither033/ComfyUI-Krea2-PixelHead](https://github.com/Wither033/ComfyUI-Krea2-PixelHead) | [pocis/Krea-2-PixelHead](https://huggingface.co/pocis/Krea-2-PixelHead) | `52.6 MiB` BF16，`6144d` 降维桥 RGB，`59.55 ms` (`24.7×`) |
| **MiniMax-H3** *(预览版)* | [Wither033/ComfyUI-MiniMaxH3-PixelHead](https://github.com/Wither033/ComfyUI-MiniMaxH3-PixelHead) | [pocis/MiniMax-H3-Video-PixelHead](https://huggingface.co/pocis/MiniMax-H3-Video-PixelHead) | 早期实验预览版（复杂运动碎纹理偏软） |
| **训练代码与方法整理** | [Wither033/FastFrames-PixelHead-Trainer](https://github.com/Wither033/FastFrames-PixelHead-Trainer) | [HF 合集页](https://huggingface.co/collections/pocis/fastframes-zero-vae-direct-pixelheads-comfyui-6b031df8e9212f8bae294154) | 通用训练框架、挂载说明与 `01`～`07` 训练脚本 |

## 客观局限说明

- **极高密度 `1px` 木刻版画微纹理**：`v0.3` 在近景人像（`39.17 dB`）、斜排票据小字（`33.61 dB`）与铜版交叉排线上已非常接近 `644 MiB` 官方 VAE；但在全屏 `1px` 极密黑白木刻羽毛纹样（`17.9 dB`，`LPIPS 0.105`）上仍比官方重型卷积解码器略柔和一些。
- **离线 `.latent` 文件**：解码依赖采样最后一步实时截获的 `4096` 维 DiT 特征，无法单独解码外部保存的 `.latent` 文件。

## 原作者署名、致谢与开源协议


- **原作者与核心管线致谢**：本项目的像素头核心结构、ComfyUI `Tap + Decode` 挂载范式及训练管线（包括无跳连 ConvNeXt + `2× PixelShuffle` `1x/2x` 解码结构、末层线性映射前 `f_u + g·(f_c - f_u)` 特征截获与潜变量块融合、基于 `VAE(x̂0)` 的跨 Base/Turbo 单步 `ReplayBuffer` 蒸馏、RGBA 四通道读出，以及 `v0.3` 的 2D Haar 小波子带预测与锁定低频 `LL` 的高频对抗训练思路）衍生并改编自原作者 **[Seunghyun Ji (`sorryhyun`)](https://huggingface.co/sorryhyun)**（[GitHub `@sorryhyun`](https://github.com/sorryhyun)）在 **[`sorryhyun/anima_lora`](https://github.com/sorryhyun/anima_lora)**、**[`sorryhyun/anima_pixel_decoder`](https://huggingface.co/sorryhyun/anima-pixel-decoder)** 与 **[`sorryhyun/ComfyUI-Anima-PixelDecoder`](https://github.com/sorryhyun/ComfyUI-Anima-PixelDecoder)** 中的开创性开源工作（MIT License，`Copyright (c) 2026 Seunghyun Ji`）。特此向原作者 `sorryhyun` 致谢！
- **基础论文参考**：直接像素读出（Direct Pixel Readout）理论框架参考 [*FastFrames: Video Generation at Interactive Latency* (`arXiv:2611.23226`)](https://arxiv.org/abs/2611.23226)（Gene Chou et al., Apple & Cornell, 2026）。
- **开源协议**：[MIT License](LICENSE)（完整保留原作者 `Copyright (c) 2026 Seunghyun Ji` 版权声明；配合各扩散底座模型使用时请同时遵守对应底模的许可证条款）。
