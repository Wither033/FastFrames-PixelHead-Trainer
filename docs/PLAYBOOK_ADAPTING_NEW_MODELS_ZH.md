# 深度实战手册：10 分钟为任意新扩散模型适配 PixelHead（附 5 个真实踩坑复盘）

本文档完整沉淀了我们在为 `Qwen-Image 2.1`、`Krea-2`、`FLUX.3-Image-Klein`、`LTX-2.5` 与 `MiniMax-H3` 五大图像/视频底座从零研发 PixelHead 时的**标准 5 步源码解剖法**，以及过程中真实踩过并彻底解决的 **5 个硬核工程 Bug**。照着这份手册，你可以把 `pixelhead_trainer` 快速迁移到 `Wan-2.1/2.2`、`HunyuanVideo`、`Chroma`、`SD3.5` 或未来任何新发布的 DiT 模型上。

---

## 第一部分：任意新模型 5 步源码解剖清单

当 ComfyUI 新增了一个扩散模型（位于 `comfy/ldm/<模型名>/model.py`）时，打开它的主 `forward` / `_forward` 函数，按顺序确认以下 5 件事：

### 第 1 步：找到「末层 AdaLN 调制之后、最终线性层之前」的特征张量 `f`
查看 Transformer 主干最末尾的几行代码：
- 绝大多数 DiT 都会先做一次自适应归一化调制（`norm_out` 或 `final_layer.adaln_modulation`），紧接着送入最后一个 `nn.Linear`（`proj_out` 或 `linear`）。
- **铁律**：钩子必须挂在**最后一次归一化调制之后、最终 `nn.Linear` 之前**。
- **隐藏陷阱排查**：务必看一眼加速蒸馏版权重是否绕过了 `nn.Linear.forward`！典型例子是 `MiniMax-H3`（`comfy/ldm/minimax/model.py`）：其 8步 PDD 蒸馏模型在 `_pdd_head` 内直接调用 `F.linear(x, self.video_out.weight)`，导致挂在 `video_out` 上的钩子根本不会触发！解决办法是退半步挂在 `final_layer` 入口，用 3 行无梯度代码复算一次 `mod(video_seg)`，即可 100% 通吃原版与 PDD 蒸馏版。

### 第 2 步：剥离非目标图像/视频的序列前缀与后缀 Token
检查进入末层线性层的序列到底拼了哪些东西：
- **天然已剥离**：`Qwen-Image 2.1` 在进 `proj_out` 前已经执行了 `hidden_states[:, prefix_len:]`；`FLUX.3-Klein` 在进 `final_layer["image"]` 前已经切出了 `x[:, :H*W]`。
- **单流拼接序列**：`Krea-2`（`SingleStreamDiT`）把 `[文本 txtlen, 目标图像 img_tokens, 可选参考图 ref_tokens]` 拼在一起送进 `last.linear`。此时只需算出目标图像网格 `img_tokens = gh * gw` 及参考图 `ref_total`，按 `comb[:, txtlen : txtlen + img_tokens, :]` 切片即可同时兼容文生图与风格参考图编辑。
- **音画多模态打包**：`LTX-2.5` 使用 `NestedTensor` 同时打包视频与音频潜变量——只需解绑第 `0` 项提取视频 `x0_hat_vid` 并只拦截视频 `proj_out`，用户原有的轻量音频 VAE 解码器（`LTXVAudioVAEDecode`）就能原封不动保留！

### 第 3 步：核算 `(d_feat, d_lat)` 并严守「末层过完备法则」
1. `d_feat` = 被挂钩 `nn.Linear` 的输入特征维度（如 `3072`、`4096`、`5376`、`6144`）。
2. `d_lat` = `潜空间通道数 × DiT_patch_t × DiT_patch_h × DiT_patch_w`（例如 `Krea-2` 潜空间 `16` 通道且 `patch=2`，故 `d_lat = 16 × 2 × 2 = 64`；`MiniMax-H3` 潜空间 `24` 通道且 `patch=(1,2,2)`，故 `d_lat = 96`）。
3. 单个 DiT Token 对应的输出像素体积 = `(VAE时间倍率 × patch_t, VAE空间倍率 × patch_h, VAE空间倍率 × patch_w)`。
4. **末层过完备铁律**：设计渐进 `2× PixelShuffle` 级数，确保到达最后一层 `out_proj` 时，输入通道数 `w_final` **至少是输出像素维度 `out_dim` 的 `2.5` 倍以上**：
   - 2D `16×16` 图像头：1 级 `2×` 上采样至 `8×8` 块，`out_dim = 8×8×3 = 192`，`w_final = 512`（**`2.67×` 过完备**）。
   - 3D `8t × 32×32` 视频头（`LTX-2.5`）：3 级时空上采样 `(1,2,2) -> (2,2,2) -> (2,2,2)` 至 `2t × 4×4` 块，`out_dim = 2×4×4×3 = 96`，`w_final = 256`（**`2.67×` 过完备**）。
   - 3D `(1,4,4,4,4)t × 32×32` 非均匀视频头（`MiniMax-H3`）：3 级空间上采样至 `4×4` + 展开至统一帧时间轴因果精修，`out_dim = 1×4×4×3 = 48`，`w_final = 256`（**`5.33×` 过完备**）。

### 第 4 步：标定「蒸馏加速 + 多步精修」双轨末步噪声 `sigma_last`
打印出该模型在 ComfyUI 实际推理时的 `sigmas` 数列，记录**倒数第二个值（即最后一步去噪起点的非零 `sigma`）**：
- 蒸馏加速档（如 `Krea-2` 8步 `shift=1.15` 末步精确为 `0.31090`；`LTX-2.5` 为 `0.421875`；`MiniMax-H3` `video_shift=12.0` 8步末步区间为 `[0.444, 0.800]`）；
- 常规多步档（20~40步末步通常落在 `[0.04, 0.20]`）；
- 训练时 **50% / 50% 交替抽样**，即可让单文件权重在两种工作流下都丝滑清晰。

### 第 5 步：正式开训前必过「非正方形裁剪等变自检 + 90 步 P0 门禁」
调用 `pixelhead_trainer.smoke_gate` 中的验证函数，10 秒确认无误后再跑完整的 `2000~3000` Rollouts。

---

## 第二部分：五个真实硬核踩坑复盘（避坑指南）

### 坑 1：一行 `lambda` Pre-Hook 引发的张量篡改惨案（`EinopsError: 2282 != 2304`）
- **现象**：评测脚本里顺手写了句 `last.linear.register_forward_pre_hook(lambda _m, inp: cap.setdefault("f", inp[0][:, -img_tokens:, :].detach()))`，结果 DiT 前向直接在 `unpatchify` 报维度不匹配崩溃！
- **根因**：PyTorch 的 `forward_pre_hook` 有一个极易被忽略的机制——**只要钩子函数返回值不是 `None`，PyTorch 就会拿这个返回值直接替换掉原模块的输入参数！** `dict.setdefault(...)` 返回了切片后的 `2304` 个图像 token，直接把原本 `2326` 长度的 `[文本+图像]` 序列覆盖了。
- **铁律**：所有特征截获钩子一律写成显式 `def _pre_hook(_mod, inputs): ...` 函数且不写 `return`（默认返回 `None`）。

### 坑 2：3D `PixelShuffle` 维度折叠错位（测试集只有 `9.28 dB` 彩条乱码）
- **现象**：早期 `v1` 视频头在小方块裁剪训练时 Loss 一路下降，结果一跑 `512×320` 非正方形全画幅测试集只有 `9.28 dB` 满屏条纹。
- **根因**：把 `(B, T)` 合并后对 5D 张量写了错误的 `u.permute(0, 2, 4, 1, 3)`，把空间高度 `H` 和子像素倍率轴混排在了一起！
- **解法**：在 `spatial_pixel_shuffle_3d` 与 `spatiotemporal_pixel_shuffle_3d` 中绝不折叠 `T` 维，显式 `view` 成 7D `(B, T, H, W, C_out, sh, sw)` / 8D `(B, T, H, W, C_out, pt, ps, ps)` 后精确 `permute`，并用 `T != H != W` 的非正方形张量跑 `verify_spatial_crop_equivariance_3d` 锁死精度（误差 `< 1e-6`）。

### 坑 3：`32×32` 大感受野视频 Token 的「欠完备线性瓶颈」（卡在 `23.86 dB` 发糊）
- **现象**：修对轴顺序后，若在 `8×8` 像素网格直接用 `256` 维特征线性预测 `8×8×4×3 = 768` 或 `8×8×8×3 = 1536` 个像素值，PSNR 死活卡在 `~23.86 dB`，高频发糊。
- **根因**：拿 `256` 个数线性张成 `1536` 个数，意味着每个时空块被死死锁在 `1/6` 维度的线性子空间里，根本表达不出自由的斜向运动边缘！
- **解法**：再加第 3 级 `PixelShuffle` 下探到 `4×4` 网格，末层瞬间变成 `256 -> 96` / `256 -> 48` 过完备映射，参数量反而降到 `~23.9M`，清晰度质的飞跃。

### 坑 4：随机小窗口裁剪导致的画幅边缘「梯度饥饿」
- **现象**：在 `8×12` 的视频 Token 网格里随机切 `6×6` 窗口跑 90 步门禁时，中心区域已达 `21+ dB`，但测全画幅只有 `16.86 dB`。
- **根因**：均匀随机起点 `x0 in [0..6]` 时，中心列每次都被覆盖，而最左/最右边缘列只有 `1/7` 的概率被抽中——90 步里边缘像素只更新了约 `6` 次！
- **解法**：门禁阶段直接跑全画幅监督；正式训练采用「全画幅 + 去边界局部窗口混合更新」，确保画面四角与中心获得同等充足的梯度。

### 坑 5：ComfyUI `post_cfg_function` 的真实触发时机陷阱
- **现象**：在 `comfy/samplers.py` 内部，`cfg_function` 是在 `calc_cond_batch` 的 `try...finally` 块**完全退出之后**才被调用的！
- **根因**：如果 `close_step()` 把 `state.current_step` 清成了 `None`，而 `post_cfg_hook` 只往 `state.current_step` 里写 `cond_scale` 和 `denoised`，就会静默丢掉 CFG 融合潜变量！
- **解法**：在所有 `tap.py` 的 `post_cfg_hook` 中统一写入 `state.current_step if state.current_step is not None else state.last_valid_step`。

### 坑 6：高压缩比 3D 视频头的「纯 `MSE` 回归均值磨皮陷阱」与平均分假象（`MiniMax-H3 v0.2.0` 实战复盘）
- **现象**：在 `MiniMax-H3` 预览版（[Wither033/ComfyUI-MiniMaxH3-PixelHead](https://github.com/Wither033/ComfyUI-MiniMaxH3-PixelHead)）实训中，虽然总体平均分跑到了 `31.03 dB`（体积 `45.7 MiB` vs 官方 `2.68 GB` ViT3D VAE，提速 `8.5×`），但肉眼检视发现复杂油画笔触、人物眉眼与快速视差遮挡（`25.91 dB`，`LPIPS = 0.29~0.37`）存在明显的柔焦涂抹与塑料磨皮感。
- **三大根因**：
  1. **简单样本拉高平均分的假象**：校准集中占比 `25%` 的纯色扁平文字动画跑出了 `36.73 dB`（`LPIPS = 0.0547`），掩盖了真实复杂纹理子集只有 `25.9~30.1 dB` 的短板——**视频评测必须按纹理/运动复杂度拆表汇报 PSNR 与 LPIPS**。
  2. **`16×` 解压负担 + 高末步噪声（$\sigma_{\text{last}} = 0.6316$）+ 缺少感知锐化损失**：`MiniMax-H3` 单个 DiT Token 需同时解压 `4帧 × 32×32 = 4,096` 个像素（负担为 `16×16` 图像头的 `16` 倍），且 `video_shift=12.0` 下 8 步末步起点噪声高达 `0.6316`。若训练期只用纯像素 `L1 + 5.0 × MSE` 而不加 VGG/LPIPS 感知损失与频域损失，`24M` 小网络面对不确定高频细节必然输出「平滑条件期望值（磨皮）」。
  3. **`16px` 原生潜变量提前降采样折损**：`x0_hat_vid`（`[B, 24, T_lat, H_lat, W_lat]`）天生处于 `16px` 精细空间网格，在入口将其 `2×2` 折叠进 `32px` 粗网格损失了一半直接定位锐度。
- **解法（`v3 Sharp` 升级路线）**：在 `16px`（`s16`）阶段原位跨层直注未折叠的 `24ch` `x0_hat_vid`，在训练后半程挂载多尺度 `LPIPS-VGG` + 2D FFT 高频幅度损失，将 3D 主干拓宽至 `width >= 960`，并剔除纯色合成块、改用纯高细节自然运动片段训练。
