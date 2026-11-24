# 新扩散模型适配 PixelHead 流程与问题记录

本文档整理了为 ComfyUI 中的 2D 图像或 3D 视频 DiT 模型适配免 VAE 像素直出头（PixelHead）的五个排查步骤，以及在开发 `Qwen-Image 2.1`、`Krea-2`、`FLUX.3-Klein`、`LTX-2.5` 和 `MiniMax-H3` 过程中实际遇到并处理过的七个问题与结构迭代经验。

---

## 第一部分：新模型适配的五个排查步骤

打开 `comfy/ldm/<模型名>/model.py`，找到主干网络的 `forward` 或 `_forward` 函数，依次确认以下五项：

### 1. 找到末层归一化调制之后、最终线性层之前的张量 `f`
- 大多数 DiT 在末尾会先做一次自适应归一化调制（`norm_out` 或 `final_layer.adaln_modulation`），紧接着进入最后一个 `nn.Linear`（`proj_out` 或 `linear`）。
- 将 `register_forward_pre_hook` 挂在**最后一次归一化调制之后、最终 `nn.Linear` 之前**。
- 留意蒸馏模型是否绕过了 `nn.Linear.forward`：例如 `comfy/ldm/minimax/model.py` 中的 8 步 PDD 模型（`n > 1`）在 `_pdd_head` 内直接调用 `F.linear(x, self.video_out.weight)`，挂在 `video_out` 上的钩子不会执行。退一步挂在 `final_layer` 入口并用 3 行代码算出 `mod(video_seg)`，即可同时兼容原版（`n = 1`）和 PDD 版（`n > 1`）。

### 2. 切掉非目标图像/视频的序列 Token
查看进入末层线性层的序列包含哪些分段：
- **进入前已切干净**：`Qwen-Image 2.1` 在进 `proj_out` 前已执行 `hidden_states[:, prefix_len:]`；`FLUX.3-Klein` 在进 `final_layer["image"]` 前已切出 `x[:, :H*W]`。
- **单流拼接序列**：`Krea-2`（`SingleStreamDiT`）将 `[文本 txtlen, 目标图像 img_tokens, 参考图 ref_tokens]` 拼在一起送入 `last.linear`。先由潜变量尺寸算出 `img_tokens = gh * gw` 及参考图 `ref_total`，再按 `comb[:, txtlen : txtlen + img_tokens, :]` 切片即可。
- **音画打包序列**：`LTX-2.5` 用 `NestedTensor` 打包视频和音频潜变量，取第 `0` 项作为 `x0_hat_vid` 并只挂钩视频 `proj_out`，原有的音频解码节点 `LTXVAudioVAEDecode` 不受影响。

### 3. 核对 `(d_feat, d_lat)` 与末层过完备比例
1. `d_feat`：被挂钩 `nn.Linear` 的输入特征维度（如 `3072`、`4096`、`5376`、`6144`）。
2. `d_lat`：`潜空间通道数 × patch_t × patch_h × patch_w`（例如 `Krea-2` 为 `16 × 2 × 2 = 64`；`MiniMax-H3` 为 `24 × 1 × 2 × 2 = 96`）。
3. 单个 DiT Token 对应的输出像素范围：`(VAE时间倍率 × patch_t, VAE空间倍率 × patch_h, VAE空间倍率 × patch_w)`。
4. 通过多级 `2× PixelShuffle`，保证最后一层 `out_proj` 的输入通道数 `w_final` 至少是输出维度 `out_dim` 的 `2.5` 倍以上：
   - 2D `16×16` 图像头：1 级 `2×` 上采样到 `8×8` 块（`out_dim = 192`，`w_final = 512`）。
   - 3D `8t × 32×32` 视频头（`LTX-2.5`）：3 级时空上采样 `(1,2,2) -> (2,2,2) -> (2,2,2)` 到 `2t × 4×4` 块（`out_dim = 96`，`w_final = 256`）。
   - 3D `(1,4,4,4,4)t × 32×32` 视频头（`MiniMax-H3`）：3 级空间上采样到 `4×4` + 展开到统一帧时间轴（`out_dim = 48`，`w_final = 256`）。

### 4. 记录实际推理时的末步噪声 `sigma_last`
打印该模型在 ComfyUI 采样器生成的 `sigmas` 序列，取倒数第二个值（最后一个非零 `sigma`）：
- 加速蒸馏档（如 `Krea-2` 8 步 `shift = 1.15` 末步为 `0.31090`；`LTX-2.5` 为 `0.421875`；`MiniMax-H3` `video_shift = 12.0` 8 步末步区间为 `[0.444, 0.800]`）；
- 常规多步档（20～40 步末步一般在 `[0.04, 0.20]`）；
- 训练时按 50% / 50% 交替抽样。

### 5. 开训前先跑等变自检与 90 步冒烟测试
正式跑长程训练前，先调用 `pixelhead_trainer.smoke_gate` 检查异形张量裁剪等变性与 90 步 PSNR 爬升情况。

---

## 第二部分：开发过程中遇到的七个问题与迭代记录

### 1. `lambda` 写法导致 `forward_pre_hook` 覆盖模块输入（`EinopsError: 2282 != 2304`）
- **现象**：评测脚本里写了 `last.linear.register_forward_pre_hook(lambda _m, inp: cap.setdefault("f", inp[0][:, -img_tokens:, :].detach()))`，前向在 `unpatchify` 处报维度不一致。
- **原因**：PyTorch 中如果 `forward_pre_hook` 返回了非 `None` 值，会直接用该返回值替换原模块的输入。`dict.setdefault(...)` 返回了切片后的 `2304` 个图像 token，覆盖了原本长 `2326` 的 `[文本 + 图像]` 序列。
- **处理**：钩子统一写成显式 `def _pre_hook(_mod, inputs): ...` 且不写 `return`。

### 2. 3D `PixelShuffle` 维度排列写错（非正方形画幅只有 `9.28 dB`）
- **现象**：早期 `v1` 视频头在正方形小窗口上 Loss 正常下降，但跑 `512×320` 全画幅测试只有 `9.28 dB` 条纹。
- **原因**：将 `(B, T)` 合并后对 5D 张量执行 `u.permute(0, 2, 4, 1, 3)`，把高度 `H` 与上采样子块轴排混了。
- **处理**：保留独立的 `T` 维，显式 `view` 为 7D `(B, T, H, W, C_out, sh, sw)` 后按 `(0, 4, 1, 2, 5, 3, 6)` 置换，并用 `T != H != W` 的张量做裁剪等变测试。

### 3. `32×32` 视频 Token 的末层线性降秩瓶颈（卡在 `23.86 dB`）
- **现象**：修对轴顺序后，在 `8×8` 网格用 `256` 维特征线性预测 `768` 或 `1536` 个像素值，PSNR 停在 `23.86 dB` 左右，边缘发虚。
- **原因**：用 `256` 维直接线性映射到 `1536` 维，每个时空块被限制在 `256` 维线性子空间内。
- **处理**：增加第 3 级 `PixelShuffle` 到 `4×4` 网格，把末层改为 `256 → 96` / `256 → 48` 的过完备映射，总参数量仍保持在 `23.9M` 左右。

### 4. 小窗口随机裁剪导致画幅四角更新不足
- **现象**：在 `8×12` 网格内随机裁 `6×6` 窗口跑 90 步冒烟测试时，中心区域超过 `21 dB`，但全画幅测试只有 `16.86 dB`。
- **原因**：均匀随机取起点时，最左和最右两列只有 `1/7` 的概率被裁中，90 步里只更新了约 6 次。
- **处理**：冒烟门禁直接跑全画幅；正式训练改为全画幅与去边界子窗口混合更新。

### 5. ComfyUI `post_cfg_function` 的调用顺序
- **现象**：在 `comfy/samplers.py` 中，`cfg_function` 是在 `calc_cond_batch` 的 `try...finally` 结束之后才调用的。
- **原因**：`close_step()` 把 `state.current_step` 置回 `None` 后，如果 `post_cfg_hook` 只往 `current_step` 写数据，就会漏记 `cond_scale` 和 `denoised`。
- **处理**：在 `post_cfg_hook` 中优先写 `state.current_step`，若为 `None` 则写入刚归档的 `state.last_valid_step`。

### 6. 高压缩比 3D 视频头在纯像素 `MSE` 下的高频涂抹与均值假象（`MiniMax-H3 v0.2.0`）
- **现象**：`MiniMax-H3` 预览版总体平均 PSNR 达到了 `31.03 dB`（`45.7 MiB` vs 官方 `2.68 GB` ViT3D VAE，耗时 `29.6 ms` vs `251.3 ms`），但实际查看油画花瓣、人物眉眼及快速运动遮挡（`25.91 dB`，`LPIPS = 0.29～0.37`）时，细部有明显的平滑涂抹感。
- **原因**：
  1. **合成纯色样本拉高了总体均分**：校准集中占 `25%` 的纯色扁平文字片段跑出了 `36.73 dB`（`LPIPS = 0.0547`），掩盖了真实自然纹理子集只有 `25.9～30.1 dB` 的问题。评测视频头时必须分场景看 PSNR 和 LPIPS。
  2. **`16×` 解压倍率 + 高末步噪声（$\sigma_{\text{last}} = 0.6316$）+ 缺少感知损失**：`MiniMax-H3` 单个 DiT Token 对应 `4帧 × 32×32 = 4,096` 个像素（是 `16×16` 图像头的 16 倍），且 `video_shift = 12.0` 下 8 步末步起点噪声高达 `0.6316`。只用像素空间的 `L1 + 5.0 × MSE` 训练而不加 `LPIPS-VGG` 等感知约束时，小网络对不确定的高频细节会输出平滑平均值。
  3. **`16px` 潜变量提前降采样**：`x0_hat_vid`（`[B, 24, T_lat, H_lat, W_lat]`）本身就在 `16px` 网格上，入口处将其 `2×2` 折叠进 `32px` 层损失了一部分空间精度。
- **后续改进方向**：在 `16px`（`s16`）阶段直接注入未折叠的 `24ch` `x0_hat_vid`，在训练后半程加入 `LPIPS-VGG` 与 2D FFT 频域幅度损失，适当加宽主干通道（`width >= 960`），并去掉纯色合成块、改用更高分辨率的自然动态画面训练。


### 7. 在不引入全图归一化耗时瓶颈的前提下恢复 1～2px 微距锐度（`Qwen-Image 2.1 v0.1 → v0.2 → v0.3`）
- **现象**：首版 `v0.1`（`8×8` 线性展开，L4 `1024²` 耗时 `7.77 ms`）在车票细字与近景毛孔上偏软（`LPIPS = 0.1272`）。`v0.2` 补齐 `16px → 1px` 全分辨率 `GroupNorm + 逐通道卷积` 金字塔并加入 `LPIPS + 2D FFT` 监督后，`LPIPS` 降至 `0.0808`，但解码耗时涨至 `115.05 ms`（激活显存 `904 MiB`），且铜版交叉排线子集几乎未涨（`26.82 → 26.85 dB`）。
- **原因**：
  1. 在 `1024×1024` 全图上跑 `GroupNorm` 与逐通道分离卷积受限于显存带宽；
  2. `x0_hat_norm` 仅通过逐 Token `1×1` 映射注入，缺少跨相邻 Token 边界的斜线几何先验；
  3. 常规像素空间判别器容易改动低频底色，导致 GAN 微调掉 `0.5～1.5 dB` PSNR。
- **处理（`v0.3`）**：
  1. 神经网络止步于 `512×512`（`2px` 网格），用无归一化融合残差块预测 2D Haar 小波四子带 `(LL, LH, HL, HH)`，再经零参数解析逆变换 `idwt_haar_2d` 还原 `1024×1024`（L4 耗时降至 **`37.51 ms`**，比 `v0.2` 快 `3.07×`，激活显存降至 `504 MiB`）；
  2. 增加 `LatentSpatialHighway`，对 `64` 通道潜变量做 `3×3` 空间卷积并跨级注入 `8px`、`4px`、`2px` 层；
  3. 判别器输入改为 `torch.cat([LL.detach(), LH, HL, HH], dim=1)`，锁死低频 `LL` 梯度只推高频边缘，最终留出集总体升至 **`33.81 dB` PSNR**（人像 `39.17 dB`、版画比 `v0.2` 涨 `+0.56 dB`）、**`0.0736` LPIPS**。
