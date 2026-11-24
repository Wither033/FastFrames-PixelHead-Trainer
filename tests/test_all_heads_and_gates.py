"""Automated CPU verification suite for `pixelhead_trainer` and all 5 ComfyUI backbone heads.

Run locally with zero GPU usage:
    CUDA_VISIBLE_DEVICES="" python tests/test_all_heads_and_gates.py
"""

from __future__ import annotations

import importlib.util
import sys
import tempfile
from pathlib import Path

import torch

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from pixelhead_trainer import (
    BridgedPixelHead2D,
    BridgedPixelHead2DX2,
    CausalVideoPixelHead3D,
    export_bf16_trunk_fp32_out,
    load_generic_pixel_head_2d,
    pixel_head_loss_2d,
    save_generic_pixel_head_2d,
    verify_chunked_causal_decode_3d,
    verify_spatial_crop_equivariance_2d,
    verify_spatial_crop_equivariance_3d,
    video_pixel_head_loss_3d,
)


def _load_mod(name: str, rel_path: str):
    p = REPO_ROOT / rel_path
    spec = importlib.util.spec_from_file_location(name, str(p))
    mod = importlib.util.module_from_spec(spec)
    assert spec and spec.loader
    spec.loader.exec_module(mod)
    return mod


def test_generic_2d_and_bf16_export() -> None:
    head = BridgedPixelHead2D(d_feat=256, d_lat=16, token_pixels=16, width=128, out_channels=4)
    diff2d = verify_spatial_crop_equivariance_2d(
        head, d_feat=256, d_lat=16, token_pixels=16, gh=12, gw=16, crop_h=8, crop_w=10, margin_tok=3
    )
    assert diff2d < 1e-5

    head_x2 = BridgedPixelHead2DX2(base=head, depth_x2=1)
    f = torch.randn(1, 4, 6, 256)
    x = torch.randn(1, 16, 4, 6)
    out1 = head_x2(f, x, scale=1)
    out2 = head_x2(f, x, scale=2)
    assert out1.shape == (1, 4, 64, 96)
    assert out2.shape == (1, 4, 128, 192)

    loss, psnr = pixel_head_loss_2d(out1, out1 * 0.95)
    assert torch.isfinite(loss) and psnr > 20.0

    with tempfile.TemporaryDirectory() as td:
        p_fp32 = Path(td) / "head_fp32.safetensors"
        p_bf16 = Path(td) / "head_bf16.safetensors"
        save_generic_pixel_head_2d(head_x2, p_fp32)
        export_bf16_trunk_fp32_out(p_fp32, p_bf16)
        assert p_bf16.stat().st_size < p_fp32.stat().st_size * 0.65
        reloaded = load_generic_pixel_head_2d(p_bf16, dtype=torch.bfloat16)
        assert reloaded.out2_proj.weight.dtype == torch.float32
        assert reloaded.base.out_proj.weight.dtype == torch.float32
    print(f"[PASS] Generic 2D + 2x Super-Res + BF16/FP32 safetensors (crop diff={diff2d:.2e})")


def test_generic_3d_video() -> None:
    vhead = CausalVideoPixelHead3D(d_feat=128, d_lat=32, width=128)
    diff3d = verify_spatial_crop_equivariance_3d(
        vhead, d_feat=128, d_lat=32, token_pixels=32, t_lat=3, gh=12, gw=15, crop_h=10, crop_w=11, margin_tok=4
    )
    cdiff = verify_chunked_causal_decode_3d(vhead, d_feat=128, d_lat=32, t_lat=9, gh=3, gw=4, chunk_t=3)
    vid = vhead(torch.randn(1, 3, 3, 4, 128), torch.randn(1, 32, 3, 3, 4))
    assert vid.shape == (1, 3, 17, 96, 128)
    vloss, vpsnr = video_pixel_head_loss_3d(vid, vid * 0.9)
    assert torch.isfinite(vloss) and vpsnr > 15.0
    print(f"[PASS] Generic 3D Causal Video Head (crop diff={diff3d:.2e}, chunk diff={cdiff:.2e})")


def test_all_five_comfyui_model_heads() -> None:
    qwen = _load_mod("qwen21_ph", "comfyui_nodes/ComfyUI-QwenImage21-PixelHead/qwen21_pixel_head.py")
    hq = qwen.QwenImage21PixelHead(width=128)
    assert verify_spatial_crop_equivariance_2d(hq, 4096, 64, 16, 11, 14, 8, 10, 3) < 1e-5

    # Verify QwenImage21PixelHeadV3 Haar DWT invertibility, v2->v3 warm-start, tiling, and safetensors round-trip
    img_test = torch.randn(1, 4, 64, 96)
    haar_err = float((qwen.idwt_haar_2d(*qwen.dwt_haar_2d(img_test)) - img_test).abs().max().item())
    assert haar_err < 1e-6, f"Haar DWT invertibility error too high: {haar_err}"
    hq_v2 = qwen.QwenImage21PixelHeadV2(base=hq, c_4px=64, c_2px=32, c_1px=16).eval()
    hq_v3 = qwen.QwenImage21PixelHeadV3.from_v2(hq_v2, c_2px=32).eval()
    with torch.no_grad():
        f_q3 = torch.randn(1, 6, 8, 4096)
        x_q3 = torch.randn(1, 64, 6, 8)
        out_q3 = hq_v3(f_q3, x_q3, sigma=0.18)
        assert out_q3.shape == (1, 4, 96, 128) and torch.isfinite(out_q3).all()
        tiled_q3 = qwen.decode_pixel_head_tiled(hq_v3, f_q3, x_q3, sigma=0.18, tile_tokens=4, overlap_tokens=2)
        assert tiled_q3.shape == out_q3.shape and torch.isfinite(tiled_q3).all()
    with tempfile.TemporaryDirectory() as td:
        p3_fp32 = Path(td) / "qwen_v3.safetensors"
        p3_bf16 = Path(td) / "qwen_v3_bf16.safetensors"
        qwen.save_qwen21_pixel_head(hq_v3, p3_fp32, save_bf16_trunk=False)
        qwen.save_qwen21_pixel_head(hq_v3, p3_bf16, save_bf16_trunk=True)
        hq_v3_loaded = qwen.load_qwen21_pixel_head(p3_bf16, dtype=torch.bfloat16)
        assert isinstance(hq_v3_loaded, qwen.QwenImage21PixelHeadV3)
        assert hq_v3_loaded.wavelet_out.weight.dtype == torch.float32

    krea = _load_mod("krea2_ph", "comfyui_nodes/ComfyUI-Krea2-PixelHead/krea2_pixel_head.py")
    hk = krea.Krea2PixelHead(width=128)
    assert verify_spatial_crop_equivariance_2d(hk, 6144, 64, 16, 11, 14, 8, 10, 3) < 1e-5

    flux = _load_mod("flux3_ph", "comfyui_nodes/ComfyUI-Flux3-PixelHead/flux3_pixel_head.py")
    hf = flux.Flux3PixelHead(width=128)
    assert verify_spatial_crop_equivariance_2d(hf, 3072, 128, 16, 11, 14, 8, 10, 3) < 1e-5

    ltx = _load_mod("ltx25_ph", "comfyui_nodes/ComfyUI-LTX25-PixelHead/ltx25_pixel_head.py")
    hl = ltx.LTX25VideoPixelHead(width=128)
    assert verify_spatial_crop_equivariance_3d(hl, 4096, 128, 32, 3, 12, 15, 10, 11, 4) < 1e-5
    assert verify_chunked_causal_decode_3d(hl, 4096, 128, t_lat=9, gh=3, gw=4, chunk_t=3) < 1e-5

    mm = _load_mod("mmh3_ph", "comfyui_nodes/ComfyUI-MiniMaxH3-PixelHead/minimax_h3_pixel_head.py")
    hm = mm.MiniMaxH3VideoPixelHead(width=128).eval()
    assert verify_spatial_crop_equivariance_3d(hm, 5376, 96, 32, 6, 12, 15, 10, 11, 4) < 1e-5
    with torch.no_grad():
        f_h3 = torch.randn(1, 11, 3, 4, 5376)
        x_h3_raw24 = torch.randn(1, 24, 11, 6, 8)
        diff_h3_chunk = float((hm(f_h3, x_h3_raw24) - hm.decode_chunked(f_h3, x_h3_raw24, chunk_t=5)).abs().max().item())
        assert diff_h3_chunk < 1e-5, f"MiniMax-H3 chunk diff too high: {diff_h3_chunk}"
    print(f"[PASS] All 5 ComfyUI Backbone Heads (Qwen-Image 2.1, Krea-2, FLUX.3-Klein, LTX-2.5 v2, MiniMax-H3 v2 chunk diff={diff_h3_chunk:.2e}) verified!")


if __name__ == "__main__":
    test_generic_2d_and_bf16_export()
    test_generic_3d_video()
    test_all_five_comfyui_model_heads()
    print("ALL TESTS PASSED CLEANLY!")
