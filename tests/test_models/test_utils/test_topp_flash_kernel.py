import importlib.util
import shutil
from pathlib import Path

import pytest
import torch


def _load_kernel_module():
    root = Path(__file__).resolve().parents[3]
    path = root / 'mmseg' / 'models' / 'utils' / 'topp_flash_kernel.py'
    spec = importlib.util.spec_from_file_location('topp_flash_kernel', path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _make_inputs(dtype=torch.float64, device='cpu'):
    torch.manual_seed(7)
    n = 2
    n_win = 2
    p2 = n_win * n_win
    h = w = 4
    q_len = 4
    kv_len = 3
    qk_dim = 4
    dim = 6
    topk = 3
    q_pix = torch.randn(n, p2, q_len, qk_dim, dtype=dtype, device=device)
    kv_pix = torch.randn(n, p2, kv_len, qk_dim + dim, dtype=dtype, device=device)
    r_weight = torch.rand(n, p2, topk, dtype=dtype, device=device)
    r_idx = torch.randint(0, p2, (n, p2, topk), device=device)
    keep_len = torch.tensor(
        [[1, 2, topk, 1], [topk, 1, 2, topk]], device=device)
    pos = torch.arange(topk, device=device).view(1, 1, topk)
    r_mask = pos < keep_len[..., None]
    r_weight = r_weight * r_mask.to(dtype)
    r_idx = r_idx.masked_fill(~r_mask, 0)
    return {
        'q_pix': q_pix,
        'kv_pix': kv_pix,
        'r_weight': r_weight,
        'r_idx': r_idx,
        'r_mask': r_mask,
        'num_heads': 2,
        'qk_dim': qk_dim,
        'dim': dim,
        'scale': qk_dim**-0.5,
        'n_win': n_win,
        'H': h,
        'W': w,
    }


def test_topp_flash_forward_matches_reference():
    kernel = _load_kernel_module()
    inputs = _make_inputs()
    out_ref = kernel.topp_attention_reference(**inputs)
    out_flash = kernel.topp_flash_attention(**inputs, block_windows=3)
    torch.testing.assert_close(out_flash, out_ref, rtol=1e-10, atol=1e-10)


def test_topp_flash_cuda_forward_matches_reference_if_available(monkeypatch):
    if not torch.cuda.is_available():
        pytest.skip('CUDA is unavailable.')
    if shutil.which('ninja') is None:
        pytest.skip('Ninja is unavailable.')
    kernel = _load_kernel_module()
    if not kernel.is_topp_flash_available('cuda'):
        pytest.skip('PVSA CUDA extension build environment is unavailable.')
    monkeypatch.setenv('PVSA_TOPP_FLASH_STRICT_CUDA', '1')

    inputs = _make_inputs(dtype=torch.float32, device='cuda')
    with torch.no_grad():
        out_ref = kernel.topp_attention_reference(**inputs)
        out_flash = kernel.topp_flash_attention(
            **inputs, block_windows=3, backend='cuda')
    torch.testing.assert_close(out_flash, out_ref, rtol=1e-4, atol=1e-4)


def test_topp_flash_cuda_forward_fp16_if_available(monkeypatch):
    if not torch.cuda.is_available():
        pytest.skip('CUDA is unavailable.')
    if not torch.cuda.is_bf16_supported():
        pytest.skip('BF16 is not supported on this device.')
    if shutil.which('ninja') is None:
        pytest.skip('Ninja is unavailable.')
    kernel = _load_kernel_module()
    if not kernel.is_topp_flash_available('cuda'):
        pytest.skip('PVSA CUDA extension build environment is unavailable.')
    monkeypatch.setenv('PVSA_TOPP_FLASH_STRICT_CUDA', '1')

    inputs = _make_inputs(dtype=torch.float16, device='cuda')
    with torch.no_grad():
        out_ref = kernel.topp_attention_reference(**inputs)
        out_flash = kernel.topp_flash_attention(
            **inputs, block_windows=3, backend='cuda')
    # 半精度容差更大
    torch.testing.assert_close(out_flash, out_ref, rtol=1e-2, atol=1e-2)


def test_topp_flash_cuda_forward_bf16_if_available(monkeypatch):
    if not torch.cuda.is_available():
        pytest.skip('CUDA is unavailable.')
    if not torch.cuda.is_bf16_supported():
        pytest.skip('BF16 is not supported on this device.')
    if shutil.which('ninja') is None:
        pytest.skip('Ninja is unavailable.')
    kernel = _load_kernel_module()
    if not kernel.is_topp_flash_available('cuda'):
        pytest.skip('PVSA CUDA extension build environment is unavailable.')
    monkeypatch.setenv('PVSA_TOPP_FLASH_STRICT_CUDA', '1')

    inputs = _make_inputs(dtype=torch.bfloat16, device='cuda')
    with torch.no_grad():
        out_ref = kernel.topp_attention_reference(**inputs)
        out_flash = kernel.topp_flash_attention(
            **inputs, block_windows=3, backend='cuda')
    # 半精度容差更大
    torch.testing.assert_close(out_flash, out_ref, rtol=1e-2, atol=1e-2)


def test_topp_flash_backward_matches_reference():
    kernel = _load_kernel_module()
    inputs = _make_inputs()

    ref_inputs = dict(inputs)
    flash_inputs = dict(inputs)
    for name in ('q_pix', 'kv_pix', 'r_weight'):
        ref_inputs[name] = inputs[name].detach().clone().requires_grad_(True)
        flash_inputs[name] = inputs[name].detach().clone().requires_grad_(True)

    out_ref = kernel.topp_attention_reference(**ref_inputs)
    out_flash = kernel.topp_flash_attention(**flash_inputs, block_windows=3)
    loss_ref = out_ref.square().mean()
    loss_flash = out_flash.square().mean()
    loss_ref.backward()
    loss_flash.backward()

    for name in ('q_pix', 'kv_pix', 'r_weight'):
        torch.testing.assert_close(
            flash_inputs[name].grad,
            ref_inputs[name].grad,
            rtol=1e-9,
            atol=1e-9)
