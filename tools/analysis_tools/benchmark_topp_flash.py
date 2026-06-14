"""性能测试脚本：对比不同 kernel 版本的性能。

用法:
    python tools/analysis_tools/benchmark_topp_flash.py
"""

import time
import torch
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from mmseg.models.utils.topp_flash_kernel import (
    topp_attention_reference,
    topp_flash_attention,
    _load_cuda_extension,
    is_topp_flash_available,
)


def make_benchmark_inputs(n=2, n_win=4, h=32, w=32, qk_dim=64, dim=128,
                          num_heads=8, topk=8, kv_len=49, dtype=torch.float32, device='cuda'):
    """创建 benchmark 用的输入。"""
    p2 = n_win * n_win
    q_h = h // n_win
    q_w = w // n_win
    q_len = q_h * q_w

    q_pix = torch.randn(n, p2, q_len, qk_dim, dtype=dtype, device=device)
    kv_pix = torch.randn(n, p2, kv_len, qk_dim + dim, dtype=dtype, device=device)
    r_weight = torch.rand(n, p2, topk, dtype=dtype, device=device)
    r_idx = torch.randint(0, p2, (n, p2, topk), device=device)
    keep_len = torch.randint(1, topk + 1, (n, p2), device=device)
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
        'num_heads': num_heads,
        'qk_dim': qk_dim,
        'dim': dim,
        'scale': qk_dim ** -0.5,
        'n_win': n_win,
        'H': h,
        'W': w,
    }


def benchmark_fn(fn, inputs, warmup=5, repeats=20):
    """测量函数执行时间。"""
    # Warmup
    for _ in range(warmup):
        fn(**inputs)
    torch.cuda.synchronize()

    start = time.perf_counter()
    for _ in range(repeats):
        fn(**inputs)
    torch.cuda.synchronize()
    elapsed = (time.perf_counter() - start) / repeats

    return elapsed * 1000  # 返回毫秒


def main():
    if not torch.cuda.is_available():
        print("CUDA 不可用，跳过性能测试")
        return

    print("=" * 70)
    print("TOPP Flash Attention 性能测试")
    print("=" * 70)

    # 测试配置
    configs = [
        {"name": "小规模", "n": 2, "n_win": 2, "h": 16, "w": 16, "qk_dim": 32, "dim": 64, "num_heads": 4, "topk": 4, "kv_len": 16},
        {"name": "中规模", "n": 2, "n_win": 4, "h": 32, "w": 32, "qk_dim": 64, "dim": 128, "num_heads": 8, "topk": 8, "kv_len": 49},
        {"name": "大规模", "n": 2, "n_win": 4, "h": 64, "w": 64, "qk_dim": 128, "dim": 256, "num_heads": 16, "topk": 8, "kv_len": 49},
    ]

    for config in configs:
        print(f"\n{'=' * 70}")
        print(f"配置: {config['name']}")
        print(f"  n={config['n']}, n_win={config['n_win']}, H=W={config['h']}")
        print(f"  qk_dim={config['qk_dim']}, dim={config['dim']}, heads={config['num_heads']}")
        print(f"  topk={config['topk']}, kv_len={config['kv_len']}")
        print("=" * 70)

        for dtype_name, dtype in [("float32", torch.float32), ("float16", torch.float16), ("bfloat16", torch.bfloat16)]:
            print(f"\n--- {dtype_name} ---")
            try:
                inputs = make_benchmark_inputs(
                    n=config['n'], n_win=config['n_win'], h=config['h'], w=config['w'],
                    qk_dim=config['qk_dim'], dim=config['dim'], num_heads=config['num_heads'],
                    topk=config['topk'], kv_len=config['kv_len'], dtype=dtype, device='cuda'
                )

                # 测试 reference（PyTorch 实现）
                with torch.no_grad():
                    time_ref = benchmark_fn(topp_attention_reference, inputs)
                    print(f"  Reference (PyTorch): {time_ref:.2f} ms")

                # 测试 CUDA kernel（如果可用）
                if is_topp_flash_available('cuda'):
                    with torch.no_grad():
                        time_cuda = benchmark_fn(
                            lambda **kw: topp_flash_attention(**kw, backend='cuda'),
                            inputs
                        )
                        speedup = time_ref / time_cuda
                        print(f"  CUDA Kernel: {time_cuda:.2f} ms")
                        print(f"  加速比: {speedup:.2f}x")

                # 测试 torch_block 后端
                with torch.no_grad():
                    time_block = benchmark_fn(
                        lambda **kw: topp_flash_attention(**kw, backend='torch_block'),
                        inputs
                    )
                    speedup_block = time_ref / time_block
                    print(f"  Torch Block: {time_block:.2f} ms")
                    print(f"  加速比 vs Reference: {speedup_block:.2f}x")

            except Exception as e:
                print(f"  测试失败: {e}")

    print("\n" + "=" * 70)
    print("测试完成")
    print("=" * 70)


if __name__ == "__main__":
    main()
