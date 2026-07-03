"""Run framework ablations and summarize best mIoU.

照 run_backbone_block_ablation.py 的模式，对比三类语义分割框架：
    B0  纯视觉 PVSA-Net（无 CLIP）        base=pvsa_net_baseline.py
    B1  CLIP + BiFormer attention        base=attn_waterseg.py
    B2  CLIP + ToppAttention             base=waterseg.py

每个变体有独立 base config，关键差异已体现在各自 base config 里，因此本脚本
不接 config 位置参数、也不覆盖任何 cfg-options：仅由变体本身决定 base config。
用法（从仓库根目录运行）：
    CUDA_VISIBLE_DEVICES=0 python ablation/run_framework_ablation.py \
        --work-dir-root ablation/clip-framework \
        --shape 256 256 \
        --skip-existing
"""

import argparse
import csv
import json
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace

from mmengine.logging import MMLogger

from tools.analysis_tools.get_flops import inference as get_flops_inference


# 变体名列表（与 NORMALIZE_VARIANTS 同款）；具体 base config 在 get_variant 内查表。
# B0=纯视觉、B1=CLIP+BiFormer、B2=CLIP+ToppAttention。
FRAMEWORK_VARIANTS = [
    'framework-B0-pvsa-vision',  # 纯视觉 PVSA-Net（EncoderDecoder + SegformerHead）
    'framework-B1-clip-brg',    # CLIP + BiFormer 标准双层路由
    'framework-B2-clip-topp',   # CLIP + ToppAttention（top-p 投票路由）
]


def parse_args():
    parser = argparse.ArgumentParser(
        description='Run framework ablations and summarize best mIoU.')
    parser.add_argument(
        '--work-dir-root',
        default='ablation/clip-framework',
        help='Root directory for ablation runs.')
    parser.add_argument(
        '--python',
        default=sys.executable,
        help='Python executable used to launch training.')
    parser.add_argument(
        '--variants',
        nargs='+',
        choices=FRAMEWORK_VARIANTS,
        default=FRAMEWORK_VARIANTS,
        help='Variant names to run (subset of the three B0-B2 groups).')
    parser.add_argument(
        '--extra-cfg-options',
        nargs='*',
        default=[],
        help='Additional cfg-options passed through to tools/train.py.')
    parser.add_argument(
        '--skip-existing',
        action='store_true',
        help='Skip runs that already have a best mIoU result.')
    parser.add_argument(
        '--dry-run',
        action='store_true',
        help='Only print commands without launching training.')
    parser.add_argument(
        '--summary-only',
        action='store_true',
        help='Do not launch training, only refresh summary.csv from existing '
             'work_dirs.')
    parser.add_argument(
        '--shape',
        type=int,
        nargs='+',
        default=[256, 256],
        help='Input shape used for FLOPs/Params summary.')
    return parser.parse_args()


def get_variant(variant_name):
    """按变体名查表返回 base_config 相对路径。

    对照矩阵：
        B0  纯视觉 PVSA-Net   base=vision/pvsa_net_baseline.py
        B1  CLIP + BiFormer   base=clip/attn_waterseg.py
        B2  CLIP + ToppAttention  base=clip/waterseg.py
    """
    if variant_name == 'framework-B0-pvsa-vision':
        return 'configs-h/vision/pvsa_net_baseline.py'
    if variant_name == 'framework-B1-clip-brg':
        return 'configs-h/clip/attn_waterseg.py'
    if variant_name == 'framework-B2-clip-topp':
        return 'configs-h/clip/waterseg.py'
    raise ValueError(f'Unknown variant: {variant_name}. '
                     f'Valid: {FRAMEWORK_VARIANTS}')


def parse_best_miou(work_dir: Path):
    if not work_dir.exists():
        return None

    candidates = []
    candidates.extend(sorted(work_dir.rglob('scalars.json')))
    candidates.extend(sorted(
        json_file for json_file in work_dir.rglob('*.json')
        if json_file.name != 'summary.csv' and json_file.name != 'scalars.json'))

    best = None
    for json_file in candidates:
        value = parse_best_miou_from_json(json_file)
        if value is not None:
            best = value if best is None else max(best, value)
    return best


def parse_best_miou_from_json(json_file: Path):
    best = None
    try:
        with json_file.open('r', encoding='utf-8') as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    record = json.loads(line)
                except json.JSONDecodeError:
                    continue
                for key, value in record.items():
                    if 'mIoU' not in key or not isinstance(value, (int, float)):
                        continue
                    value = float(value)
                    best = value if best is None else max(best, value)
    except OSError:
        return None
    return best


def compute_model_complexity(config_path: Path, shape):
    """FLOPs/Params 不需要任何 cfg-options 覆盖，base config 直接算。"""
    logger = MMLogger.get_instance(name='framework_ablation_complexity')
    args = SimpleNamespace(
        config=str(config_path),
        shape=list(shape),
        cfg_options={},
    )
    result = get_flops_inference(args, logger)
    return result['flops'], result['params']


def main():
    args = parse_args()
    # 脚本位于 ablation/ 下，仓库根 = parents[1]；base_config 路径相对仓库根。
    repo_root = Path(__file__).resolve().parents[1]
    work_dir_root = Path(args.work_dir_root).resolve()
    work_dir_root.mkdir(parents=True, exist_ok=True)

    summary_rows = []

    for variant_name in args.variants:
        base_rel = get_variant(variant_name)
        config_path = (repo_root / base_rel).resolve()
        work_dir = work_dir_root / variant_name
        best_existing = parse_best_miou(work_dir)
        flops, params = compute_model_complexity(config_path, args.shape)

        if args.skip_existing and best_existing is not None:
            summary_rows.append({
                'run_name': variant_name,
                'base_config': base_rel,
                'flops': flops,
                'params': params,
                'best_mIoU': f'{best_existing:.6f}',
                'status': 'skipped_existing',
            })
            continue

        if args.summary_only:
            summary_rows.append({
                'run_name': variant_name,
                'base_config': base_rel,
                'flops': flops,
                'params': params,
                'best_mIoU': '' if best_existing is None
                else f'{best_existing:.6f}',
                'status': 'summary_only' if best_existing is not None
                else 'missing',
            })
            continue

        # 本消融不覆盖任何 cfg-options：base config 已包含全部差异。
        command = [
            args.python,
            'tools/train.py',
            str(config_path),
            '--work-dir',
            str(work_dir),
            *args.extra_cfg_options,
        ]

        print(' '.join(command))
        if args.dry_run:
            summary_rows.append({
                'run_name': variant_name,
                'base_config': base_rel,
                'flops': flops,
                'params': params,
                'best_mIoU': '',
                'status': 'dry_run',
            })
            continue

        result = subprocess.run(command, check=False)
        best_miou = parse_best_miou(work_dir)
        summary_rows.append({
            'run_name': variant_name,
            'base_config': base_rel,
            'flops': flops,
            'params': params,
            'best_mIoU': '' if best_miou is None
            else f'{best_miou:.6f}',
            'status':
            'ok' if result.returncode == 0 else f'failed({result.returncode})',
        })
        write_summary(work_dir_root / 'summary.csv', summary_rows)

    write_summary(work_dir_root / 'summary.csv', summary_rows)


def write_summary(path: Path, rows):
    with path.open('w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                'run_name', 'base_config', 'flops', 'params', 'best_mIoU',
                'status'
            ])
        writer.writeheader()
        writer.writerows(rows)


if __name__ == '__main__':
    main()
