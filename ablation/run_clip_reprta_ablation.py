
import argparse
import csv
import json
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace

from mmengine.logging import MMLogger

from tools.analysis_tools.get_flops import inference as get_flops_inference


# 变体名列表（与 CNN_BLOCK_TYPES 同款）；具体 cfg-options 在 get_variant 内查表。
# 三个开关含义：
#   use_reprta      : True(默认启用) | False(训期关闭，退化为纯注意力池化原型)
#   reprta_ffn_type : 'swiglu'(默认门控) | 'gelu'(普通 FFN，参数量与 SwiGLU 对齐)
#   reprta_zero_init: True(默认 w3 零初始化，保护 CLIP 原型) | False(随机初始化)
REPRTA_VARIANTS = [
    'reprta-R0-no',        # use_reprta=False       — 无适配器，纯注意力池化原型
    'reprta-R1-gelu',      # reprta_ffn_type=gelu   — 普通 FFN 残差适配器（对照门控贡献）
    'reprta-R2-default',   # 无覆盖                  — 当前默认 RepRTA（SwiGLU + 零初始化，主结果）
    'reprta-R3-no-zero',   # reprta_zero_init=False — SwiGLU 但不零初始化（对照保护 CLIP 原型）
]


def parse_args():
    parser = argparse.ArgumentParser(
        description='Run CLIP RepRTA ablations and summarize best mIoU.')
    parser.add_argument('config', help='Base config file path.')
    parser.add_argument(
        '--work-dir-root',
        default='ablation/clip-RepRTA',
        help='Root directory for ablation runs.')
    parser.add_argument(
        '--python',
        default=sys.executable,
        help='Python executable used to launch training.')
    parser.add_argument(
        '--variants',
        nargs='+',
        choices=REPRTA_VARIANTS,
        default=REPRTA_VARIANTS,
        help='Variant names to run (subset of the four R0-R3 groups).')
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
    """按变体名查表返回该组的 cfg-options dict。

    对照矩阵：
        R0  无 RepRTA              use_reprta=False
        R1  GELU 普通 FFN          reprta_ffn_type=gelu
        R2  当前默认 RepRTA        (无覆盖)
        R3  SwiGLU 但不零初始化    reprta_zero_init=False
    """
    if variant_name == 'reprta-R0-no':
        return {'model.text_encoder.use_reprta': False}
    if variant_name == 'reprta-R1-gelu':
        return {'model.text_encoder.reprta_ffn_type': 'gelu'}
    if variant_name == 'reprta-R2-default':
        return {}
    if variant_name == 'reprta-R3-no-zero':
        return {'model.text_encoder.reprta_zero_init': False}
    raise ValueError(f'Unknown variant: {variant_name}. '
                     f'Valid: {REPRTA_VARIANTS}')


def build_cfg_options(opts_dict):
    """返回 dict 与 list 两种形式：dict 给 get_flops，list 给 --cfg-options。"""
    cfg_list = []
    for key, value in opts_dict.items():
        if isinstance(value, bool):
            value = 'True' if value else 'False'
        cfg_list.append(f'{key}={value}')
    return opts_dict, cfg_list


def cfg_options_to_str(opts_dict):
    """summary.csv 用的人读字符串，如 'use_reprta=False'；空字典作 'default'。"""
    if not opts_dict:
        return 'default'
    parts = []
    for key, value in opts_dict.items():
        if isinstance(value, bool):
            value = 'True' if value else 'False'
        parts.append(f'{key.split(".")[-1]}={value}')
    return ','.join(parts)


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


def compute_model_complexity(config_path: Path, cfg_options: dict, shape):
    logger = MMLogger.get_instance(name='clip_reprta_ablation_complexity')
    args = SimpleNamespace(
        config=str(config_path),
        shape=list(shape),
        cfg_options=cfg_options,
    )
    result = get_flops_inference(args, logger)
    return result['flops'], result['params']


def main():
    args = parse_args()
    config_path = Path(args.config).resolve()
    work_dir_root = Path(args.work_dir_root).resolve()
    work_dir_root.mkdir(parents=True, exist_ok=True)

    summary_rows = []

    for variant_name in args.variants:
        cfg_options = get_variant(variant_name)
        work_dir = work_dir_root / variant_name
        best_existing = parse_best_miou(work_dir)
        cfg_dict, cfg_list = build_cfg_options(cfg_options)
        flops, params = compute_model_complexity(
            config_path, cfg_dict, args.shape)

        cfg_opts_str = cfg_options_to_str(cfg_options)

        if args.skip_existing and best_existing is not None:
            summary_rows.append({
                'run_name': variant_name,
                'cfg_options': cfg_opts_str,
                'flops': flops,
                'params': params,
                'best_mIoU': f'{best_existing:.6f}',
                'status': 'skipped_existing',
            })
            continue

        if args.summary_only:
            summary_rows.append({
                'run_name': variant_name,
                'cfg_options': cfg_opts_str,
                'flops': flops,
                'params': params,
                'best_mIoU': '' if best_existing is None
                else f'{best_existing:.6f}',
                'status': 'summary_only' if best_existing is not None
                else 'missing',
            })
            continue

        command = [
            args.python,
            'tools/train.py',
            str(config_path),
            '--work-dir',
            str(work_dir),
            '--cfg-options',
            *cfg_list,
            *args.extra_cfg_options,
        ]

        print(' '.join(command))
        if args.dry_run:
            summary_rows.append({
                'run_name': variant_name,
                'cfg_options': cfg_opts_str,
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
            'cfg_options': cfg_opts_str,
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
                'run_name', 'cfg_options', 'flops', 'params',
                'best_mIoU', 'status'
            ])
        writer.writeheader()
        writer.writerows(rows)


if __name__ == '__main__':
    main()
