"""Run CLIP head ablations and summarize best mIoU.

前四组固定使用 BRG + CLIP 入口 configs-h/clip/attn_waterseg.py，消融图相关
query 的两个二值参数；第五组使用纯视觉 BRG 入口，不引入任何文本：
    image_query_source:    backbone_pool / decode_fusion
    image_query_head_type: joint / separate

五组变体：
    Q0  backbone_pool + joint
    Q1  backbone_pool + separate
    Q2  decode_fusion + joint
    Q3  decode_fusion + separate
    Q4  纯视觉 BiFormer，无文本、无图相关 query

用法（从仓库根目录运行）：
    CUDA_VISIBLE_DEVICES=0 python ablation/clip_head.py \
        --work-dir-root ablation/clip_head \
        --shape 256 256 \
        --skip-existing

只打印命令：
    python ablation/clip_head.py --dry-run
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


CLIP_BRG_CONFIG = 'configs-h/clip/attn_waterseg.py'
VISION_BRG_CONFIG = 'configs-h/vision/attn_ablation_waterseg.py'

QUERY_VARIANTS = [
    'brg-query-Q0-backbone-joint',      # 骨干多阶段池化 + 单个联合输出头
    'brg-query-Q1-backbone-separate',   # 骨干多阶段池化 + 每类独立输出头
    'brg-query-Q2-decode-joint',        # 解码融合特征池化 + 单个联合输出头
    'brg-query-Q3-decode-separate',     # 解码融合特征池化 + 每类独立输出头
    'brg-query-Q4-no-text',             # 纯视觉 BiFormer，不引入任何文本
]


def parse_args():
    parser = argparse.ArgumentParser(
        description='Run CLIP head image-query ablations.')
    parser.add_argument(
        '--work-dir-root',
        default='ablation/clip_head',
        help='Root directory for ablation runs.')
    parser.add_argument(
        '--python',
        default=sys.executable,
        help='Python executable used to launch training.')
    parser.add_argument(
        '--variants',
        nargs='+',
        choices=QUERY_VARIANTS,
        default=QUERY_VARIANTS,
        help='Variant names to run.')
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
        help='Do not launch training, only refresh summary.csv.')
    parser.add_argument(
        '--shape',
        type=int,
        nargs='+',
        default=[256, 256],
        help='Input shape used for FLOPs/Params summary.')
    return parser.parse_args()


def get_variant(variant_name):
    """Return (base_config, image_query_source, image_query_head_type)."""
    if variant_name == 'brg-query-Q0-backbone-joint':
        return CLIP_BRG_CONFIG, 'backbone_pool', 'joint'
    if variant_name == 'brg-query-Q1-backbone-separate':
        return CLIP_BRG_CONFIG, 'backbone_pool', 'separate'
    if variant_name == 'brg-query-Q2-decode-joint':
        return CLIP_BRG_CONFIG, 'decode_fusion', 'joint'
    if variant_name == 'brg-query-Q3-decode-separate':
        return CLIP_BRG_CONFIG, 'decode_fusion', 'separate'
    if variant_name == 'brg-query-Q4-no-text':
        return VISION_BRG_CONFIG, 'none', 'none'
    raise ValueError(f'Unknown variant: {variant_name}. '
                     f'Valid: {QUERY_VARIANTS}')


def build_cfg_options(image_query_source, image_query_head_type):
    if image_query_source == 'none':
        return {}, []
    cfg_dict = {
        'model.image_query_proj.source': image_query_source,
        'model.image_query_proj.query_head_type': image_query_head_type,
    }
    cfg_list = [
        f'model.image_query_proj.source={image_query_source}',
        f'model.image_query_proj.query_head_type={image_query_head_type}',
    ]
    return cfg_dict, cfg_list


def parse_best_miou(work_dir: Path):
    if not work_dir.exists():
        return None

    candidates = []
    candidates.extend(sorted(work_dir.rglob('scalars.json')))
    candidates.extend(sorted(
        json_file for json_file in work_dir.rglob('*.json')
        if json_file.name not in ('summary.csv', 'scalars.json')))

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
    logger = MMLogger.get_instance(name='clip_head_ablation_complexity')
    args = SimpleNamespace(
        config=str(config_path),
        shape=list(shape),
        cfg_options=cfg_options,
    )
    result = get_flops_inference(args, logger)
    return result['flops'], result['params']


def main():
    args = parse_args()
    repo_root = Path(__file__).resolve().parents[1]
    work_dir_root = Path(args.work_dir_root).resolve()
    work_dir_root.mkdir(parents=True, exist_ok=True)

    summary_rows = []

    for variant_name in args.variants:
        base_config, image_query_source, image_query_head_type = get_variant(
            variant_name)
        config_path = (repo_root / base_config).resolve()
        cfg_dict, cfg_list = build_cfg_options(
            image_query_source, image_query_head_type)
        if cfg_list:
            cfg_opts_str = (
                f'image_query_source={image_query_source};'
                f'image_query_head_type={image_query_head_type}')
        else:
            cfg_opts_str = 'no_text_biformer'

        work_dir = work_dir_root / variant_name
        best_existing = parse_best_miou(work_dir)
        flops, params = compute_model_complexity(
            config_path, cfg_dict, args.shape)

        if args.skip_existing and best_existing is not None:
            summary_rows.append({
                'run_name': variant_name,
                'base_config': base_config,
                'image_query_source': image_query_source,
                'image_query_head_type': image_query_head_type,
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
                'base_config': base_config,
                'image_query_source': image_query_source,
                'image_query_head_type': image_query_head_type,
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
        ]
        all_cfg_options = [*cfg_list, *args.extra_cfg_options]
        if all_cfg_options:
            command.extend(['--cfg-options', *all_cfg_options])

        print(' '.join(command))
        if args.dry_run:
            summary_rows.append({
                'run_name': variant_name,
                'base_config': base_config,
                'image_query_source': image_query_source,
                'image_query_head_type': image_query_head_type,
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
            'base_config': base_config,
            'image_query_source': image_query_source,
            'image_query_head_type': image_query_head_type,
            'cfg_options': cfg_opts_str,
            'flops': flops,
            'params': params,
            'best_mIoU': '' if best_miou is None else f'{best_miou:.6f}',
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
                'run_name', 'base_config', 'image_query_source',
                'image_query_head_type', 'cfg_options', 'flops', 'params',
                'best_mIoU', 'status'
            ])
        writer.writeheader()
        writer.writerows(rows)


if __name__ == '__main__':
    main()
