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

泛化测试（单独命令启动，不训练）：
    CUDA_VISIBLE_DEVICES=0 python ablation/clip_head.py \
        --generalization-test \
        --work-dir-root ablation/clip_head

只打印命令：
    python ablation/clip_head.py --dry-run
"""

import argparse
import csv
import json
import os
import re
import shutil
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace


CLIP_BRG_CONFIG = 'configs-h/clip/attn_waterseg.py'
VISION_BRG_CONFIG = 'configs-h/vision/attn_ablation_waterseg.py'

GENERALIZATION_TARGETS = [
    dict(
        name='gqy_val',
        display_name='gqy',
        dataset_candidates=['configs-h/_base_/datasets/gqy.py'],
        split='val',
        # KAKA 训练输出顺序为：背景/船/可航行水域。
        # gqy 原始类为 water/ground/object；类别顺序搜索显示最优对应为：
        # water -> background，object -> boat，ground -> free-space。
        metainfo=dict(
            classes=('water', 'object', 'ground'),
            palette=[[0, 0, 255], [255, 0, 0], [0, 255, 0]]),
        iou_map=dict(
            background='water',
            boat='object',
            free_space='ground')),
    dict(
        name='kaka_test',
        display_name='kaka',
        dataset_candidates=[
            'configs-h/_base_/datasets/KAKA.py',
            'configs-h/_base_/datasets/kaka.py',
        ],
        split='test',
        # KAKA 原始训练/测试顺序。
        metainfo=dict(
            classes=('background', 'boat', 'free-space'),
            palette=[[0, 0, 0], [128, 0, 128], [0, 0, 255]]),
        iou_map=dict(
            background='background',
            boat='boat',
            free_space='free-space')),
    dict(
        name='gba_val',
        display_name='gba',
        dataset_candidates=['configs-h/_base_/datasets/GBA.py'],
        split='val',
        # GBA 原始类为 object/water/ground，同样重排到 KAKA 语义顺序。
        metainfo=dict(
            classes=('ground', 'object', 'water'),
            palette=[[0, 255, 0], [255, 0, 0], [0, 0, 255]]),
        iou_map=dict(
            background='ground',
            boat='object',
            free_space='water')),
]

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
        '--generalization-test',
        action='store_true',
        help='Test trained variants on gqy val, kaka test, and GBA val.')
    parser.add_argument(
        '--extra-test-args',
        nargs='*',
        default=[],
        help='Additional arguments passed through to tools/test.py.')
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


def get_variant_display_name(variant_name):
    return variant_name.replace('brg-query-', '')


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


def resolve_dataset_config(repo_root: Path, candidates):
    for candidate in candidates:
        path = repo_root / candidate
        if path.exists():
            return candidate, path.resolve()
    return None, None


def find_checkpoint(work_dir: Path):
    if not work_dir.exists():
        return None

    preferred = []
    preferred.extend(sorted(work_dir.rglob('best*.pth')))
    preferred.extend(sorted(work_dir.rglob('latest.pth')))
    if preferred:
        return preferred[0]

    epoch_ckpts = sorted(
        work_dir.rglob('epoch_*.pth'),
        key=lambda p: p.stat().st_mtime,
        reverse=True)
    if epoch_ckpts:
        return epoch_ckpts[0]

    iter_ckpts = sorted(
        work_dir.rglob('iter_*.pth'),
        key=lambda p: p.stat().st_mtime,
        reverse=True)
    if iter_ckpts:
        return iter_ckpts[0]
    return None


def relpath_for_config(path: Path, start: Path):
    try:
        return os.path.relpath(Path(path).resolve(), start.resolve()).replace(
            os.sep, '/')
    except ValueError:
        return Path(path).resolve().as_posix()


def write_eval_config(eval_config_path: Path, repo_root: Path,
                      base_config: str, dataset_config: str,
                      image_query_source: str, image_query_head_type: str,
                      split: str, target_metainfo: dict):
    base_rel = relpath_for_config(repo_root / base_config,
                                  eval_config_path.parent)
    dataset_path = (repo_root / dataset_config).resolve().as_posix()

    model_lines = [
        'model = dict(',
        '    data_preprocessor=data_preprocessor,',
        '    test_cfg=dict(mode="whole"),',
    ]
    if image_query_source != 'none':
        model_lines.extend([
            '    image_query_proj=dict(',
            f'        source="{image_query_source}",',
            f'        query_head_type="{image_query_head_type}"),',
        ])
    model_lines.append(')')

    split_lines = []
    if split == 'val':
        split_lines.extend([
            'test_dataloader = val_dataloader',
        ])

    text = f'''# Auto-generated by ablation/clip_head.py.
_base_ = [
    '{base_rel}',
]

_dataset_config = '{dataset_path}'
with open(_dataset_config, 'r', encoding='utf-8') as _f:
    exec(compile(_f.read(), _dataset_config, 'exec'))
del _dataset_config, _f

data_preprocessor = dict(
    type='SegDataPreProcessor',
    mean=[123.675, 116.28, 103.53],
    std=[58.395, 57.12, 57.375],
    bgr_to_rgb=True,
    pad_val=0,
    seg_pad_val=255,
    size=crop_size)

target_metainfo = dict(
    classes={target_metainfo['classes']!r},
    palette={target_metainfo['palette']!r})

for _loader_name in ('train_dataloader', 'val_dataloader', 'test_dataloader'):
    if _loader_name in globals():
        globals()[_loader_name]['dataset']['metainfo'] = target_metainfo

val_evaluator = dict(
    type='IoUMetric',
    iou_metrics=['mIoU', 'mDice'],
    ignore_index=255,
    classwise=True)
test_evaluator = val_evaluator

{chr(10).join(model_lines)}
{chr(10).join(split_lines)}
'''
    eval_config_path.parent.mkdir(parents=True, exist_ok=True)
    eval_config_path.write_text(text, encoding='utf-8')


def parse_test_metrics(work_dir: Path):
    metrics = {}
    if not work_dir.exists():
        return metrics

    parse_json_metrics(work_dir, metrics)
    parse_log_metrics(work_dir, metrics)
    if not first_metric(metrics, 'mAcc'):
        macc = mean_metric(metrics, 'Acc.')
        if macc is not None:
            metrics['mAcc'] = macc
    return metrics


def parse_json_metrics(work_dir: Path, metrics: dict):
    json_files = sorted(
        p for p in work_dir.rglob('*.json')
        if p.name not in ('summary.csv', 'generalization_summary.csv'))
    for json_file in json_files:
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
                        if not isinstance(value, (int, float)):
                            continue
                        if any(name in key for name in (
                                'mIoU', 'mDice', 'mAcc', 'aAcc', 'IoU.',
                                'Acc.', 'Dice.')):
                            metrics[key] = float(value)
        except OSError:
            continue


def parse_log_metrics(work_dir: Path, metrics: dict):
    class_row = re.compile(
        r'^\|\s*([^|]+?)\s*\|\s*([0-9.]+)\s*\|\s*([0-9.]+)\s*\|\s*'
        r'([0-9.]+)\s*\|')
    scalar = re.compile(r'\b(aAcc|mIoU|mAcc|mDice):\s*([0-9.]+)')
    for log_file in sorted(work_dir.rglob('*.log')):
        try:
            with log_file.open('r', encoding='utf-8', errors='ignore') as f:
                for line in f:
                    row = class_row.match(line.strip())
                    if row and row.group(1) not in ('Class', '--------'):
                        class_name = row.group(1).strip().replace(' ', '_')
                        metrics[f'IoU.{class_name}'] = float(row.group(2))
                        metrics[f'Acc.{class_name}'] = float(row.group(3))
                        metrics[f'Dice.{class_name}'] = float(row.group(4))
                    for key, value in scalar.findall(line):
                        metrics[key] = float(value)
        except OSError:
            continue


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
    from mmengine.logging import MMLogger
    from tools.analysis_tools.get_flops import inference as get_flops_inference

    logger = MMLogger.get_instance(name='clip_head_ablation_complexity')
    args = SimpleNamespace(
        config=str(config_path),
        shape=list(shape),
        cfg_options=cfg_options,
    )
    result = get_flops_inference(args, logger)
    return result['flops'], result['params']


def run_generalization_tests(args):
    repo_root = Path(__file__).resolve().parents[1]
    work_dir_root = Path(args.work_dir_root).resolve()
    eval_config_root = work_dir_root / '_generalization_configs'
    eval_work_root = work_dir_root / 'generalization'
    eval_work_root.mkdir(parents=True, exist_ok=True)

    summary_rows = []

    for variant_name in args.variants:
        base_config, image_query_source, image_query_head_type = get_variant(
            variant_name)
        checkpoint = find_checkpoint(work_dir_root / variant_name)

        for target in GENERALIZATION_TARGETS:
            dataset_config, dataset_path = resolve_dataset_config(
                repo_root, target['dataset_candidates'])
            ablation_name = get_variant_display_name(variant_name)
            run_name = f'{target["display_name"]}__{ablation_name}'
            eval_work_dir = eval_work_root / run_name

            row = {
                'dataset': target['display_name'],
                'mIoU': '',
                'IoU_background': '',
                'IoU_boat': '',
                'IoU_free_space': '',
                'mACC': '',
                'ablation': ablation_name,
                'status': '',
            }

            if dataset_path is None:
                row['status'] = 'missing_dataset_config'
                summary_rows.append(row)
                continue
            if checkpoint is None:
                row['status'] = 'missing_checkpoint'
                summary_rows.append(row)
                continue

            eval_config = eval_config_root / f'{run_name}.py'
            write_eval_config(
                eval_config, repo_root, base_config, dataset_config,
                image_query_source, image_query_head_type, target['split'],
                target['metainfo'])

            command = [
                args.python,
                'tools/test.py',
                str(eval_config),
                str(checkpoint),
                '--work-dir',
                str(eval_work_dir),
                *args.extra_test_args,
            ]

            print(' '.join(command))
            if args.dry_run:
                row['status'] = 'dry_run'
                summary_rows.append(row)
                continue

            if eval_work_dir.exists():
                shutil.rmtree(eval_work_dir)
            result = subprocess.run(command, check=False)
            metrics = parse_test_metrics(eval_work_dir)
            row['mIoU'] = first_metric(metrics, 'mIoU')
            row['IoU_background'] = class_iou(
                metrics, target['iou_map']['background'])
            row['IoU_boat'] = class_iou(metrics, target['iou_map']['boat'])
            row['IoU_free_space'] = class_iou(
                metrics, target['iou_map']['free_space'])
            row['mACC'] = first_metric(metrics, 'mAcc')
            row['status'] = (
                'ok' if result.returncode == 0
                else f'failed({result.returncode})')
            summary_rows.append(row)
            write_generalization_summary(
                work_dir_root / 'generalization_summary.csv', summary_rows)

    write_generalization_summary(
        work_dir_root / 'generalization_summary.csv', summary_rows)


def first_metric(metrics, pattern):
    for key in sorted(metrics):
        if pattern in key:
            return f'{metrics[key]:.6f}'
    return ''


def mean_metric(metrics, pattern):
    values = [value for key, value in metrics.items() if pattern in key]
    if not values:
        return None
    return sum(values) / len(values)


def class_iou(metrics, class_name):
    pattern = f'IoU.{class_name}'.replace(' ', '_')
    for key in sorted(metrics):
        if key.endswith(pattern):
            return f'{metrics[key]:.6f}'
    return ''


def write_generalization_summary(path: Path, rows):
    with path.open('w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                'dataset', 'mIoU', 'IoU_background', 'IoU_boat',
                'IoU_free_space', 'mACC', 'ablation', 'status'
            ])
        writer.writeheader()
        writer.writerows(rows)


def main():
    args = parse_args()
    if args.generalization_test:
        run_generalization_tests(args)
        return

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
