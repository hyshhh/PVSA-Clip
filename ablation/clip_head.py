
import argparse
import csv
import json
import math
import os
import re
import shutil
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace

# 训练 KAKA：background / boat / free-space -> land / ship / water
# CUDA_VISIBLE_DEVICES=0 python ablation/clip_head.py --work-dir-root ablation/clip_head_fix_prompt --shape 256 256 --variants clip-v2-actprompt --train-dataset kaka
# CUDA_VISIBLE_DEVICES=0 python ablation/clip_head.py --work-dir-root ablation/clip_head_fix_prompt --shape 256 256 --variants brg-query-Q4-no-text clip-v1-best clip-v2-actprompt --train-dataset kaka
# 单独跑 headv2（不加 --train-dataset 时默认 kaka，--skip-existing 跳过已有 best mIoU 的运行）
# CUDA_VISIBLE_DEVICES=0 python ablation/clip_head.py --work-dir-root ablation/clip_head_fix_prompt --shape 256 256 --variants clip-v2-actprompt --skip-existing
#
# 训练 gqy：water / ground / object -> water / land / ship
# CUDA_VISIBLE_DEVICES=0 python ablation/clip_head.py --work-dir-root ablation/clip_head_fix_prompt --shape 256 256 --variants clip-v2-actprompt --train-dataset gqy
# CUDA_VISIBLE_DEVICES=0 python ablation/clip_head.py --work-dir-root ablation/clip_head_fix_prompt --shape 256 256 --variants brg-query-Q4-no-text clip-v1-best clip-v2-actprompt --train-dataset gqy
#
# 训练 GBA：object / water / ground -> ship / water / land
# CUDA_VISIBLE_DEVICES=0 python ablation/clip_head.py --work-dir-root ablation/clip_head_fix_prompt --shape 256 256 --variants clip-v2-actprompt --train-dataset gba
# CUDA_VISIBLE_DEVICES=0 python ablation/clip_head.py --work-dir-root ablation/clip_head_fix_prompt --shape 256 256 --variants brg-query-Q4-no-text clip-v1-best clip-v2-actprompt --train-dataset gba
CLIP_BRG_CONFIG = 'configs-h/clip/attn_waterseg.py'
VISION_BRG_CONFIG = 'configs-h/vision/attn_ablation_waterseg.py'
VISION_CLIP_BACKBONE_CONFIG = (
    'configs-h/vision/attn_clipbackbone_seghead_waterseg.py')

PROMPT_CATEGORY_ORDERS = dict(
    # KAKA: background / boat / free-space -> land / ship / water
    kaka=['land', 'ship', 'water'],
    # gqy: water / ground / object -> water / land / ship
    gqy=['water', 'land', 'ship'],
    # GBA: object / water / ground -> ship / water / land
    gba=['ship', 'water', 'land'])

TRAIN_DATASETS = dict(
    kaka=dict(
        dataset_candidates=[
            'configs-h/_base_/datasets/KAKA.py',
            'configs-h/_base_/datasets/kaka.py',
        ]),
    gqy=dict(
        dataset_candidates=['configs-h/_base_/datasets/gqy.py']),
    gba=dict(
        dataset_candidates=['configs-h/_base_/datasets/GBA.py']))

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
    'brg-query-Q5-same-backbone-no-text',  # 同 CLIP 骨干结构的无文本基线
    'clip-v1-best',                     # 旧版最佳：decode_fusion + separate
    'clip-v2-actprompt',                # 新版：类激活视觉提示 + 文本原型增量
]

DEFAULT_VARIANTS = [
    'brg-query-Q4-no-text',
    'clip-v1-best',
    'clip-v2-actprompt',
]

VARIANT_NOTES = {
    'brg-query-Q4-no-text':
    '旧无文本基线：EncoderDecoder + BiFormer_standalone + SegformerHead；'
    '不和 CLIP 分支严格同构。',
    'brg-query-Q5-same-backbone-no-text':
    '公平无文本基线：同 CLIP 分支的 BiFormer_fusion_clip 骨干 + SegformerHead。',
    'clip-v1-best':
    'CLIPSegHead v1：decode_fusion 图像查询 + 按类提示词池化。',
    'clip-v2-actprompt':
    'CLIPSegHead v2：普通视觉分支 + 类激活视觉提示 + 文本分支辅助监督。',
}


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
        default=DEFAULT_VARIANTS,
        help='Variant names to run.')
    parser.add_argument(
        '--extra-cfg-options',
        nargs='*',
        default=[],
        help='Additional cfg-options passed through to tools/train.py.')
    parser.add_argument(
        '--train-dataset',
        choices=sorted(TRAIN_DATASETS),
        default='kaka',
        help='Dataset config used for training.')
    parser.add_argument(
        '--prompt-dataset',
        choices=sorted(PROMPT_CATEGORY_ORDERS),
        default=None,
        help='Label order used to align prompt prototypes. Defaults to '
        '--train-dataset.')
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
    if variant_name == 'brg-query-Q5-same-backbone-no-text':
        return VISION_CLIP_BACKBONE_CONFIG, 'none', 'same_backbone_seghead'
    if variant_name == 'clip-v1-best':
        return CLIP_BRG_CONFIG, 'decode_fusion', 'separate'
    if variant_name == 'clip-v2-actprompt':
        return CLIP_BRG_CONFIG, 'class_activation', 'v2'
    raise ValueError(f'Unknown variant: {variant_name}. '
                     f'Valid: {QUERY_VARIANTS}')


def get_variant_display_name(variant_name):
    return variant_name.replace('brg-query-', '')


def get_work_dir(work_dir_root: Path, variant_name: str, train_dataset: str):
    return work_dir_root / f'{variant_name}__{train_dataset}'


def decode_head_type_for_variant(variant_name):
    if variant_name == 'clip-v2-actprompt':
        return 'CLIPSegHeadV2'
    if variant_name in (
            'brg-query-Q4-no-text',
            'brg-query-Q5-same-backbone-no-text'):
        return None
    return 'CLIPSegHead'


def build_cfg_options(image_query_source, image_query_head_type,
                      variant_name=None, prompt_dataset='kaka'):
    decode_head_type = decode_head_type_for_variant(variant_name)
    cfg_dict = {}
    cfg_list = []
    if decode_head_type is not None:
        cfg_dict['model.decode_head.type'] = decode_head_type
        cfg_list.append(f'model.decode_head.type={decode_head_type}')
        prompt_order = PROMPT_CATEGORY_ORDERS[prompt_dataset]
        cfg_dict['model.text_encoder.prompt_category_order'] = prompt_order
        cfg_list.append(
            f'model.text_encoder.prompt_category_order={prompt_order!r}')

    if image_query_source in ('none', 'class_activation'):
        return cfg_dict, cfg_list
    cfg_dict.update({
        'model.image_query_proj.source': image_query_source,
        'model.image_query_proj.query_head_type': image_query_head_type,
    })
    cfg_list.extend([
        f'model.image_query_proj.source={image_query_source}',
        f'model.image_query_proj.query_head_type={image_query_head_type}',
    ])
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


def make_train_config(repo_root: Path, work_dir_root: Path,
                      base_config: str, train_dataset: str):
    """Return a config path whose dataset matches --train-dataset."""
    base_path = (repo_root / base_config).resolve()
    dataset_config, dataset_path = resolve_dataset_config(
        repo_root, TRAIN_DATASETS[train_dataset]['dataset_candidates'])
    if dataset_path is None:
        candidates = TRAIN_DATASETS[train_dataset]['dataset_candidates']
        raise FileNotFoundError(
            f'Missing dataset config for {train_dataset}: {candidates}')
    if train_dataset == 'kaka':
        return base_path

    train_config_root = work_dir_root / '_train_configs'
    train_config_root.mkdir(parents=True, exist_ok=True)
    train_config_path = train_config_root / (
        f'{Path(base_config).stem}__{train_dataset}.py')
    base_rel = relpath_for_config(base_path, train_config_path.parent)
    dataset_abs = dataset_path.as_posix()

    text = f'''# Auto-generated by ablation/clip_head.py.
_base_ = [
    '{base_rel}',
]

_dataset_config = '{dataset_abs}'
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

val_evaluator = dict(
    type='IoUMetric',
    iou_metrics=['mIoU', 'mDice'],
    ignore_index=255,
    classwise=True)
test_evaluator = val_evaluator

model.update(
    data_preprocessor=data_preprocessor,
    test_cfg=dict(mode='whole'))
'''
    train_config_path.write_text(text, encoding='utf-8')
    return train_config_path


def write_eval_config(eval_config_path: Path, repo_root: Path,
                      base_config: str, dataset_config: str,
                      image_query_source: str, image_query_head_type: str,
                      split: str, target_metainfo: dict,
                      variant_name: str, prompt_dataset: str):
    base_rel = relpath_for_config(repo_root / base_config,
                                  eval_config_path.parent)
    dataset_path = (repo_root / dataset_config).resolve().as_posix()

    model_lines = [
        'model = dict(',
        '    data_preprocessor=data_preprocessor,',
        '    test_cfg=dict(mode="whole"),',
    ]
    decode_head_type = decode_head_type_for_variant(variant_name)
    if decode_head_type is not None:
        prompt_order = PROMPT_CATEGORY_ORDERS[prompt_dataset]
        model_lines.extend([
            '    decode_head=dict(',
            f'        type="{decode_head_type}"),',
            '    text_encoder=dict(',
            f'        prompt_category_order={prompt_order!r}),',
        ])
    if image_query_source not in ('none', 'class_activation'):
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


def read_previous_summary(path: Path):
    if not path.exists():
        return {}
    try:
        with path.open('r', newline='', encoding='utf-8') as f:
            return {
                row.get('run_name', ''): row
                for row in csv.DictReader(f)
                if row.get('run_name')
            }
    except OSError:
        return {}


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
        checkpoint = find_checkpoint(
            get_work_dir(work_dir_root, variant_name, args.train_dataset))

        for target in GENERALIZATION_TARGETS:
            dataset_config, dataset_path = resolve_dataset_config(
                repo_root, target['dataset_candidates'])
            ablation_name = get_variant_display_name(variant_name)
            run_name = (
                f'{target["display_name"]}__{ablation_name}'
                f'__trained-{args.train_dataset}')
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
                target['metainfo'], variant_name, args.prompt_dataset)

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
    if args.prompt_dataset is None:
        args.prompt_dataset = args.train_dataset
    if args.generalization_test:
        run_generalization_tests(args)
        return

    repo_root = Path(__file__).resolve().parents[1]
    work_dir_root = Path(args.work_dir_root).resolve()
    work_dir_root.mkdir(parents=True, exist_ok=True)
    previous_summary = read_previous_summary(work_dir_root / 'summary.csv')

    summary_rows = []

    for variant_name in args.variants:
        base_config, image_query_source, image_query_head_type = get_variant(
            variant_name)
        config_path = make_train_config(
            repo_root, work_dir_root, base_config, args.train_dataset)
        cfg_dict, cfg_list = build_cfg_options(
            image_query_source, image_query_head_type, variant_name,
            args.prompt_dataset)
        if cfg_list:
            prompt_order = PROMPT_CATEGORY_ORDERS[args.prompt_dataset]
            cfg_opts_str = (
                f'train_dataset={args.train_dataset};'
                f'image_query_source={image_query_source};'
                f'image_query_head_type={image_query_head_type};'
                f'prompt_dataset={args.prompt_dataset};'
                f'prompt_category_order={prompt_order}')
        else:
            cfg_opts_str = f'train_dataset={args.train_dataset};no_text_biformer'

        work_dir = get_work_dir(
            work_dir_root, variant_name, args.train_dataset)
        best_existing = parse_best_miou(work_dir)
        try:
            flops, params = compute_model_complexity(
                config_path, cfg_dict, args.shape)
        except Exception as exc:
            print(f'Warning: complexity summary unavailable: {exc}')
            old_row = previous_summary.get(work_dir.name, {})
            flops = old_row.get('flops') or 'unavailable'
            params = old_row.get('params') or 'unavailable'

        if args.skip_existing and best_existing is not None:
            summary_rows.append({
                'run_name': work_dir.name,
                'variant': variant_name,
                'train_dataset': args.train_dataset,
                'base_config': relpath_for_config(config_path, repo_root),
                'image_query_source': image_query_source,
                'image_query_head_type': image_query_head_type,
                'cfg_options': cfg_opts_str,
                'flops': flops,
                'params': params,
                'best_mIoU': f'{best_existing:.6f}',
                'status': 'skipped_existing',
                'note': VARIANT_NOTES.get(variant_name, ''),
            })
            continue

        if args.summary_only:
            summary_rows.append({
                'run_name': work_dir.name,
                'variant': variant_name,
                'train_dataset': args.train_dataset,
                'base_config': relpath_for_config(config_path, repo_root),
                'image_query_source': image_query_source,
                'image_query_head_type': image_query_head_type,
                'cfg_options': cfg_opts_str,
                'flops': flops,
                'params': params,
                'best_mIoU': '' if best_existing is None
                else f'{best_existing:.6f}',
                'status': 'summary_only' if best_existing is not None
                else 'missing',
                'note': VARIANT_NOTES.get(variant_name, ''),
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
                'run_name': work_dir.name,
                'variant': variant_name,
                'train_dataset': args.train_dataset,
                'base_config': relpath_for_config(config_path, repo_root),
                'image_query_source': image_query_source,
                'image_query_head_type': image_query_head_type,
                'cfg_options': cfg_opts_str,
                'flops': flops,
                'params': params,
                'best_mIoU': '',
                'status': 'dry_run',
                'note': VARIANT_NOTES.get(variant_name, ''),
            })
            continue

        result = subprocess.run(command, check=False)
        best_miou = parse_best_miou(work_dir)
        summary_rows.append({
            'run_name': work_dir.name,
            'variant': variant_name,
            'train_dataset': args.train_dataset,
            'base_config': relpath_for_config(config_path, repo_root),
            'image_query_source': image_query_source,
            'image_query_head_type': image_query_head_type,
            'cfg_options': cfg_opts_str,
            'flops': flops,
            'params': params,
            'best_mIoU': '' if best_miou is None else f'{best_miou:.6f}',
            'status':
            'ok' if result.returncode == 0 else f'failed({result.returncode})',
            'note': VARIANT_NOTES.get(variant_name, ''),
        })
        write_summary(work_dir_root / 'summary.csv', summary_rows)
        write_markdown_summary(work_dir_root / 'summary.md', summary_rows)

    write_summary(work_dir_root / 'summary.csv', summary_rows)
    write_markdown_summary(work_dir_root / 'summary.md', summary_rows)


def write_summary(path: Path, rows):
    with path.open('w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                'run_name', 'variant', 'train_dataset', 'base_config',
                'image_query_source', 'image_query_head_type', 'cfg_options',
                'flops', 'params', 'best_mIoU', 'status', 'note'
            ])
        writer.writeheader()
        writer.writerows(rows)


def parse_size_to_number(value):
    if value in (None, ''):
        return None
    if isinstance(value, (int, float)):
        return float(value)
    match = re.match(r'^\s*([0-9.]+)\s*([KMGTP]?)\s*$', str(value))
    if not match:
        return None
    number = float(match.group(1))
    scale = match.group(2)
    multiplier = {
        '': 1,
        'K': 1e3,
        'M': 1e6,
        'G': 1e9,
        'T': 1e12,
        'P': 1e15,
    }[scale]
    return number * multiplier


def parse_float(value):
    try:
        if value in (None, ''):
            return None
        return float(value)
    except (TypeError, ValueError):
        return None


def fmt_delta(value, suffix=''):
    if value is None:
        return ''
    sign = '+' if value >= 0 else ''
    return f'{sign}{value:.3f}{suffix}'


def find_baseline(rows, variant):
    for row in rows:
        if row.get('variant') == variant and parse_float(row.get('best_mIoU')) is not None:
            return row
    return None


def write_markdown_summary(path: Path, rows):
    q4 = find_baseline(rows, 'brg-query-Q4-no-text')
    q5 = find_baseline(rows, 'brg-query-Q5-same-backbone-no-text')
    strict_baseline = q5 or q4
    loose_baseline = q4

    headers = [
        '变体', '数据集', 'mIoU', '相对同构基线', '相对旧 Q4',
        '参数量', '参数差', '每百万参数收益', '状态'
    ]
    lines = ['# CLIP Head 消融汇总', '', '| ' + ' | '.join(headers) + ' |',
             '| ' + ' | '.join(['---'] * len(headers)) + ' |']

    for row in rows:
        miou = parse_float(row.get('best_mIoU'))
        params = parse_size_to_number(row.get('params'))
        strict_miou = parse_float(strict_baseline.get('best_mIoU')) if strict_baseline else None
        loose_miou = parse_float(loose_baseline.get('best_mIoU')) if loose_baseline else None
        strict_params = parse_size_to_number(strict_baseline.get('params')) if strict_baseline else None

        delta_strict = miou - strict_miou if miou is not None and strict_miou is not None else None
        delta_loose = miou - loose_miou if miou is not None and loose_miou is not None else None
        delta_params = params - strict_params if params is not None and strict_params is not None else None
        if delta_strict is not None and delta_params not in (None, 0):
            gain_per_m = delta_strict / (delta_params / 1e6)
            if not math.isfinite(gain_per_m):
                gain_per_m = None
        else:
            gain_per_m = None

        lines.append('| ' + ' | '.join([
            row.get('variant') or row.get('run_name', ''),
            row.get('train_dataset', ''),
            '' if miou is None else f'{miou:.3f}',
            fmt_delta(delta_strict),
            fmt_delta(delta_loose),
            row.get('params', ''),
            '' if delta_params is None else f'{delta_params / 1e6:+.3f}M',
            '' if gain_per_m is None else f'{gain_per_m:+.3f}',
            row.get('status', ''),
        ]) + ' |')

    lines.extend(['', '## 备注', ''])
    if q5 is None:
        lines.append(
            '- 当前没有 `brg-query-Q5-same-backbone-no-text` 结果；'
            '表中的“相对同构基线”暂时退回旧 `Q4`，它与 CLIP 分支不是严格同构。')
    for row in rows:
        note = row.get('note', '')
        if note:
            lines.append(f'- `{row.get("variant")}`：{note}')

    path.write_text('\n'.join(lines) + '\n', encoding='utf-8')


if __name__ == '__main__':
    main()
