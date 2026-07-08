
import argparse
import csv
import json
import os
import re
import shutil
import subprocess
import sys
from pathlib import Path
# 单独统计三个消融的参数量和计算量：
# python tools/analysis_tools/get_flops.py configs-h/vision/attn_ablation_waterseg.py --shape 256 256
# python tools/analysis_tools/get_flops.py configs-h/vision/attn_clipbackbone_seghead_waterseg.py --shape 256 256
# python tools/analysis_tools/get_flops.py configs-h/clip/attn_waterseg.py --shape 256 256 --cfg-options model.decode_head.type=CLIPSegHeadV2 model.text_encoder.prompt_category_order="['land','ship','water']"
# python tools/analysis_tools/get_flops.py configs-h/clip/attn_waterseg.py --shape 256 256 --cfg-options clip_embed_dim=256 model.decode_head.type=CLIPSegHeadV2 model.text_encoder.prompt_category_order="['land','ship','water']"
#
# 训练 KAKA：background / boat / free-space -> land / ship / water
# CUDA_VISIBLE_DEVICES=0 python ablation/clip_head_fix_prompt/clip_head.py --work-dir-root ablation/clip_head_fix_prompt --shape 256 256 --variants brg-query-Q4-no-text brg-query-Q5-same-backbone-no-text clip-v2-actprompt --train-dataset kaka --skip-existing
# 单独刷新 kaka 的训练摘要（不重新训练）
# python ablation/clip_head_fix_prompt/clip_head.py --work-dir-root ablation/clip_head_fix_prompt --train-dataset kaka --summary-only
# 单独跑 headv2 并测试：--skip-existing 跳过已有 best mIoU 的运行
# CUDA_VISIBLE_DEVICES=0 python ablation/clip_head_fix_prompt/clip_head.py --work-dir-root ablation/clip_head_fix_prompt --shape 256 256 --variants clip-v2-actprompt --train-dataset kaka --skip-existing
# 单独跑 headv2-256：TextEncoder 会自动把 512 维 prompt bank 映射到 256 维工作空间
# CUDA_VISIBLE_DEVICES=0 python ablation/clip_head_fix_prompt/clip_head.py --work-dir-root ablation/clip_head_fix_prompt --shape 256 256 --variants clip-v2-actprompt-d256 --train-dataset kaka --skip-existing
# CUDA_VISIBLE_DEVICES=0 python ablation/clip_head_fix_prompt/clip_head.py --work-dir-root ablation/clip_head_fix_prompt --variants clip-v2-actprompt --train-dataset kaka --generalization-test --save-vis

# 训练 gqy：water / ground / object -> water / land / ship
# CUDA_VISIBLE_DEVICES=0 python ablation/clip_head_fix_prompt/clip_head.py --work-dir-root ablation/clip_head_fix_prompt --shape 256 256 --variants brg-query-Q4-no-text brg-query-Q5-same-backbone-no-text clip-v2-actprompt --train-dataset gqy --skip-existing
# 单独刷新 gqy 的训练摘要（不重新训练）
# python ablation/clip_head_fix_prompt/clip_head.py --work-dir-root ablation/clip_head_fix_prompt --train-dataset gqy --summary-only
# 单独跑 headv2 并测试
# CUDA_VISIBLE_DEVICES=0 python ablation/clip_head_fix_prompt/clip_head.py --work-dir-root ablation/clip_head_fix_prompt --shape 256 256 --variants clip-v2-actprompt --train-dataset gqy --skip-existing
# 单独跑 headv2-256
# CUDA_VISIBLE_DEVICES=0 python ablation/clip_head_fix_prompt/clip_head.py --work-dir-root ablation/clip_head_fix_prompt --shape 256 256 --variants clip-v2-actprompt-d256 --train-dataset gqy --skip-existing
# CUDA_VISIBLE_DEVICES=0 python ablation/clip_head_fix_prompt/clip_head.py --work-dir-root ablation/clip_head_fix_prompt --variants clip-v2-actprompt --train-dataset gqy --generalization-test --save-vis
#
# 训练 GBA：object / water / ground -> ship / water / land
# CUDA_VISIBLE_DEVICES=0 python ablation/clip_head_fix_prompt/clip_head.py --work-dir-root ablation/clip_head_fix_prompt --shape 256 256 --variants brg-query-Q4-no-text brg-query-Q5-same-backbone-no-text clip-v2-actprompt --train-dataset gba --skip-existing
# 单独刷新 gba 的训练摘要（不重新训练）
# python ablation/clip_head_fix_prompt/clip_head.py --work-dir-root ablation/clip_head_fix_prompt --train-dataset gba --summary-only
# 单独跑 headv2 并测试
# CUDA_VISIBLE_DEVICES=0 python ablation/clip_head_fix_prompt/clip_head.py --work-dir-root ablation/clip_head_fix_prompt --shape 256 256 --variants clip-v2-actprompt --train-dataset gba --skip-existing
# 单独跑 headv2-256
# CUDA_VISIBLE_DEVICES=0 python ablation/clip_head_fix_prompt/clip_head.py --work-dir-root ablation/clip_head_fix_prompt --shape 256 256 --variants clip-v2-actprompt-d256 --train-dataset gba --skip-existing
# CUDA_VISIBLE_DEVICES=0 python ablation/clip_head_fix_prompt/clip_head.py --work-dir-root ablation/clip_head_fix_prompt --variants clip-v2-actprompt --train-dataset gba --generalization-test --save-vis
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
        ],
        classes=('background', 'boat', 'free-space')),
    gqy=dict(
        dataset_candidates=['configs-h/_base_/datasets/gqy.py'],
        classes=('water', 'ground', 'object')),
    gba=dict(
        dataset_candidates=['configs-h/_base_/datasets/GBA.py'],
        classes=('object', 'water', 'ground')))

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

VARIANT_SPECS = {
    'brg-query-Q4-no-text':
    dict(
        alias='q4',
        base_config=VISION_BRG_CONFIG,
        decode_head_type=None,
        note='旧无文本基线：EncoderDecoder + BiFormer_standalone + SegformerHead；'
        '不和 CLIP 分支严格同构。'),
    'brg-query-Q5-same-backbone-no-text':
    dict(
        alias='q5',
        base_config=VISION_CLIP_BACKBONE_CONFIG,
        decode_head_type=None,
        note='公平无文本基线：同 CLIP 分支的 BiFormer_fusion_clip 骨干 + '
        'SegformerHead。'),
    'clip-v2-actprompt':
    dict(
        alias='v2',
        base_config=CLIP_BRG_CONFIG,
        decode_head_type='CLIPSegHeadV2',
        clip_embed_dim=512,
        note='CLIPSegHead v2：普通视觉分支 + 类激活视觉提示 + 文本分支辅助监督。'
        '显式锁定 512 维（覆盖 base 默认），与 d256 构成维度对照。'),
    'clip-v2-actprompt-d256':
    dict(
        alias='v2-d256',
        base_config=CLIP_BRG_CONFIG,
        decode_head_type='CLIPSegHeadV2',
        clip_embed_dim=256,
        note='CLIPSegHead v2-256：将视觉提示和文本工作维统一降到 256；'
        'TextEncoder 自动完成 512/256 prompt bank 映射。'),
}

QUERY_VARIANTS = list(VARIANT_SPECS)
DEFAULT_VARIANTS = [
    'brg-query-Q4-no-text',
    'brg-query-Q5-same-backbone-no-text',
    'clip-v2-actprompt',
]


def parse_args():
    parser = argparse.ArgumentParser(
        description='Run CLIP head image-query ablations.')
    parser.add_argument(
        '--work-dir-root',
        default='ablation/clip_head_fix_prompt',
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
        '--save-vis',
        action='store_true',
        help='Save painted prediction images for --generalization-test.')
    parser.add_argument(
        '--shape',
        type=int,
        nargs='+',
        default=[256, 256],
        help='Input shape used for eval resize.')
    return parser.parse_args()


def get_variant_spec(variant_name):
    try:
        return VARIANT_SPECS[variant_name]
    except KeyError as exc:
        raise ValueError(f'Unknown variant: {variant_name}. '
                         f'Valid: {QUERY_VARIANTS}') from exc


def get_variant_display_name(variant_name):
    return get_variant_spec(variant_name)['alias']


def get_variant_note(variant_name):
    return get_variant_spec(variant_name)['note']


def get_train_dataset_classes(train_dataset: str):
    return TRAIN_DATASETS[train_dataset]['classes']


def sanitize_class_name(class_name: str):
    return class_name.replace(' ', '_').replace('-', '_')


def get_train_summary_fields(train_dataset: str):
    fields = ['dataset', 'mIoU']
    fields.extend(
        f'IoU_{sanitize_class_name(class_name)}'
        for class_name in get_train_dataset_classes(train_dataset))
    fields.extend(['ablation', 'status'])
    return fields


def get_dataset_root(work_dir_root: Path, train_dataset: str):
    return work_dir_root / train_dataset


def get_work_dir(work_dir_root: Path, variant_name: str, train_dataset: str):
    return (
        get_dataset_root(work_dir_root, train_dataset) / 'train'
        / get_variant_display_name(variant_name))


def build_cfg_options(variant_name, prompt_dataset='kaka'):
    variant_spec = get_variant_spec(variant_name)
    decode_head_type = variant_spec['decode_head_type']
    cfg_list = []
    if decode_head_type is not None:
        cfg_list.append(f'model.decode_head.type={decode_head_type}')
        prompt_order = PROMPT_CATEGORY_ORDERS[prompt_dataset]
        cfg_list.append(
            f'model.text_encoder.prompt_category_order={prompt_order!r}')
    clip_embed_dim = variant_spec.get('clip_embed_dim')
    if clip_embed_dim is not None:
        cfg_list.append(f'clip_embed_dim={clip_embed_dim}')

    return cfg_list


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

    train_config_root = (
        get_dataset_root(work_dir_root, train_dataset) / '_train_configs')
    train_config_root.mkdir(parents=True, exist_ok=True)
    train_config_path = train_config_root / (
        f'{Path(base_config).stem}__{train_dataset}.py')
    base_rel = relpath_for_config(base_path, train_config_path.parent)
    dataset_abs = dataset_path.as_posix()

    text = f'''# Auto-generated by ablation/clip_head_fix_prompt/clip_head.py.
_base_ = [
    '{base_rel}',
]

_dataset_config = '{dataset_abs}'
with open(_dataset_config, 'r', encoding='utf-8') as _f:
    exec(compile(_f.read(), _dataset_config, 'exec'))
del _dataset_config, _f

_train_dataset = train_dataloader['dataset']
_val_dataset = val_dataloader['dataset']
_test_dataset = test_dataloader['dataset']
train_dataloader = dict(dataset=_train_dataset)
val_dataloader = dict(dataset=_val_dataset)
test_dataloader = dict(dataset=_test_dataset)
del _train_dataset, _val_dataset, _test_dataset

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

model = dict(
    data_preprocessor=data_preprocessor,
    test_cfg=dict(mode='whole'))
'''
    train_config_path.write_text(text, encoding='utf-8')
    return train_config_path


def write_eval_config(eval_config_path: Path, repo_root: Path,
                      base_config: str, dataset_config: str, split: str,
                      target_metainfo: dict, variant_name: str,
                      prompt_dataset: str, eval_shape):
    base_rel = relpath_for_config(repo_root / base_config,
                                  eval_config_path.parent)
    dataset_path = (repo_root / dataset_config).resolve().as_posix()

    model_lines = [
        'model = dict(',
        '    data_preprocessor=data_preprocessor,',
        '    test_cfg=dict(mode="whole"),',
    ]
    variant_spec = get_variant_spec(variant_name)
    decode_head_type = variant_spec['decode_head_type']
    clip_embed_dim = variant_spec.get('clip_embed_dim')
    extra_lines = []
    if clip_embed_dim is not None:
        extra_lines.append(f'clip_embed_dim = {clip_embed_dim!r}')
    if decode_head_type is not None:
        prompt_order = PROMPT_CATEGORY_ORDERS[prompt_dataset]
        model_lines.extend([
            '    decode_head=dict(',
            f'        type="{decode_head_type}",',
            '    ),',
            '    text_encoder=dict(',
            f'        prompt_category_order={prompt_order!r},',
            '    ),',
        ])
        # 维度不由这里手写，改由顶层 `clip_embed_dim` 一处声明，
        # 经 tools/test.py 的 sync_clip_embed_dim 同步回填到
        # decode_head.embed_dim / text_encoder.embed_dim / text_refiner.in_dim。
    model_lines.append(')')

    split_lines = []
    if split == 'val':
        split_lines.extend([
            'test_dataloader = val_dataloader',
        ])

    text = f'''# Auto-generated by ablation/clip_head_fix_prompt/clip_head.py.
_base_ = [
    '{base_rel}',
]

_dataset_config = '{dataset_path}'
with open(_dataset_config, 'r', encoding='utf-8') as _f:
    exec(compile(_f.read(), _dataset_config, 'exec'))
del _dataset_config, _f

crop_size = {tuple(eval_shape)!r}
img_scale = crop_size
for _pipeline_name in ('val_pipeline', 'test_pipeline'):
    if _pipeline_name in globals():
        for _step in globals()[_pipeline_name]:
            if isinstance(_step, dict) and _step.get('type') == 'Resize':
                _step['scale'] = crop_size

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
default_hooks = globals().get('default_hooks', {{}})
default_hooks.update(dict(visualization=dict(type='SegVisualizationHook')))

{chr(10).join(extra_lines)}
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
        if p.name not in ('summary.csv', 'eval_summary.csv'))
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


def make_train_summary_row(train_dataset: str, variant_name: str,
                           best_miou: str, train_ious: dict, status: str):
    row = {
        'dataset': train_dataset,
        'mIoU': best_miou,
        'ablation': get_variant_display_name(variant_name),
        'status': status,
    }
    for class_name in get_train_dataset_classes(train_dataset):
        key = f'IoU_{sanitize_class_name(class_name)}'
        row[key] = train_ious.get(key, '')
    return row


def run_generalization_tests(args):
    repo_root = Path(__file__).resolve().parents[2]
    work_dir_root = Path(args.work_dir_root).resolve()
    dataset_root = get_dataset_root(work_dir_root, args.train_dataset)
    eval_config_root = dataset_root / '_eval_configs'
    eval_work_root = dataset_root / 'eval'
    eval_work_root.mkdir(parents=True, exist_ok=True)

    summary_rows = []

    for variant_name in args.variants:
        variant_spec = get_variant_spec(variant_name)
        base_config = variant_spec['base_config']
        variant_alias = get_variant_display_name(variant_name)
        checkpoint = find_checkpoint(
            get_work_dir(work_dir_root, variant_name, args.train_dataset))

        for target in GENERALIZATION_TARGETS:
            dataset_config, dataset_path = resolve_dataset_config(
                repo_root, target['dataset_candidates'])
            ablation_name = variant_alias
            run_name = f'to_{target["display_name"]}_{variant_alias}'
            eval_work_dir = (
                eval_work_root / f'to_{target["display_name"]}'
                / variant_alias)

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
                target['split'], target['metainfo'], variant_name,
                args.prompt_dataset, args.shape)

            command = [
                args.python,
                'tools/test.py',
                str(eval_config),
                str(checkpoint),
                '--work-dir',
                str(eval_work_dir),
            ]
            if args.save_vis:
                command.extend(['--show-dir', str(eval_work_dir / 'vis')])
            command.extend(args.extra_test_args)

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
                dataset_root / 'eval_summary.csv', summary_rows)

    write_generalization_summary(
        dataset_root / 'eval_summary.csv', summary_rows)


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


def train_iou_summary(work_dir: Path, train_dataset: str):
    metrics = parse_test_metrics(work_dir)
    values = {}
    for class_name in get_train_dataset_classes(train_dataset):
        values[f'IoU_{sanitize_class_name(class_name)}'] = class_iou(
            metrics, class_name)
    return values


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

    repo_root = Path(__file__).resolve().parents[2]
    work_dir_root = Path(args.work_dir_root).resolve()
    dataset_root = get_dataset_root(work_dir_root, args.train_dataset)
    dataset_root.mkdir(parents=True, exist_ok=True)

    summary_rows = []

    for variant_name in args.variants:
        variant_spec = get_variant_spec(variant_name)
        base_config = variant_spec['base_config']
        config_path = make_train_config(
            repo_root, work_dir_root, base_config, args.train_dataset)
        cfg_list = build_cfg_options(variant_name, args.prompt_dataset)

        work_dir = get_work_dir(work_dir_root, variant_name,
                                args.train_dataset)
        best_existing = parse_best_miou(work_dir)
        train_ious = train_iou_summary(work_dir, args.train_dataset)

        if args.skip_existing and best_existing is not None:
            summary_rows.append(
                make_train_summary_row(
                    args.train_dataset, variant_name, f'{best_existing:.6f}',
                    train_ious, 'skipped_existing'))
            continue

        if args.summary_only:
            summary_rows.append(
                make_train_summary_row(
                    args.train_dataset, variant_name,
                    '' if best_existing is None else f'{best_existing:.6f}',
                    train_ious,
                    'summary_only' if best_existing is not None else 'missing'))
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
            summary_rows.append(
                make_train_summary_row(args.train_dataset, variant_name, '',
                                       train_ious, 'dry_run'))
            continue

        result = subprocess.run(command, check=False)
        best_miou = parse_best_miou(work_dir)
        train_ious = train_iou_summary(work_dir, args.train_dataset)
        summary_rows.append(
            make_train_summary_row(
                args.train_dataset, variant_name,
                '' if best_miou is None else f'{best_miou:.6f}', train_ious,
                'ok' if result.returncode == 0
                else f'failed({result.returncode})'))
        write_summary(dataset_root / 'summary.csv', summary_rows)
        write_markdown_summary(dataset_root / 'summary.md', summary_rows)

    write_summary(dataset_root / 'summary.csv', summary_rows)
    write_markdown_summary(dataset_root / 'summary.md', summary_rows)


def write_summary(path: Path, rows):
    train_dataset = path.parent.name
    fieldnames = get_train_summary_fields(train_dataset)
    with path.open('w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def write_markdown_summary(path: Path, rows):
    train_dataset = path.parent.name
    fieldnames = get_train_summary_fields(train_dataset)
    lines = ['# Train Summary', '', '| ' + ' | '.join(fieldnames) +
             ' |', '| ' + ' | '.join(['---'] * len(fieldnames)) +
             ' |']

    for row in rows:
        lines.append('| ' + ' | '.join(
            [str(row.get(field, '')) for field in fieldnames]) +
                     ' |')

    lines.extend(['', '## 备注', ''])
    seen = set()
    for row in rows:
        ablation = row.get('ablation', '')
        if not ablation or ablation in seen:
            continue
        seen.add(ablation)
        for variant_name in QUERY_VARIANTS:
            if get_variant_display_name(variant_name) != ablation:
                continue
            lines.append(f'- `{ablation}`：{get_variant_note(variant_name)}')
            break

    path.write_text('\n'.join(lines) + '\n', encoding='utf-8')


if __name__ == '__main__':
    main()
