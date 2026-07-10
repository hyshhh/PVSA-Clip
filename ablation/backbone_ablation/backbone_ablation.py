
import argparse
import csv
import json
import os
import re
import shutil
import subprocess
import sys
from pathlib import Path
# 固定设置：
# 1. 解码头使用 clip head 消融得到的 CLIPSegHeadV2（actprompt）
# 2. 头部文本工作维固定为 256
# 3. 骨干文本注入维固定为 512
# 4. 本脚本只比较骨干文本注入路径、注入位置与注意力主路径
#
# 训练 KAKA：
# CUDA_VISIBLE_DEVICES=0 python ablation/backbone_ablation/backbone_ablation.py --work-dir-root ablation/backbone_ablation --shape 256 256 --train-dataset kaka --skip-existing
# 单独刷新 kaka 的训练摘要（不重新训练）
# python ablation/backbone_ablation/backbone_ablation.py --work-dir-root ablation/backbone_ablation --train-dataset kaka --summary-only
# 阶段一：骨干结构消融（顺序训练 kaka / gqy / gba，并分类保存）
# CUDA_VISIBLE_DEVICES=0 python ablation/backbone_ablation/backbone_ablation.py --phase structure --train-datasets kaka gqy gba --work-dir-root ablation/backbone_ablation --shape 256 256 --skip-existing
# 阶段二：RepRTA 文本重映射消融。将 --remap-backbone 改为阶段一选出的最优完整注入结构。
# CUDA_VISIBLE_DEVICES=0 python ablation/backbone_ablation/backbone_ablation.py --phase remap --remap-backbone bb-q3-route12-align3 --train-datasets kaka gqy gba --work-dir-root ablation/backbone_ablation --shape 256 256 --skip-existing
# 单独跑跨数据集泛化测试
# CUDA_VISIBLE_DEVICES=0 python ablation/backbone_ablation/backbone_ablation.py --phase remap --remap-backbone bb-q3-route12-align3 --variants bb-r1-route12-align3-prompts-reprta --train-dataset kaka --generalization-test --save-vis
CLIP_BRG_CONFIG = 'configs-h/clip/attn_waterseg.py'
CLIP_TOPP_CONFIG = 'configs-h/clip/waterseg.py'

FIXED_HEAD_CLIP_DIM = 256
FIXED_BACKBONE_TEXT_DIM = 512

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
    'bb-q0-no-inject':
    dict(
        alias='base',
        base_config=CLIP_BRG_CONFIG,
        use_backbone_text_injection=False,
        ttrm_stages=[],
        cross_attn_stages=[],
        note='固定 256 维 CLIP 头，仅关闭骨干文本注入，作为骨干消融基线。'),
    'bb-q1-route12':
    dict(
        alias='route12',
        base_config=CLIP_BRG_CONFIG,
        use_backbone_text_injection=True,
        ttrm_stages=[1, 2],
        cross_attn_stages=[],
        note='仅启用中层路由文本注入（stage1-2 的 TTRM），验证文本是否主要通过路由筛选生效。'),
    'bb-q2-align3':
    dict(
        alias='align3',
        base_config=CLIP_BRG_CONFIG,
        use_backbone_text_injection=True,
        ttrm_stages=[],
        cross_attn_stages=[3],
        note='仅在末层做文本-视觉特征对齐，验证直接特征注入的独立贡献。'),
    'bb-q3-route12-align3':
    dict(
        alias='route12_align3',
        base_config=CLIP_BRG_CONFIG,
        use_backbone_text_injection=True,
        ttrm_stages=[1, 2],
        cross_attn_stages=[3],
        note='推荐主方案：中层路由注入 + 末层特征对齐，在语义调制与定位稳定性之间折中最好。'),
    'bb-q4-route23-align3':
    dict(
        alias='route23_align3',
        base_config=CLIP_BRG_CONFIG,
        use_backbone_text_injection=True,
        ttrm_stages=[2, 3],
        cross_attn_stages=[3],
        note='深层路由注入：把文本调制后移到高语义阶段，验证晚注入是否更有效。'),
    'bb-q5-route01-align3':
    dict(
        alias='route01_align3',
        base_config=CLIP_BRG_CONFIG,
        use_backbone_text_injection=True,
        ttrm_stages=[0, 1],
        cross_attn_stages=[3],
        note='浅层路由注入：更早引入文本先验，验证其对边界细节与纹理噪声的影响。'),
    'bb-q6-topp-route12-align3':
    dict(
        alias='topp_route12_align3',
        base_config=CLIP_TOPP_CONFIG,
        use_backbone_text_injection=True,
        ttrm_stages=[1, 2],
        cross_attn_stages=[3],
        note='保持最优注入位置不变，仅将骨干主注意力从 BRG 切到 Topp，验证注意力家族差异。'),
    'bb-r0-prompts-prompts':
    dict(
        alias='remap_prompts_prompts',
        base_config=CLIP_BRG_CONFIG,
        use_backbone_text_injection=True,
        ttrm_stages=[1, 2],
        cross_attn_stages=[3],
        backbone_route_text_mode='prompts',
        backbone_align_text_mode='prompts',
        note='文本重映射基线：路由与对齐均使用 30 条展开提示；实际骨干结构由 --remap-backbone 固定。'),
    'bb-r1-route12-align3-prompts-reprta':
    dict(
        alias='remap_prompts_reprta',
        base_config=CLIP_BRG_CONFIG,
        use_backbone_text_injection=True,
        ttrm_stages=[1, 2],
        cross_attn_stages=[3],
        backbone_route_text_mode='prompts',
        backbone_align_text_mode='reprta',
        note='混合文本源：TTRM 保留 30 条展开提示做路由，末层 CrossAttention 改用 3 条 RepRTA 类别原型。'),
    'bb-r2-route12-align3-reprta-reprta':
    dict(
        alias='remap_reprta_reprta',
        base_config=CLIP_BRG_CONFIG,
        use_backbone_text_injection=True,
        ttrm_stages=[1, 2],
        cross_attn_stages=[3],
        backbone_route_text_mode='reprta',
        backbone_align_text_mode='reprta',
        note='纯 RepRTA 文本源：路由注入与末层对齐都只使用 3 条稳定类别原型。'),
    'bb-r3-route12-align3-reprta-reprta-noreprta':
    dict(
        alias='remap_reprta_noreprta',
        base_config=CLIP_BRG_CONFIG,
        use_backbone_text_injection=True,
        ttrm_stages=[1, 2],
        cross_attn_stages=[3],
        backbone_route_text_mode='reprta',
        backbone_align_text_mode='reprta',
        backbone_text_encoder_overrides=dict(use_reprta=False),
        note='去掉骨干文本编码器中的 RepRTA 残差精炼，仅保留提示池化后的类别原型。'),
    'bb-r4-route12-align3-reprta-reprta-gelu':
    dict(
        alias='remap_reprta_gelu',
        base_config=CLIP_BRG_CONFIG,
        use_backbone_text_injection=True,
        ttrm_stages=[1, 2],
        cross_attn_stages=[3],
        backbone_route_text_mode='reprta',
        backbone_align_text_mode='reprta',
        backbone_text_encoder_overrides=dict(reprta_ffn_type='gelu'),
        note='保持 RepRTA 开启，但把门控前馈从 SwiGLU 改为普通 GELU 前馈，验证门控结构本身的贡献。'),
    'bb-r5-route12-align3-reprta-reprta-nozero':
    dict(
        alias='remap_reprta_nozero',
        base_config=CLIP_BRG_CONFIG,
        use_backbone_text_injection=True,
        ttrm_stages=[1, 2],
        cross_attn_stages=[3],
        backbone_route_text_mode='reprta',
        backbone_align_text_mode='reprta',
        backbone_text_encoder_overrides=dict(reprta_zero_init=False),
        note='取消 RepRTA 最后一层零初始化保护，验证稳定残差起点是否关键。'),
}

QUERY_VARIANTS = list(VARIANT_SPECS)
STRUCTURE_VARIANTS = [
    'bb-q0-no-inject',
    'bb-q1-route12',
    'bb-q2-align3',
    'bb-q3-route12-align3',
    'bb-q4-route23-align3',
    'bb-q5-route01-align3',
    'bb-q6-topp-route12-align3',
]
# 阶段二复跑 prompts/prompts 基线，保证文本重映射比较使用同一训练设置。
REMAPPING_VARIANTS = [
    'bb-r0-prompts-prompts',
    'bb-r1-route12-align3-prompts-reprta',
    'bb-r2-route12-align3-reprta-reprta',
    'bb-r3-route12-align3-reprta-reprta-noreprta',
    'bb-r4-route12-align3-reprta-reprta-gelu',
    'bb-r5-route12-align3-reprta-reprta-nozero',
]
# 阶段二只接受同时包含路由和对齐的完整注入结构。
REMAPPING_BACKBONE_CHOICES = [
    'bb-q3-route12-align3',
    'bb-q4-route23-align3',
    'bb-q5-route01-align3',
    'bb-q6-topp-route12-align3',
]
DEFAULT_VARIANTS = STRUCTURE_VARIANTS


def parse_args():
    parser = argparse.ArgumentParser(
        description='Run backbone text-injection ablations.')
    parser.add_argument(
        '--work-dir-root',
        default='ablation/backbone_ablation',
        help='Root directory for ablation runs.')
    parser.add_argument(
        '--python',
        default=sys.executable,
        help='Python executable used to launch training.')
    parser.add_argument(
        '--phase',
        choices=('structure', 'remap'),
        default=None,
        help='Two-stage protocol: structure selects the backbone; remap fixes it and compares text remapping.')
    parser.add_argument(
        '--remap-backbone',
        choices=REMAPPING_BACKBONE_CHOICES,
        default='bb-q3-route12-align3',
        help='Best complete injection backbone selected in phase one. Only used by --phase remap.')
    parser.add_argument(
        '--variants',
        nargs='+',
        choices=QUERY_VARIANTS,
        default=None,
        help='Variant names to run. Overrides the phase default list.')
    parser.add_argument(
        '--extra-cfg-options',
        nargs='*',
        default=[],
        help='Additional cfg-options passed through to tools/train.py.')
    parser.add_argument(
        '--train-dataset',
        choices=sorted(TRAIN_DATASETS),
        default='kaka',
        help='Dataset config used for one-dataset training.')
    parser.add_argument(
        '--train-datasets',
        nargs='+',
        choices=sorted(TRAIN_DATASETS),
        default=None,
        help='Run datasets sequentially in one command; each dataset keeps its own train/summary directory.')
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


def get_effective_variant_spec(variant_name, remap_backbone=None):
    """返回实际执行的变体配置，并将阶段二固定到选定骨干。"""
    variant_spec = dict(get_variant_spec(variant_name))
    if variant_name not in REMAPPING_VARIANTS or remap_backbone is None:
        return variant_spec

    if remap_backbone not in REMAPPING_BACKBONE_CHOICES:
        raise ValueError(
            f'Remapping backbone must be one of {REMAPPING_BACKBONE_CHOICES}, '
            f'got {remap_backbone!r}.')
    backbone_spec = get_variant_spec(remap_backbone)
    variant_spec.update(
        base_config=backbone_spec['base_config'],
        use_backbone_text_injection=True,
        ttrm_stages=list(backbone_spec['ttrm_stages']),
        cross_attn_stages=list(backbone_spec['cross_attn_stages']))
    return variant_spec


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


def build_backbone_text_encoder_cfg(prompt_dataset='kaka', overrides=None):
    cfg = dict(
        embed_dim=FIXED_BACKBONE_TEXT_DIM,
        num_categories=3,
        prompts_per_category=10,
        prompt_bank_path='tools/prompt_bank_water.pt',
        prompt_category_order=PROMPT_CATEGORY_ORDERS[prompt_dataset],
        use_reprta=True,
        reprta_ffn_type='swiglu',
        reprta_zero_init=True)
    if overrides:
        cfg.update(overrides)
    return cfg


def build_text_refiner_cfg():
    return dict(in_dim=FIXED_BACKBONE_TEXT_DIM, hidden_mult=4)


def get_variant_runtime_settings(variant_name, prompt_dataset='kaka',
                                 remap_backbone=None):
    variant_spec = get_effective_variant_spec(variant_name, remap_backbone)
    ttrm_stages = list(variant_spec['ttrm_stages'])
    cross_attn_stages = list(variant_spec['cross_attn_stages'])
    use_backbone_text_injection = variant_spec['use_backbone_text_injection']
    backbone_route_text_mode = variant_spec.get('backbone_route_text_mode', 'prompts')
    backbone_align_text_mode = variant_spec.get('backbone_align_text_mode', 'prompts')
    encoder_overrides = dict(variant_spec.get('backbone_text_encoder_overrides', {}))
    needs_prompt_text = 'prompts' in {
        backbone_route_text_mode, backbone_align_text_mode}
    settings = dict(
        head_clip_dim=FIXED_HEAD_CLIP_DIM,
        backbone_text_dim=FIXED_BACKBONE_TEXT_DIM,
        prompt_order=PROMPT_CATEGORY_ORDERS[prompt_dataset],
        use_backbone_text_injection=use_backbone_text_injection,
        backbone_route_text_mode=backbone_route_text_mode,
        backbone_align_text_mode=backbone_align_text_mode,
        use_ttrm=bool(ttrm_stages),
        ttrm_stages=ttrm_stages,
        cross_attn_stages=cross_attn_stages,
        backbone_text_encoder=(
            build_backbone_text_encoder_cfg(prompt_dataset, encoder_overrides)
            if use_backbone_text_injection else None),
        text_refiner=(
            build_text_refiner_cfg()
            if use_backbone_text_injection and needs_prompt_text else None),
    )
    return settings


def build_cfg_options(variant_name, prompt_dataset='kaka',
                      remap_backbone=None):
    settings = get_variant_runtime_settings(
        variant_name, prompt_dataset, remap_backbone)
    cfg_list = [
        f"head_clip_dim={settings['head_clip_dim']}" ,
        f"clip_embed_dim={settings['head_clip_dim']}" ,
        f"backbone_text_dim={settings['backbone_text_dim']}" ,
        f"model.use_backbone_text_injection={settings['use_backbone_text_injection']!r}" ,
        f"model.backbone_route_text_mode={settings['backbone_route_text_mode']!r}" ,
        f"model.backbone_align_text_mode={settings['backbone_align_text_mode']!r}" ,
        f"model.text_encoder.prompt_category_order={settings['prompt_order']!r}" ,
        f"model.backbone.use_ttrm={settings['use_ttrm']!r}" ,
        f"model.backbone.ttrm_stages={settings['ttrm_stages']!r}" ,
        f"model.backbone.cross_attn_stages={settings['cross_attn_stages']!r}" ,
        f"model.backbone.text_dim={settings['backbone_text_dim']}" ,
    ]

    backbone_text_encoder = settings['backbone_text_encoder']
    text_refiner = settings['text_refiner']
    if backbone_text_encoder is None:
        cfg_list.append('model.backbone_text_encoder=None')
    else:
        cfg_list.append(f'model.backbone_text_encoder={backbone_text_encoder!r}')
    if text_refiner is None:
        cfg_list.append('model.text_refiner=None')
    else:
        cfg_list.append(f'model.text_refiner={text_refiner!r}')
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

    text = f'''# Auto-generated by ablation/backbone_ablation/backbone_ablation.py.
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
                      prompt_dataset: str, eval_shape,
                      remap_backbone=None):
    base_rel = relpath_for_config(repo_root / base_config,
                                  eval_config_path.parent)
    dataset_path = (repo_root / dataset_config).resolve().as_posix()
    settings = get_variant_runtime_settings(
        variant_name, prompt_dataset, remap_backbone)

    model_lines = [
        'model = dict(',
        '    data_preprocessor=data_preprocessor,',
        '    test_cfg=dict(mode="whole"),',
        f"    use_backbone_text_injection={settings['use_backbone_text_injection']!r}," ,
        f"    backbone_route_text_mode={settings['backbone_route_text_mode']!r}," ,
        f"    backbone_align_text_mode={settings['backbone_align_text_mode']!r}," ,
        '    text_encoder=dict(',
        f"        prompt_category_order={settings['prompt_order']!r}," ,
        '    ),',
        '    backbone=dict(',
        f"        use_ttrm={settings['use_ttrm']!r}," ,
        f"        ttrm_stages={settings['ttrm_stages']!r}," ,
        f"        cross_attn_stages={settings['cross_attn_stages']!r}," ,
        '        text_dim=backbone_text_dim,',
        '    ),',
    ]

    if settings['backbone_text_encoder'] is None:
        model_lines.extend([
            '    backbone_text_encoder=None,',
            '    text_refiner=None,',
        ])
    else:
        backbone_text_encoder = settings['backbone_text_encoder']
        text_refiner = settings['text_refiner']
        model_lines.extend([
            '    backbone_text_encoder=dict(',
            '        embed_dim=backbone_text_dim,',
            f"        num_categories={backbone_text_encoder['num_categories']!r}," ,
            f"        prompts_per_category={backbone_text_encoder['prompts_per_category']!r}," ,
            f"        prompt_bank_path={backbone_text_encoder['prompt_bank_path']!r}," ,
            f"        prompt_category_order={backbone_text_encoder['prompt_category_order']!r}," ,
            f"        use_reprta={backbone_text_encoder['use_reprta']!r}," ,
            f"        reprta_ffn_type={backbone_text_encoder['reprta_ffn_type']!r}," ,
            f"        reprta_zero_init={backbone_text_encoder['reprta_zero_init']!r}," ,
            '    ),',
        ])
        if text_refiner is None:
            model_lines.append('    text_refiner=None,')
        else:
            model_lines.extend([
                '    text_refiner=dict(',
                '        in_dim=backbone_text_dim,',
                f"        hidden_mult={text_refiner['hidden_mult']!r}," ,
                '    ),',
            ])
    model_lines.append(')')

    extra_lines = [
        f"head_clip_dim = {settings['head_clip_dim']!r}" ,
        f"clip_embed_dim = {settings['head_clip_dim']!r}" ,
        f"backbone_text_dim = {settings['backbone_text_dim']!r}" ,
    ]

    split_lines = []
    if split == 'val':
        split_lines.extend([
            'test_dataloader = val_dataloader',
        ])

    text = f'''# Auto-generated by ablation/backbone_ablation/backbone_ablation.py.
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
        variant_spec = get_effective_variant_spec(
            variant_name, args.remap_backbone)
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
                args.prompt_dataset, args.shape, args.remap_backbone)

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


def run_training_for_dataset(args, repo_root: Path, work_dir_root: Path,
                             train_dataset: str, prompt_dataset: str):
    """顺序执行一个数据集下的全部变体，并写入独立摘要。"""
    dataset_root = get_dataset_root(work_dir_root, train_dataset)
    dataset_root.mkdir(parents=True, exist_ok=True)
    summary_rows = []

    for variant_name in args.variants:
        variant_spec = get_effective_variant_spec(
            variant_name, args.remap_backbone)
        base_config = variant_spec['base_config']
        config_path = make_train_config(
            repo_root, work_dir_root, base_config, train_dataset)
        cfg_list = build_cfg_options(
            variant_name, prompt_dataset, args.remap_backbone)

        work_dir = get_work_dir(work_dir_root, variant_name, train_dataset)
        best_existing = parse_best_miou(work_dir)
        train_ious = train_iou_summary(work_dir, train_dataset)

        if args.skip_existing and best_existing is not None:
            summary_rows.append(
                make_train_summary_row(
                    train_dataset, variant_name, f'{best_existing:.6f}',
                    train_ious, 'skipped_existing'))
            continue

        if args.summary_only:
            summary_rows.append(
                make_train_summary_row(
                    train_dataset, variant_name,
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
                make_train_summary_row(
                    train_dataset, variant_name, '', train_ious, 'dry_run'))
            continue

        result = subprocess.run(command, check=False)
        best_miou = parse_best_miou(work_dir)
        train_ious = train_iou_summary(work_dir, train_dataset)
        summary_rows.append(
            make_train_summary_row(
                train_dataset, variant_name,
                '' if best_miou is None else f'{best_miou:.6f}', train_ious,
                'ok' if result.returncode == 0
                else f'failed({result.returncode})'))
        write_summary(dataset_root / 'summary.csv', summary_rows)
        write_markdown_summary(dataset_root / 'summary.md', summary_rows)

    write_summary(dataset_root / 'summary.csv', summary_rows)
    write_markdown_summary(dataset_root / 'summary.md', summary_rows)


def main():
    args = parse_args()
    if args.variants is None:
        if args.phase == 'structure':
            args.variants = STRUCTURE_VARIANTS
        elif args.phase == 'remap':
            args.variants = REMAPPING_VARIANTS
        else:
            args.variants = DEFAULT_VARIANTS

    train_datasets = args.train_datasets or [args.train_dataset]
    if args.prompt_dataset is not None and len(train_datasets) > 1:
        raise ValueError(
            '--prompt-dataset cannot be shared by multiple training datasets. '
            'Leave it unset to align each dataset automatically.')

    # 阶段目录下按数据集拆分，与 clip_head_fix_prompt 的结果组织一致。
    work_dir_root = Path(args.work_dir_root).resolve()
    if args.phase is not None:
        work_dir_root = work_dir_root / args.phase
    args.work_dir_root = str(work_dir_root)

    if args.generalization_test:
        if len(train_datasets) != 1:
            raise ValueError(
                '--generalization-test only supports one training dataset per run.')
        args.train_dataset = train_datasets[0]
        if args.prompt_dataset is None:
            args.prompt_dataset = args.train_dataset
        run_generalization_tests(args)
        return

    repo_root = Path(__file__).resolve().parents[2]
    for train_dataset in train_datasets:
        prompt_dataset = args.prompt_dataset or train_dataset
        run_training_for_dataset(
            args, repo_root, work_dir_root, train_dataset, prompt_dataset)


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
