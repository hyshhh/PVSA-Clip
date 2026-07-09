#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
消融实验 FLOPs / Params 批量统计脚本
=====================================
统计目标: configs-h/_base_/models/vision-topp-cnn.py (BiFormer_fusion_baseline) 链路
运行方式: python ablation/flops/ablation_flops.py [--shape 256 256]

模块映射（论文缩写 → 代码配置键）:
  FAM  = use_fam           (FeatureAlignmentModule 同层跨分支注意力对齐)
  PVSA = attention_type='topp'  (Top-P 路由自注意力，关闭时用 attention_type='bra')
  VFM  = cross_stage_fusion_mode  (跨层视觉特征融合模块)
  backbone 深度 = depth (Transformer) / cnn_block_layers (MBConv)
"""
import argparse
import csv
import sys
import os
from pathlib import Path
from copy import deepcopy

import torch
from mmengine import Config
from mmengine.model import revert_sync_batchnorm
from mmengine.analysis.print_helper import _format_size

# ── 项目根 & 注册 ──────────────────────────────────────────────────────────
PROJECT_ROOT = str(Path(__file__).resolve().parents[2])
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from mmseg.registry import MODELS
from mmseg.utils import register_all_modules, sync_clip_embed_dim
from mmseg.structures import SegDataSample

BASE_CONFIG = os.path.join(
    PROJECT_ROOT, 'configs-h', '_base_', 'models', 'vision-topp-cnn.py')


# ── 实验定义 ───────────────────────────────────────────────────────────────
# 每个实验格式: (实验编号, 实验名, config 覆盖字典)
# 覆盖字典的 key 使用点号嵌套表示法，由 _apply_overrides 展开赋值。

EXPERIMENTS = [
    # ════════════════════════════════════════════════════════════════════════
    # 实验组 1：整体消融  (FFM=FAM / PVSA / VFM 三开关)
    # ════════════════════════════════════════════════════════════════════════
    ('1-1', 'Baseline (FFM× PVSA× VFM×)', {
        'model.backbone.use_fam': False,
        'model.backbone.attention_type': 'bra',
        'model.backbone.cross_stage_fusion_mode': 'none',
    }),
    ('1-2', 'FFM√ PVSA× VFM×', {
        'model.backbone.use_fam': True,
        'model.backbone.attention_type': 'bra',
        'model.backbone.cross_stage_fusion_mode': 'none',
    }),
    ('1-3', 'FFM× PVSA√ VFM×', {
        'model.backbone.use_fam': False,
        'model.backbone.attention_type': 'topp',
        'model.backbone.cross_stage_fusion_mode': 'none',
    }),
    ('1-4', 'FFM√ PVSA× VFM√', {
        'model.backbone.use_fam': True,
        'model.backbone.attention_type': 'bra',
        'model.backbone.cross_stage_fusion_mode': 'cross_concat',
    }),
    ('1-5', 'Full (FFM√ PVSA√ VFM√)', {
        'model.backbone.use_fam': True,
        'model.backbone.attention_type': 'topp',
        'model.backbone.cross_stage_fusion_mode': 'cross_concat',
    }),

    # ════════════════════════════════════════════════════════════════════════
    # 实验组 2-A：骨干消融 (MBConv-Block / PVSA-Block 深度组合)
    # ════════════════════════════════════════════════════════════════════════
    ('2-A-1', 'MBConv[0,0,0,0] PVSA[3,4,6,3]', {
        'model.backbone.cnn_block_layers': [0, 0, 0, 0],
        'model.backbone.depth': [3, 4, 6, 3],
    }),
    ('2-A-2', 'MBConv[1,1,1,1] PVSA[3,4,6,3]', {
        'model.backbone.cnn_block_layers': [1, 1, 1, 1],
        'model.backbone.depth': [3, 4, 6, 3],
    }),
    ('2-A-3', 'MBConv[1,2,2,1] PVSA[1,3,4,2]', {
        'model.backbone.cnn_block_layers': [1, 2, 2, 1],
        'model.backbone.depth': [1, 3, 4, 2],
    }),
    ('2-A-4', 'MBConv[1,2,2,1] PVSA[2,6,8,4]', {
        'model.backbone.cnn_block_layers': [1, 2, 2, 1],
        'model.backbone.depth': [2, 6, 8, 4],
    }),
    ('2-A-5', 'MBConv[1,2,2,1] PVSA[3,4,6,3] (full)', {
        'model.backbone.cnn_block_layers': [1, 2, 2, 1],
        'model.backbone.depth': [3, 4, 6, 3],
    }),

    # ════════════════════════════════════════════════════════════════════════
    # 实验组 2-B：融合方式对比
    # ════════════════════════════════════════════════════════════════════════
    ('2-B-1', 'C+T (sequential, CNN→Trans)', {
        'model.backbone.type': 'BiFormer_sequential',
        'model.backbone.branch_order': 'cnn_first',
        'model.backbone.attention_type': 'bra',
    }),
    ('2-B-2', 'T+C (sequential, Trans→CNN)', {
        'model.backbone.type': 'BiFormer_sequential',
        'model.backbone.branch_order': 'trans_first',
        'model.backbone.attention_type': 'bra',
    }),
    ('2-B-3', 'TC1 (parallel, concat only)', {
        # 无 FAM、无 VFM，双分支直接 concat → 1×1 conv
        'model.backbone.attention_type': 'bra',
        'model.backbone.use_fam': False,
        'model.backbone.cross_stage_fusion_mode': 'none',
    }),
    ('2-B-4', 'TC2 (parallel, FAM + VFM)', {
        # FAM + VFM 全开，注意力用 BRA
        'model.backbone.attention_type': 'bra',
        'model.backbone.use_fam': True,
        'model.backbone.cross_stage_fusion_mode': 'cross_concat',
    }),
]


# ── 工具函数 ───────────────────────────────────────────────────────────────
def _apply_overrides(cfg: Config, overrides: dict) -> Config:
    """
    将 'a.b.c': val 格式的覆盖应用到 Config 对象上。
    直接操作 cfg._cfg_dict（OrderedDict），不经过 merge_from_dict，
    避免 _delete_ 等特殊标记的兼容性问题。
    """
    for dotkey, value in overrides.items():
        keys = dotkey.split('.')
        d = cfg._cfg_dict
        for k in keys[:-1]:
            if k not in d:
                raise KeyError(
                    f"覆盖键 '{dotkey}' 的中间层 '{k}' 不存在，"
                    f"可用键: {list(d.keys())}")
            d = d[k]
        d[keys[-1]] = value
    return cfg


def _build_model(cfg: Config):
    """构建模型并转为 eval 模式，跳过数据预处理。"""
    register_all_modules(init_default_scope=True)
    model = MODELS.build(cfg.model)
    if hasattr(model, 'auxiliary_head'):
        model.auxiliary_head = None
    model = revert_sync_batchnorm(model)
    model.eval()
    if hasattr(model, 'backbone'):
        model.backbone._disable_inference_fusion = True
    return model


def _measure(model, input_shape=(3, 256, 256)):
    """用 forward hook 统计 FLOPs / Params（兼容 fvcore 对 nn.Identity 的 bug）。"""
    flops_dict = {}
    hooks = []

    def _make_hook(name):
        def hook_fn(module, inp, out):
            flops = 0
            if isinstance(module, torch.nn.Linear):
                flops = 2 * inp[0].shape[0] * module.in_features * module.out_features
            elif isinstance(module, torch.nn.Conv2d):
                out_h, out_w = out.shape[2], out.shape[3]
                flops = (2 * module.in_channels * module.out_channels *
                         module.kernel_size[0] * module.kernel_size[1] *
                         out_h * out_w // module.groups)
            elif isinstance(module, torch.nn.BatchNorm2d):
                flops = inp[0].numel() * 2
            if flops > 0:
                flops_dict[name] = flops
        return hook_fn

    for name, module in model.named_modules():
        hooks.append(module.register_forward_hook(_make_hook(name)))

    data = torch.rand(1, *input_shape)
    seg_sample = SegDataSample(metainfo={
        'ori_shape': input_shape[-2:],
        'pad_shape': input_shape[-2:],
        'img_shape': input_shape[-2:],
    })
    with torch.no_grad():
        if hasattr(model, 'data_preprocessor') and model.data_preprocessor is not None:
            out = model.data_preprocessor(
                {'inputs': [data], 'data_samples': [seg_sample]})
            model(out['inputs'], out['data_samples'], mode='predict')
        else:
            model([data], [seg_sample], mode='predict')

    for h in hooks:
        h.remove()

    total_flops = sum(flops_dict.values())
    total_params = sum(p.numel() for p in model.parameters())
    del model
    torch.cuda.empty_cache() if torch.cuda.is_available() else None
    return _format_size(total_flops), _format_size(total_params)


# ── 主流程 ──────────────────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser(description='消融实验 FLOPs/Params 批量统计')
    parser.add_argument('--shape', type=int, nargs=2, default=[256, 256],
                        help='输入图像尺寸 H W， 默认 256 256')
    args = parser.parse_args()
    input_shape = (3, *args.shape)

    # 加载基线 config（含最新 use_fam 参数）
    base_cfg = Config.fromfile(BASE_CONFIG)
    sync_clip_embed_dim(base_cfg)

    print('=' * 76)
    print(f'  消融实验 FLOPs / Params 统计')
    print(f'  基线配置: {BASE_CONFIG}')
    print(f'  输入尺寸: {input_shape}')
    print('=' * 76)

    results = []

    for exp_id, exp_name, overrides in EXPERIMENTS:
        print(f'\n▶ [{exp_id}] {exp_name}')
        cfg = deepcopy(base_cfg)
        _apply_overrides(cfg, overrides)
        try:
            model = _build_model(cfg)
            flops, params = _measure(model, input_shape)
            results.append((exp_id, exp_name, params, flops))
            print(f'  Params: {params}  |  FLOPs: {flops}')
        except Exception as e:
            results.append((exp_id, exp_name, 'ERROR', str(e)[:60]))
            print(f'  ✗ 错误: {e}')

    # ── 汇总表 ────────────────────────────────────────────────────────────
    print('\n' + '=' * 76)
    print('  汇总结果')
    print('=' * 76)
    header = f'{"编号":<10} {"实验名":<42} {"Params":>10} {"FLOPs":>12}'
    print(header)
    print('-' * 76)

    prev_group = ''
    for exp_id, exp_name, params, flops in results:
        group = exp_id.split('-')[0]
        if group != prev_group:
            if prev_group:
                print('-' * 76)
            prev_group = group
        print(f'{exp_id:<10} {exp_name:<42} {params:>10} {flops:>12}')

    print('-' * 76)
    print('注: PVSA=attention_type\'topp\'(Top-P路由+route_mask), BRA=attention_type\'bra\'(标准固定top-k)。')
    print('    C+T / T+C 为顺序分支（BiFormer_sequential），通过 branch_order 控制方向。')
    print('    TC1/TC2 为并行双分支（BiFormer_fusion_baseline），通过 use_fam / cross_stage_fusion_mode 控制。')
    print('=' * 76)

    # ── 保存 CSV ─────────────────────────────────────────────────────────
    csv_path = os.path.join(os.path.dirname(__file__), 'ablation_results.csv')
    with open(csv_path, 'w', encoding='utf-8-sig', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['编号', '实验名', 'Params', 'FLOPs'])
        for exp_id, exp_name, params, flops in results:
            writer.writerow([exp_id, exp_name, params, flops])
    print(f'\n结果已保存: {csv_path}')


if __name__ == '__main__':
    main()
