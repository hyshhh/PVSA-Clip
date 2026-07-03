"""
对比实验配置文件 — 与 PVSA-Net (BiFormer_fusion_baseline) 对比的基线模型

模型定义在 configs-h/_base_/compare_models/ 下，通过 _base_ 引用。
切换模型：取消注释对应的 _base_ 行，注释掉其余的。

训练：
  CUDA_VISIBLE_DEVICES=0 python tools/train.py configs-h/vision/baselines_compare.py --work-dir work_dirs/<model_name>

测试：
  CUDA_VISIBLE_DEVICES=0 python tools/test.py configs-h/vision/baselines_compare.py work_dirs/<model_name>/best_mIoU.pth
"""

_base_ = [
    # ============ 数据集（取消注释一个） ============
    '../_base_/datasets/gqy.py',
    # '../_base_/datasets/KAKA.py',
    # ============ 运行时 ============
    '../_base_/default_runtime.py',
    # ============ 训练策略 ============
    '../_base_/schedules/schedule_20k.py',
    # ============ 模型（取消注释一个） ============
    '../_base_/compare_models/deeplabv3plus_r18.py',
    # '../_base_/compare_models/swin_t_upernet.py',
    # '../_base_/compare_models/segformer_b2.py',
    # '../_base_/compare_models/biformer_s.py',
    # '../_base_/compare_models/biformer_b.py',
]

# ======================== 公共训练设置 ========================
crop_size = (256, 256)
data_preprocessor = dict(
    type='SegDataPreProcessor',
    mean=[123.675, 116.28, 103.53],
    std=[58.395, 57.12, 57.375],
    bgr_to_rgb=True,
    pad_val=0,
    seg_pad_val=255,
    size=crop_size,
)

train_dataloader = dict(batch_size=16, num_workers=8,
                        sampler=dict(type='DefaultSampler', shuffle=True))
val_dataloader = dict(batch_size=4, num_workers=2)
test_dataloader = dict(batch_size=1, num_workers=1)

train_cfg = dict(type='EpochBasedTrainLoop', max_epochs=200, val_interval=10)
val_cfg = dict(type='ValLoop')
test_cfg = dict(type='TestLoop')

val_evaluator = dict(type='IoUMetric', iou_metrics=['mIoU', 'mDice'],
                     ignore_index=255, classwise=True)
test_evaluator = val_evaluator

# ======================== 优化器 ========================
# Transformer 模型 (Swin-T / SegFormer / BiFormer) 用 AdamW
# CNN 模型 (DeepLabV3+) 用 SGD，切换时取消注释下面 SGD 块、注释 AdamW 块

# ---- AdamW（默认，Transformer 模型） ----
optim_wrapper = dict(
    _delete_=True,
    type='OptimWrapper',
    optimizer=dict(type='AdamW', lr=6e-4, betas=(0.9, 0.999),
                   weight_decay=0.01),
    paramwise_cfg=dict(
        custom_keys={
            'norm': dict(decay_mult=0.0),
            'head': dict(lr_mult=10.0),
        }),
)
param_scheduler = [
    dict(type='LinearLR', start_factor=0.001, by_epoch=True, begin=0, end=10),
    dict(type='PolyLR', eta_min=1e-6, power=1.0, by_epoch=True,
         begin=10, end=200),
]

# ---- SGD（DeepLabV3+ 时取消注释） ----
# optim_wrapper = dict(
#     _delete_=True,
#     type='OptimWrapper',
#     optimizer=dict(type='SGD', lr=0.01, momentum=0.9, weight_decay=0.0005),
#     paramwise_cfg=dict(
#         custom_keys={
#             'norm': dict(decay_mult=0.0),
#             'head': dict(lr_mult=10.0),
#         }),
# )
# param_scheduler = [
#     dict(type='LinearLR', start_factor=0.001, by_epoch=True, begin=0, end=10),
#     dict(type='PolyLR', eta_min=1e-6, power=0.9, by_epoch=True,
#          begin=10, end=200),
# ]

default_hooks = dict(
    checkpoint=dict(type='CheckpointHook', by_epoch=True, interval=10,
                    save_best='mIoU'),
)

model = dict(
    data_preprocessor=data_preprocessor,
)
