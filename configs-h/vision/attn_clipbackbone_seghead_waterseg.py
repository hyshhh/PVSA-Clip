_base_ = [
    # '../_base_/datasets/gqy.py',
    '../_base_/datasets/KAKA.py',
    '../_base_/default_runtime.py',
    '../_base_/schedules/schedule_20k.py'
]

import os as _os

attention_type = 'brg'
_config_dir = (
    _os.path.dirname(__file__)
    if '__file__' in globals() else '{{ fileDirname }}')
_model_base = _os.path.join(
    _config_dir, '../_base_/models/clip-topp.py')
with open(_model_base, 'r', encoding='utf-8') as _f:
    exec(compile(_f.read(), _model_base, 'exec'))
del _os, _config_dir, _model_base, _f

# 同骨干无文本公平基线：
# 使用 CLIP 路径同构的 BiFormer_fusion_clip 骨干，但不构建
# CLIPEncoderDecoder / TextEncoder / CLIPSegHeadV2。
# 用于区分“骨干融合结构和参数增加”与“文本原型分类”的贡献。

crop_size = (256, 256)
data_preprocessor = dict(
    type='SegDataPreProcessor',
    mean=[123.675, 116.28, 103.53],
    std=[58.395, 57.12, 57.375],
    bgr_to_rgb=True,
    pad_val=0,
    seg_pad_val=255,
    size=crop_size
)

model = dict(
    type='EncoderDecoder',
    pretrained=None,
    data_preprocessor=data_preprocessor,
    backbone=backbone,
    decode_head=seg_decode_head,
    train_cfg=dict(),
    test_cfg=dict(mode='whole'))

train_dataloader = dict(
    batch_size=16,
    num_workers=8,
    sampler=dict(type='DefaultSampler', shuffle=True)
)
val_dataloader = dict(batch_size=4, num_workers=2)
test_dataloader = dict(batch_size=1, num_workers=1)

param_scheduler = [
    dict(
        type='LinearLR',
        start_factor=0.001,
        by_epoch=True,
        begin=0,
        end=10
    ),
    dict(
        type='PolyLR',
        eta_min=1e-6,
        power=1.0,
        by_epoch=True,
        begin=10,
        end=200
    )
]

train_cfg = dict(
    type='EpochBasedTrainLoop',
    max_epochs=200,
    val_interval=10
)

optim_wrapper = dict(
    _delete_=True,
    type='OptimWrapper',
    optimizer=dict(type='AdamW', lr=6e-4, betas=(0.9, 0.999), weight_decay=0.01),
    paramwise_cfg=dict(
        custom_keys={
            'pos_block': dict(decay_mult=0.0),
            'norm': dict(decay_mult=0.0),
            'head': dict(lr_mult=10.0),
            'attn.router': dict(lr_mult=0.2, decay_mult=1.0),
        })
)

val_cfg = dict(type='ValLoop')
test_cfg = dict(type='TestLoop')

val_evaluator = dict(
    type='IoUMetric',
    iou_metrics=['mIoU', 'mDice'],
    ignore_index=255,
    classwise=True
)
test_evaluator = val_evaluator

default_hooks = dict(
    checkpoint=dict(type='CheckpointHook', by_epoch=True, interval=10, save_best='mIoU')
)
