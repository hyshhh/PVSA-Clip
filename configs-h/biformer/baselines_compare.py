"""
对比实验配置文件 — 与 PVSA-Net (BiFormer_fusion_baseline) 对比的基线模型

使用方法：取消注释你想用的模型块，注释掉其余的。然后运行：
  CUDA_VISIBLE_DEVICES=0 python tools/train.py configs-h/biformer/baselines_compare.py --work-dir work_dirs/<model_name>

测试：
  CUDA_VISIBLE_DEVICES=0 python tools/test.py configs-h/biformer/baselines_compare.py work_dirs/<model_name>/best_mIoU.pth
"""

_base_ = [
    '../_base_/datasets/gqy.py',
    '../_base_/default_runtime.py',
    '../_base_/schedules/schedule_20k.py',
]

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

# ======================== 公共训练设置 ========================
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

default_hooks = dict(
    checkpoint=dict(type='CheckpointHook', by_epoch=True, interval=10,
                    save_best='mIoU'),
)

# ============================================================
#  模型选择：取消注释一个，注释掉其余的
# ============================================================

# ---- 1. DeepLabV3+ (ResNet-50) ----
norm_cfg = dict(type='SyncBN', requires_grad=True)
model = dict(
    type='EncoderDecoder',
    data_preprocessor=data_preprocessor,
    pretrained='open-mmlab://resnet50_v1c',
    backbone=dict(
        type='ResNet',
        depth=50,
        num_stages=4,
        out_indices=(0, 1, 2, 3),
        dilations=(1, 1, 2, 4),
        strides=(1, 1, 1, 1),
        norm_cfg=norm_cfg,
        norm_eval=False,
        style='pytorch',
        contract_dilation=True),
    decode_head=dict(
        type='DepthwiseSeparableASPPHead',
        in_channels=2048,
        in_index=3,
        channels=512,
        dilations=(1, 12, 24, 36),
        c1_in_channels=256,
        c1_channels=48,
        dropout_ratio=0.1,
        num_classes=3,
        norm_cfg=norm_cfg,
        align_corners=False,
        loss_decode=dict(
            type='CrossEntropyLoss', use_sigmoid=False, loss_weight=1.0)),
    auxiliary_head=dict(
        type='FCNHead',
        in_channels=1024,
        in_index=2,
        channels=256,
        num_convs=1,
        concat_input=False,
        dropout_ratio=0.1,
        num_classes=3,
        norm_cfg=norm_cfg,
        align_corners=False,
        loss_decode=dict(
            type='CrossEntropyLoss', use_sigmoid=False, loss_weight=0.4)),
    test_cfg=dict(mode='whole'),
)
# DeepLabV3+ 学习率：ResNet backbone 通常用更大 lr
optim_wrapper = dict(
    _delete_=True,
    type='OptimWrapper',
    optimizer=dict(type='SGD', lr=0.01, momentum=0.9, weight_decay=0.0005),
    paramwise_cfg=dict(
        custom_keys={
            'norm': dict(decay_mult=0.0),
            'head': dict(lr_mult=10.0),
        }),
)
param_scheduler = [
    dict(type='LinearLR', start_factor=0.001, by_epoch=True, begin=0, end=10),
    dict(type='PolyLR', eta_min=1e-6, power=0.9, by_epoch=True,
         begin=10, end=200),
]


# ---- 2. Swin-T + UPerNet ----
# norm_cfg = dict(type='SyncBN', requires_grad=True)
# model = dict(
#     type='EncoderDecoder',
#     data_preprocessor=data_preprocessor,
#     pretrained=None,
#     backbone=dict(
#         type='SwinTransformer',
#         pretrain_img_size=224,
#         embed_dims=96,
#         patch_size=4,
#         window_size=7,
#         mlp_ratio=4,
#         depths=[2, 2, 6, 2],
#         num_heads=[3, 6, 12, 24],
#         strides=(4, 2, 2, 2),
#         out_indices=(0, 1, 2, 3),
#         qkv_bias=True,
#         qk_scale=None,
#         drop_rate=0.,
#         attn_drop_rate=0.,
#         drop_path_rate=0.3,
#         patch_norm=True,
#         out_after_grid=True,
#         use_checkpoint=False),
#     decode_head=dict(
#         type='UPerHead',
#         in_channels=[96, 192, 384, 768],
#         in_index=[0, 1, 2, 3],
#         pool_scales=(1, 2, 3, 6),
#         channels=512,
#         dropout_ratio=0.1,
#         num_classes=3,
#         norm_cfg=norm_cfg,
#         align_corners=False,
#         loss_decode=dict(
#             type='CrossEntropyLoss', use_sigmoid=False, loss_weight=1.0)),
#     auxiliary_head=dict(
#         type='FCNHead',
#         in_channels=384,
#         in_index=2,
#         channels=256,
#         num_convs=1,
#         concat_input=False,
#         dropout_ratio=0.1,
#         num_classes=3,
#         norm_cfg=norm_cfg,
#         align_corners=False,
#         loss_decode=dict(
#             type='CrossEntropyLoss', use_sigmoid=False, loss_weight=0.4)),
#     test_cfg=dict(mode='whole'),
# )
# optim_wrapper = dict(
#     _delete_=True,
#     type='OptimWrapper',
#     optimizer=dict(type='AdamW', lr=6e-4, betas=(0.9, 0.999),
#                    weight_decay=0.01),
#     paramwise_cfg=dict(
#         custom_keys={
#             'norm': dict(decay_mult=0.0),
#             'head': dict(lr_mult=10.0),
#         }),
# )
# param_scheduler = [
#     dict(type='LinearLR', start_factor=0.001, by_epoch=True, begin=0, end=10),
#     dict(type='PolyLR', eta_min=1e-6, power=1.0, by_epoch=True,
#          begin=10, end=200),
# ]


# ---- 3. SegFormer (MiT-B2) ----
# norm_cfg = dict(type='SyncBN', requires_grad=True)
# model = dict(
#     type='EncoderDecoder',
#     data_preprocessor=data_preprocessor,
#     pretrained=None,
#     backbone=dict(
#         type='MixVisionTransformer',
#         in_channels=3,
#         embed_dims=[64, 128, 320, 512],
#         num_heads=[1, 2, 5, 8],
#         depths=[3, 4, 6, 3],
#         patch_sizes=[7, 3, 3, 3],
#         strides=[4, 2, 2, 2],
#         out_indices=(0, 1, 2, 3),
#         sr_ratios=[8, 4, 2, 1],
#         mlp_ratio=4,
#         qkv_bias=True,
#         drop_rate=0.,
#         attn_drop_rate=0.,
#         drop_path_rate=0.1),
#     decode_head=dict(
#         type='SegformerHead',
#         in_channels=[64, 128, 320, 512],
#         in_index=[0, 1, 2, 3],
#         channels=256,
#         dropout_ratio=0.1,
#         num_classes=3,
#         norm_cfg=norm_cfg,
#         align_corners=False,
#         loss_decode=dict(
#             type='CrossEntropyLoss', use_sigmoid=False, loss_weight=1.0)),
#     test_cfg=dict(mode='whole'),
# )
# optim_wrapper = dict(
#     _delete_=True,
#     type='OptimWrapper',
#     optimizer=dict(type='AdamW', lr=6e-4, betas=(0.9, 0.999),
#                    weight_decay=0.01),
#     paramwise_cfg=dict(
#         custom_keys={
#             'norm': dict(decay_mult=0.0),
#             'head': dict(lr_mult=10.0),
#         }),
# )
# param_scheduler = [
#     dict(type='LinearLR', start_factor=0.001, by_epoch=True, begin=0, end=10),
#     dict(type='PolyLR', eta_min=1e-6, power=1.0, by_epoch=True,
#          begin=10, end=200),
# ]
