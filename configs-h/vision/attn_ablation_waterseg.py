_base_ = [
    '../_base_/models/vision-brg.py',
    # '../_base_/datasets/gqy.py',
    '../_base_/datasets/KAKA.py',
    '../_base_/default_runtime.py',
    '../_base_/schedules/schedule_20k.py'
]

# 标准 BiFormer Attention 消融训练（gqy 水面数据集, 3 类, 200 epochs）
# 对照 clip/waterseg 的纯视觉版：唯一变量为注意力机制
#   ToppAttention(top-p 投票路由)  vs  BiLevelRoutingAttention(原版双层路由) + 末层 AttentionLePE
# CUDA_VISIBLE_DEVICES=0 python tools/train.py configs-h/vision/attn_ablation_waterseg.py --work-dir work_dirs/biformer_attn

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
            # 路由器降学习率（BRG 内部也有 router）
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
