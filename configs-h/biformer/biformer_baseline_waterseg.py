_base_ = [
    '../_base_/models/VTFormer-s-baseline.py',
    '../_base_/datasets/gqy.py',
    '../_base_/default_runtime.py',
    '../_base_/schedules/schedule_20k.py'
]

# Baseline PVSA-Net training (gqy/KAKA dataset, 200 epochs)
# CUDA_VISIBLE_DEVICES=0 python tools/train.py configs-h/biformer/biformer_baseline_waterseg.py --work-dir work_dirs/baseline

train_dataloader = dict(
    batch_size=16,
    num_workers=8,
    sampler=dict(type='DefaultSampler', shuffle=True)
)
val_dataloader = dict(batch_size=4, num_workers=2)
test_dataloader = dict(batch_size=1, num_workers=1)

param_scheduler = [
    dict(type='LinearLR', start_factor=0.001, by_epoch=True, begin=0, end=10),
    dict(type='PolyLR', eta_min=1e-6, power=1.0, by_epoch=True, begin=10, end=200)
]
train_cfg = dict(type='EpochBasedTrainLoop', max_epochs=200, val_interval=10)

optim_wrapper = dict(
    _delete_=True,
    type='OptimWrapper',
    optimizer=dict(type='AdamW', lr=6e-4, betas=(0.9, 0.999), weight_decay=0.01),
    paramwise_cfg=dict(
        custom_keys={
            'pos_block': dict(decay_mult=0.0),
            'norm': dict(decay_mult=0.0),
            'head': dict(lr_mult=10.0),
            # 路由器降学习率：验证「路由器是否训练过猛」
            'PA.router': dict(lr_mult=0.2, decay_mult=1.0),
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

model = dict(
    test_cfg=dict(mode='whole')
)

default_hooks = dict(
    checkpoint=dict(type='CheckpointHook', by_epoch=True, interval=10, save_best='mIoU')
)
