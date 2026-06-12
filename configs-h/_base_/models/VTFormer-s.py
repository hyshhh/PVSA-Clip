# model settings
norm_cfg = dict(type='SyncBN', requires_grad=True)
# data_preprocessor = dict(
#     type='SegDataPreProcessor',
#     mean=[123.675, 116.28, 103.53],
#     std=[58.395, 57.12, 57.375],
#     bgr_to_rgb=True,
#     pad_val=0,
#     seg_pad_val=255)
model = dict(
    type='EncoderDecoder',
    pretrained=None,
    backbone=dict(
        type='BiFormer_fusion',
        embed_dim=[64, 128, 256, 512],     # BiFormer的通道配置
        depth=[3, 4, 6, 3],               # 每个stage的block数量
        # depth=[1, 3, 4, 2],               # 每个stage的block数量
        mlp_ratios=[3, 3, 3, 3],
        # ------------------------------
        n_win=7,
        kv_downsample_mode='identity',
        kv_per_wins=[-1, -1, -1, -1],
        topks=[1, 4, 16, -2],
        side_dwconv=5,
        before_attn_dwconv=3,
        layer_scale_init_value=-1,
        qk_dims=[64, 128, 256, 512],
        head_dim=32,
        param_routing=False, diff_routing=False, soft_routing=False,
        pre_norm=True,
        pe=None,
        auto_pad=True,
        # Top-p attention backend switch.
        # 1) Original path: use_topp_flash=False. This keeps the old
        #    kv_gather -> attention implementation and uses the most memory.
        # 2) PyTorch sparse path: set use_topp_flash=True and
        #    topp_flash_backend='torch_block'. This avoids materializing the
        #    full kv_gather tensor by processing routed windows in PyTorch
        #    blocks. You may tune topp_flash_block_windows for memory/speed.
        # 3) Custom CUDA path: set use_topp_flash=True and
        #    topp_flash_backend='cuda'. The CUDA kernel uses keep_len instead
        #    of r_mask internally and should be used on a server with a valid
        #    CUDA extension build toolchain. Set PVSA_TOPP_FLASH_STRICT_CUDA=1
        #    on the server if you want failures instead of fallback.
        use_topp_flash=True,
        topp_flash_backend='torch_block',
        topp_flash_block_windows=16,
        use_pruned_kv_gather=False
    ),
    # decode_head=dict(
    #     type='UPerHead',
    #     in_channels=[64, 128, 256, 512],   # 对应backbone输出通道
    #     in_index=[0, 1, 2, 3],
    #     pool_scales=(1, 2, 3, 6),
    #     channels=512,
    #     dropout_ratio=0.1,
    #     num_classes=2,                    # 类别数（根据数据集修改）
    #     norm_cfg=norm_cfg,
    #     align_corners=False,
    #     loss_decode=dict(
    #         type='CrossEntropyLoss', use_sigmoid=False, loss_weight=1.0)
    # ),
    decode_head=dict(
        type='SegformerHead',
        in_channels=[64, 128, 256, 512],
        in_index=[0, 1, 2, 3],
        channels=256,
        dropout_ratio=0.1,
        num_classes=19,
        norm_cfg=norm_cfg,
        align_corners=False,
        loss_decode=dict(
            type='CrossEntropyLoss', use_sigmoid=False, loss_weight=1.0)),
    # 模型训练与推理配置
    train_cfg=dict(),
    test_cfg=dict(mode='whole')
)
