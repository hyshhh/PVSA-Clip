# model settings - CLIP-enhanced PVSA-Net for water segmentation
norm_cfg = dict(type='SyncBN', requires_grad=True)
model = dict(
    type='CLIPEncoderDecoder',
    pretrained=None,
    backbone=dict(
        type='BiFormer_fusion',
        embed_dim=[64, 128, 256, 512],
        depth=[3, 4, 6, 3],
        mlp_ratios=[3, 3, 3, 3],
        n_win=7,
        kv_downsample_mode='identity',
        kv_per_wins=[-1, -1, -1, -1],
        topks=[16, 12, 8, 6],
        topp_route_configs={
            16: dict(maxk=25, p=0.2, temperature=0.0175, energy=4.0),
            12: dict(maxk=18, p=0.4, temperature=0.025, energy=1.5),
            8: dict(maxk=36, p=0.6, temperature=0.05, energy=0.75),
            6: dict(maxk=49, p=0.8, temperature=0.15, energy=0.4),
        },
        debug_route=False,
        side_dwconv=5,
        before_attn_dwconv=3,
        layer_scale_init_value=-1,
        qk_dims=[64, 128, 256, 512],
        head_dim=32,
        param_routing=False, diff_routing=False, soft_routing=True,
        pre_norm=True,
        pe=None,
        auto_pad=True,
        fam_reduction=4,
        # Remove CNN branch (pure Transformer path)
        remove_cnn_branch=True,
        # Soft KV routing weight (0=pure gather, 1=original soft routing)
        soft_kv_weight=0.5,
        # TTRM: Text-guided Top-P Routing Module
        use_ttrm=True,
        ttrm_stages=[0, 1, 2, 3],
        # CPFM: Category-aware Prompt Fusion Module (training only)
        cpfm_config=dict(
            embed_dim=512,
            num_heads=8,
            top_m=8,
            cpfm_stages=[2, 3]),
        # CUDA inference backend
        topp_flash_backend=None,
        topp_flash_block_windows=64,
        topp_flash_debug=False,
        feature_vis_config=dict(enabled=False),
        attn_vis_config=dict(enabled=False),
    ),
    decode_head=dict(
        type='CLIPSegHead',
        in_channels=[64, 128, 256, 512],
        in_index=[0, 1, 2, 3],
        channels=256,
        embed_dim=512,
        dropout_ratio=0.1,
        num_classes=3,
        norm_cfg=norm_cfg,
        align_corners=False,
        loss_decode=dict(
            type='CrossEntropyLoss', use_sigmoid=False, loss_weight=1.0)),
    # CLIP Text Encoder config (for training)
    text_encoder=dict(
        embed_dim=512,
        num_categories=3,
        prompts_per_category=10,
        prompt_bank_path='tools/prompt_bank_water.pt'),
    train_cfg=dict(),
    test_cfg=dict(mode='whole')
)
