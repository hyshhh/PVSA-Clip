# model settings - CLIP-enhanced PVSA-Net for water segmentation
norm_cfg = dict(type='SyncBN', requires_grad=True)
use_clip_decode_head = True

clip_decode_head = dict(
    type='CLIPSegHead',
    in_channels=[64, 128, 256, 512],
    in_index=[0, 1, 2, 3],
    channels=256,
    embed_dim=512,
    normalize_visual=False,  # False=默认点积 | True=严格余弦（通道维 L2 归一化）
    dropout_ratio=0.1,
    num_classes=3,
    norm_cfg=norm_cfg,
    align_corners=False,
    loss_decode=dict(
        type='CrossEntropyLoss', use_sigmoid=False, loss_weight=1.0))

# 消融用普通 seg head：仅替换解码头，保留 CLIP 文本路径与 backbone 文本注入。
seg_decode_head = dict(
    type='SegformerHead',
    in_channels=[64, 128, 256, 512],
    in_index=[0, 1, 2, 3],
    channels=256,
    dropout_ratio=0.1,
    num_classes=3,
    norm_cfg=norm_cfg,
    align_corners=False,
    loss_decode=dict(
        type='CrossEntropyLoss', use_sigmoid=False, loss_weight=1.0))

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
        topks=[16, 12, 8, 6],
        topp_route_configs={
            16: dict(maxk=5, mink=1, p=0.2, temperature=0.5, energy=3.0),
            12: dict(maxk=10, mink=3, p=0.6, temperature=4, energy=6.0),
            8: dict(maxk=25, mink=5, p=0.6, temperature=8, energy=12.0),
        },
        side_dwconv=5,
        before_attn_dwconv=3,
        qk_dims=[64, 128, 256, 512],
        head_dim=32,
        param_routing=False,
        diff_routing=False,
        soft_routing=False,
        pre_norm=True,
        auto_pad=True,
        remove_cnn_branch=True,
        # TTRM: routing-level text injection (Stage 0-2 only;
        # Stage 3 uses plain self-attention and has no TTRM)
        use_ttrm=True,
        ttrm_stages=[0, 1, 2],
        # Cross-attention: feature-level text injection (deep stages only)
        cross_attn_stages=[2, 3],
        # CUDA inference backend
        topp_flash_backend=None,
        use_route_mask=True,
        # 路由 token 池化方式: 'avg' | 'max' | 'avgmax'
        route_pooling='avgmax',
        # 最后一层 stage 用普通 self-attention 替代 ToppAttention
        use_plain_attn_last_stage=True,
    ),
    decode_head=clip_decode_head if use_clip_decode_head else seg_decode_head,
    text_encoder=dict(
        embed_dim=512,
        num_categories=3,
        prompts_per_category=10,
        prompt_bank_path='tools/prompt_bank_water.pt',
        use_reprta=True,                  # 是否启用 RepRTA 文本原型精炼
        reprta_ffn_type='swiglu',         # 'swiglu'(门控) | 'gelu'(普通 FFN)
        reprta_zero_init=True),           # w3 是否零初始化（保护 CLIP 原型）
    text_refiner=dict(in_dim=512, hidden_mult=4),
    image_query_proj=dict(stage_channels=[64, 128, 256, 512], hidden_dim=512),
    train_cfg=dict(),
    test_cfg=dict(mode='whole')
)
