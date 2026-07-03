# 模型设置 - CLIP 版 backbone 的纯视觉 baseline（移除全部文本相关配置）
# 与 clip-topp.py 的差异：
#   1. 不使用 CLIPEncoderDecoder，退回 EncoderDecoder（无 text_encoder / 文本原型注入）
#   2. Backbone 仍复用 BiFormer_fusion，但关闭文本通道：
#      - use_ttrm=False / ttrm_stages=[]   路由级文本注入关闭
#      - cross_attn_stages=[]              特征级文本注入关闭（不构造 TextCrossAttention）
#   3. decode_head 由 CLIPSegHead 退回 SegformerHead（无 embed_dim 文本对齐）
#   4. block 切换规则沿用 clip 版：use_plain_attn_last_stage=True
#      —— stage1-3 含 stage0-2 用 ToppAttention，最后一层 stage3 用普通 self-attention
# 不需要修改任何 .py 代码，纯配置层即可生效。
norm_cfg = dict(type='SyncBN', requires_grad=True)

# 数据预处理搬入 base model（与 clip 版训练配置一致）
data_preprocessor = dict(
    type='SegDataPreProcessor',
    mean=[123.675, 116.28, 103.53],
    std=[58.395, 57.12, 57.375],
    bgr_to_rgb=True,
    pad_val=0,
    seg_pad_val=255,
    size=(256, 256),
)

model = dict(
    type='EncoderDecoder',
    pretrained=None,
    data_preprocessor=data_preprocessor,
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
        # === 文本通道全部关闭 ===
        use_ttrm=False,
        ttrm_stages=[],
        cross_attn_stages=[],
        # === CUDA 推理后端 ===
        topp_flash_backend=None,
        use_route_mask=True,
        # 路由 token 池化方式: 'avg' | 'max' | 'avgmax'
        route_pooling='avgmax',
        # 最后一层 stage 用普通 self-attention 替代 ToppAttention
        use_plain_attn_last_stage=True,
    ),
    decode_head=dict(
        type='SegformerHead',
        in_channels=[64, 128, 256, 512],
        in_index=[0, 1, 2, 3],
        channels=256,
        dropout_ratio=0.1,
        num_classes=3,
        norm_cfg=norm_cfg,
        align_corners=False,
        loss_decode=dict(
            type='CrossEntropyLoss', use_sigmoid=False, loss_weight=1.0)),
    train_cfg=dict(),
    test_cfg=dict(mode='whole')
)
