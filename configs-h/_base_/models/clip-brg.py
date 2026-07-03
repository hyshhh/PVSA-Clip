# 模型设置 - 标准 BiFormer Attention 消融（保留完整 CLIP 文本路径）
# 与 clip-topp.py 的唯一差异：backbone 注意力主路径
#   ToppAttention(top-p 投票路由)  →  BiLevelRoutingAttention(原版双层路由) + 末层 AttentionLePE
# 文本处理保持同款：CLIPEncoderDecoder / text_encoder / TTRM(Stage 0-2 路由级注入) /
# Cross-Attn(Stage 2-3 特征级注入) / CLIPSegHead 保留
# backbone 由 BiFormer_fusion 换成 BiFormer_fusion_clip（同款跨层融合，注意力改用标准 BRG）
norm_cfg = dict(type='SyncBN', requires_grad=True)

model = dict(
    type='CLIPEncoderDecoder',
    pretrained=None,
    backbone=dict(
        type='BiFormer_fusion_clip',
        embed_dim=[64, 128, 256, 512],
        depth=[3, 4, 6, 3],
        mlp_ratios=[3, 3, 3, 3],
        n_win=7,
        kv_downsample_mode='identity',
        kv_per_wins=[-1, -1, -1, -1],
        # 标准 BiFormer-S 路由：前 3 层 BRG，末层 AttentionLePE
        # （use_plain_attn_last_stage 会把末层 topk>0 强制改为 -2）
        topks=[1, 4, 16, -2],
        side_dwconv=5,
        before_attn_dwconv=3,
        layer_scale_init_value=-1,
        qk_dims=[64, 128, 256, 512],
        head_dim=32,
        param_routing=False,
        diff_routing=False,
        soft_routing=False,
        pre_norm=True,
        pe=None,
        auto_pad=True,
        drop_path_rate=0.3,
        # === 文本路径：TTRM 到 Stage 0-2，Cross-Attn 到 Stage 2-3 ===
        use_ttrm=True,
        ttrm_stages=[0, 1, 2],
        cross_attn_stages=[2, 3],
        # 末层 stage 用 plain AttentionLePE（与 topp 版 use_plain_attn_last_stage 对齐）
        use_plain_attn_last_stage=True,
    ),
    decode_head=dict(
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
            type='CrossEntropyLoss', use_sigmoid=False, loss_weight=1.0)),
    text_encoder=dict(
        embed_dim=512,
        num_categories=3,
        prompts_per_category=10,
        prompt_bank_path='tools/prompt_bank_water.pt',
        use_reprta=True,                  # 是否启用 RepRTA 文本原型精炼
        reprta_ffn_type='swiglu',         # 'swiglu'(门控) | 'gelu'(普通 FFN)
        reprta_zero_init=True),           # w3 是否零初始化（保护 CLIP 原型）
    train_cfg=dict(),
    test_cfg=dict(mode='whole')
)
