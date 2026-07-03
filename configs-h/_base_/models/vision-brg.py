# 模型设置 - 标准 BiFormer Attention 消融配置
# 与 vision-topp.py 的唯一差异：backbone 由 BiFormer_fusion 换成 BiFormer_standalone
#   - ToppAttention（top-p 投票路由）→ BiLevelRoutingAttention（标准双层路由，原版 BiFormer）
#   - 末层 stage3 由普通 self-attention → AttentionLePE（topk=-2，带局部位置编码的自注意力）
# 切换由 topks 取值语义控制：topk>0 → BRG，-1→普通 Attn，-2→AttentionLePE，0→卷积
# 本配置不含任何文本信号（无 text_encoder / TTRM / cross-attn），与 vision-topp 形成纯注意力机制对照
norm_cfg = dict(type='SyncBN', requires_grad=True)

# 数据预处理（与 clip-baseline 一致）
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
        type='BiFormer_standalone',
        # depth / embed_dim 与 topp 版对齐
        depth=[3, 4, 6, 3],
        in_chans=3,
        embed_dim=[64, 128, 256, 512],
        head_dim=32,
        qk_dims=[64, 128, 256, 512],
        mlp_ratios=[3, 3, 3, 3],
        # 标准 BiFormer-S 路由超参：BRG + 末层 AttentionLePE
        n_win=7,
        kv_downsample_mode='identity',
        kv_per_wins=[-1, -1, -1, -1],
        topks=[1, 4, 16, -2],
        side_dwconv=5,
        before_attn_dwconv=3,
        layer_scale_init_value=-1,
        param_routing=False,
        diff_routing=False,
        soft_routing=False,
        pre_norm=True,
        pe=None,
        auto_pad=True,
        drop_path_rate=0.3,
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
