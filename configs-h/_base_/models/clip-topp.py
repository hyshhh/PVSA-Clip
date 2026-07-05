# model settings - CLIP-enhanced PVSA-Net for water segmentation
norm_cfg = dict(type='SyncBN', requires_grad=True)

# 注意力主路径：
# 'topp' = ToppAttention(top-p 投票路由)
# 'brg'  = BiLevelRoutingAttention(原版双层路由) + 末层 AttentionLePE
attention_type = globals().get('attention_type', 'topp')
if attention_type not in ('topp', 'brg'):
    raise ValueError('attention_type must be "topp" or "brg"')

use_clip_decode_head = True
use_backbone_text_injection = False
clip_head_type = globals().get('clip_head_type', 'v2')
if clip_head_type not in ('v1', 'v2'):
    raise ValueError('clip_head_type must be "v1" or "v2"')

# 图相关 query 来源：
# 'backbone_pool'  = 池化骨干多 stage 特征（旧路径）
# 'decode_fusion'  = 池化 decode head 上采样拼接后的融合特征
image_query_source = 'decode_fusion'
# 图相关 query 输出头：
# 'joint'    = 旧路径：一个线性层一次输出 3*512
# 'separate' = 共享前层 + 每类独立线性输出头
image_query_head_type = 'separate'

# prompt bank 原始顺序为 water / ship / land。
# 这里记录“训练标签通道顺序 -> prompt 语义顺序”的映射：
#   KAKA: background / boat / free-space -> land / ship / water
#   gqy : water / ground / object       -> water / land / ship
#   GBA : object / water / ground       -> ship / water / land
prompt_category_orders = dict(
    kaka=['land', 'ship', 'water'],
    gqy=['water', 'land', 'ship'],
    gba=['ship', 'water', 'land'])
prompt_dataset = globals().get('prompt_dataset', 'kaka')
if prompt_dataset not in prompt_category_orders:
    raise ValueError(
        f'prompt_dataset must be one of {tuple(prompt_category_orders)}')
prompt_category_order = globals().get(
    'prompt_category_order', prompt_category_orders[prompt_dataset])

clip_decode_head_v1 = dict(
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

clip_decode_head_v2 = dict(
    type='CLIPSegHeadV2',
    in_channels=[64, 128, 256, 512],
    in_index=[0, 1, 2, 3],
    channels=256,
    embed_dim=512,
    visual_prompt_mode='class_activation',
    clip_logit_weight_init=0.1,
    text_delta_scale_init=0.1,
    base_loss_weight=0.4,
    clip_loss_weight=0.4,
    text_drift_loss_weight=0.02,
    dropout_ratio=0.1,
    num_classes=3,
    norm_cfg=norm_cfg,
    align_corners=False,
    loss_decode=dict(
        type='CrossEntropyLoss', use_sigmoid=False, loss_weight=1.0))

clip_decode_head = (
    clip_decode_head_v2 if clip_head_type == 'v2' else clip_decode_head_v1)

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

common_backbone = dict(
    embed_dim=[64, 128, 256, 512],
    depth=[3, 4, 6, 3],
    mlp_ratios=[3, 3, 3, 3],
    n_win=7,
    kv_downsample_mode='identity',
    side_dwconv=5,
    before_attn_dwconv=3,
    qk_dims=[64, 128, 256, 512],
    head_dim=32,
    param_routing=False,
    diff_routing=False,
    soft_routing=False,
    pre_norm=True,
    auto_pad=True,
    use_ttrm=use_backbone_text_injection,
    ttrm_stages=[0, 1, 2] if use_backbone_text_injection else [],
    cross_attn_stages=[2, 3] if use_backbone_text_injection else [],
    use_plain_attn_last_stage=True,
)

if attention_type == 'topp':
    backbone = dict(
        type='BiFormer_fusion',
        **common_backbone,
        topks=[16, 12, 8, 6],
        topp_route_configs={
            16: dict(maxk=5, mink=1, p=0.2, temperature=0.5, energy=3.0),
            12: dict(maxk=10, mink=3, p=0.6, temperature=4, energy=6.0),
            8: dict(maxk=25, mink=5, p=0.6, temperature=8, energy=12.0),
        },
        remove_cnn_branch=True,
        topp_flash_backend=None,
        use_route_mask=True,
        route_pooling='avgmax')
else:
    backbone = dict(
        type='BiFormer_fusion_clip',
        **common_backbone,
        kv_per_wins=[-1, -1, -1, -1],
        # 标准 BiFormer-S 路由：前 3 层 BRG，末层 AttentionLePE
        topks=[1, 4, 16, -2],
        layer_scale_init_value=-1,
        pe=None,
        drop_path_rate=0.3)

model = dict(
    type='CLIPEncoderDecoder',
    pretrained=None,
    use_backbone_text_injection=use_backbone_text_injection,
    backbone=backbone,
    decode_head=clip_decode_head if use_clip_decode_head else seg_decode_head,
    text_encoder=dict(
        embed_dim=512,
        num_categories=3,
        prompts_per_category=10,
        prompt_bank_path='tools/prompt_bank_water.pt',
        prompt_category_order=prompt_category_order,
        use_reprta=True,                  # 是否启用 RepRTA 文本原型精炼
        reprta_ffn_type='swiglu',         # 'swiglu'(门控) | 'gelu'(普通 FFN)
        reprta_zero_init=True),           # w3 是否零初始化（保护 CLIP 原型）
    text_refiner=(
        dict(in_dim=512, hidden_mult=4)
        if use_backbone_text_injection else None),
    image_query_proj=dict(
        source=image_query_source,
        query_head_type=image_query_head_type,
        stage_channels=[64, 128, 256, 512],
        in_dim=256,
        hidden_dim=512),
    train_cfg=dict(),
    test_cfg=dict(mode='whole')
)
