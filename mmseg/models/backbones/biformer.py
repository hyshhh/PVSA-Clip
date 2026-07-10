"""
Standalone BiFormer backbone for comparison experiments.
Adapted from https://github.com/rayleizhu/BiFormer (mmseg 2.x compatible).

Variants:
  - BiFormer_S: embed_dim=[64,128,256,512], depth=[4,4,18,4]
  - BiFormer_B: embed_dim=[96,192,384,768], depth=[4,4,18,4]
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from mmengine.runner import load_checkpoint
from timm.models.layers import DropPath, trunc_normal_
from ..utils.bra_legacy import BiLevelRoutingAttention
from ..utils.common import Attention, AttentionLePE, DWConv
from mmseg.registry import MODELS


def get_pe_layer(emb_dim, pe_dim=None, name='none'):
    if name == 'none':
        return nn.Identity()
    raise ValueError(f'PE name {name} is not supported!')


class Block(nn.Module):
    def __init__(self, dim, drop_path=0., layer_scale_init_value=-1,
                 num_heads=8, n_win=7, qk_dim=None, qk_scale=None,
                 kv_per_win=4, kv_downsample_ratio=4, kv_downsample_kernel=None,
                 kv_downsample_mode='ada_avgpool',
                 topk=4, param_attention="qkvo", param_routing=False,
                 diff_routing=False, soft_routing=False, mlp_ratio=4,
                 mlp_dwconv=False, side_dwconv=5, before_attn_dwconv=3,
                 pre_norm=True, auto_pad=False,
                 use_ttrm=False, text_dim=512, cross_attn_module=None,
                 use_plain_attn=False):
        super().__init__()
        qk_dim = qk_dim or dim
        self._use_plain_attn = use_plain_attn

        if before_attn_dwconv > 0:
            self.pos_embed = nn.Conv2d(dim, dim, kernel_size=before_attn_dwconv,
                                       padding=1, groups=dim)
        else:
            self.pos_embed = lambda x: 0
        self.norm1 = nn.LayerNorm(dim, eps=1e-6)
        # 仅 BiLevelRoutingAttention 接受 category_prototypes（TTRM 注入路由）；
        # Attention / AttentionLePE / 卷积分支 forward 仅 (x)，不透传 prototypes
        if topk > 0:
            self.attn = BiLevelRoutingAttention(
                dim=dim, num_heads=num_heads, n_win=n_win, qk_dim=qk_dim,
                qk_scale=qk_scale, kv_per_win=kv_per_win,
                kv_downsample_ratio=kv_downsample_ratio,
                kv_downsample_kernel=kv_downsample_kernel,
                kv_downsample_mode=kv_downsample_mode,
                topk=topk, param_attention=param_attention,
                param_routing=param_routing, diff_routing=diff_routing,
                soft_routing=soft_routing, side_dwconv=side_dwconv,
                auto_pad=auto_pad,
                use_ttrm=use_ttrm,
                text_dim=text_dim)
            self._attn_takes_prototypes = True
        elif topk == -1:
            self.attn = Attention(dim=dim)
            self._attn_takes_prototypes = False
        elif topk == -2:
            self.attn = AttentionLePE(dim=dim, side_dwconv=side_dwconv)
            self._attn_takes_prototypes = False
        elif topk == 0:
            from einops.layers.torch import Rearrange
            self.attn = nn.Sequential(
                Rearrange('n h w c -> n c h w'),
                nn.Conv2d(dim, dim, 1),
                nn.Conv2d(dim, dim, 5, padding=2, groups=dim),
                nn.Conv2d(dim, dim, 1),
                Rearrange('n c h w -> n h w c'))
            self._attn_takes_prototypes = False

        # Cross-attention module（与 bi_topp_vote.Block 一致，仅在配置的 stage 构造）
        self.cross_attn = cross_attn_module

        self.norm2 = nn.LayerNorm(dim, eps=1e-6)
        self.mlp = nn.Sequential(
            nn.Linear(dim, int(mlp_ratio * dim)),
            DWConv(int(mlp_ratio * dim)) if mlp_dwconv else nn.Identity(),
            nn.GELU(),
            nn.Linear(int(mlp_ratio * dim), dim))
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()

        if layer_scale_init_value > 0:
            self.use_layer_scale = True
            self.gamma1 = nn.Parameter(
                layer_scale_init_value * torch.ones((dim)), requires_grad=True)
            self.gamma2 = nn.Parameter(
                layer_scale_init_value * torch.ones((dim)), requires_grad=True)
        else:
            self.use_layer_scale = False
        self.pre_norm = pre_norm

    def forward(self, x, category_prototypes=None):
        x = x + self.pos_embed(x)
        x = x.permute(0, 2, 3, 1)  # NCHW -> NHWC

        def call_attn(inp):
            if self._attn_takes_prototypes:
                return self.attn(inp, category_prototypes=category_prototypes)
            return self.attn(inp)

        if self.pre_norm:
            if self.use_layer_scale:
                x = x + self.drop_path(self.gamma1 * call_attn(self.norm1(x)))
                if self.cross_attn is not None and category_prototypes is not None:
                    x = self.cross_attn(x, category_prototypes)
                x = x + self.drop_path(self.gamma2 * self.mlp(self.norm2(x)))
            else:
                x = x + self.drop_path(call_attn(self.norm1(x)))
                if self.cross_attn is not None and category_prototypes is not None:
                    x = self.cross_attn(x, category_prototypes)
                x = x + self.drop_path(self.mlp(self.norm2(x)))
        else:
            if self.use_layer_scale:
                x = self.norm1(x + self.drop_path(self.gamma1 * call_attn(x)))
                if self.cross_attn is not None and category_prototypes is not None:
                    x = self.cross_attn(x, category_prototypes)
                x = self.norm2(x + self.drop_path(self.gamma2 * self.mlp(x)))
            else:
                x = self.norm1(x + self.drop_path(call_attn(x)))
                if self.cross_attn is not None and category_prototypes is not None:
                    x = self.cross_attn(x, category_prototypes)
                x = self.norm2(x + self.drop_path(self.mlp(x)))

        x = x.permute(0, 3, 1, 2)  # NHWC -> NCHW
        return x


class BiFormer(nn.Module):
    """BiFormer backbone (classification version, head removed for seg)."""

    def __init__(self, depth=[3, 4, 8, 3], in_chans=3, embed_dim=[64, 128, 320, 512],
                 head_dim=64, qk_scale=None, drop_path_rate=0.,
                 use_checkpoint_stages=[], n_win=7,
                 kv_downsample_mode='ada_avgpool',
                 kv_per_wins=[2, 2, -1, -1], topks=[8, 8, -1, -1],
                 side_dwconv=5, layer_scale_init_value=-1,
                 qk_dims=[None, None, None, None],
                 param_routing=False, diff_routing=False, soft_routing=False,
                 pre_norm=True, pe=None, pe_stages=[0],
                 before_attn_dwconv=3, auto_pad=False,
                 kv_downsample_kernels=[4, 2, 1, 1],
                 kv_downsample_ratios=[4, 2, 1, 1],
                 mlp_ratios=[4, 4, 4, 4],
                 param_attention='qkvo', mlp_dwconv=False,
                 # 文本路径默认全关，让 BiFormer_standalone（纯视觉 baseline）行为等价于改动前；
                 # clip 消融配置会显式开启这些参数
                 use_ttrm=False, ttrm_stages=[],
                 cross_attn_stages=[],
                 text_dim=512,
                 use_plain_attn_last_stage=False):
        super().__init__()
        self.embed_dim = embed_dim
        self.use_ttrm = use_ttrm
        self.ttrm_stages = ttrm_stages
        self.cross_attn_stages = cross_attn_stages
        self.use_plain_attn_last_stage = use_plain_attn_last_stage

        # ---- patch embedding (stem + 3 downsample layers) ----
        self.downsample_layers = nn.ModuleList()
        stem = nn.Sequential(
            nn.Conv2d(in_chans, embed_dim[0] // 2, 3, stride=2, padding=1),
            nn.BatchNorm2d(embed_dim[0] // 2),
            nn.GELU(),
            nn.Conv2d(embed_dim[0] // 2, embed_dim[0], 3, stride=2, padding=1),
            nn.BatchNorm2d(embed_dim[0]),
        )
        if pe is not None and 0 in pe_stages:
            stem.append(get_pe_layer(emb_dim=embed_dim[0], name=pe))
        self.downsample_layers.append(stem)

        for i in range(3):
            downsample_layer = nn.Sequential(
                nn.Conv2d(embed_dim[i], embed_dim[i + 1], 3, stride=2, padding=1),
                nn.BatchNorm2d(embed_dim[i + 1]))
            if pe is not None and i + 1 in pe_stages:
                downsample_layer.append(
                    get_pe_layer(emb_dim=embed_dim[i + 1], name=pe))
            self.downsample_layers.append(downsample_layer)

        # ---- stages ----
        self.stages = nn.ModuleList()
        nheads = [dim // head_dim for dim in qk_dims]
        dp_rates = [x.item() for x in torch.linspace(0, drop_path_rate, sum(depth))]
        cur = 0

        from ..utils.text_cross_attn import TextCrossAttention

        for i in range(4):
            use_ca = i in cross_attn_stages
            use_ttrm_stage = (use_ttrm and i in ttrm_stages)
            # 末层 stage 用 plain Attention（而非 BRG），与 topp 版 use_plain_attn_last_stage 对齐
            topk_i = topks[i]
            if self.use_plain_attn_last_stage and i == 3 and topk_i > 0:
                # 把末层强制切到标准 AttentionLePE（topk=-2），保留 BiFormer 末层语义
                topk_i = -2
            stage = nn.Sequential(
                *[Block(
                    dim=embed_dim[i], drop_path=dp_rates[cur + j],
                    layer_scale_init_value=layer_scale_init_value,
                    topk=topk_i, num_heads=nheads[i], n_win=n_win,
                    qk_dim=qk_dims[i], qk_scale=qk_scale,
                    kv_per_win=kv_per_wins[i],
                    kv_downsample_ratio=kv_downsample_ratios[i],
                    kv_downsample_kernel=kv_downsample_kernels[i],
                    kv_downsample_mode=kv_downsample_mode,
                    param_attention=param_attention,
                    param_routing=param_routing,
                    diff_routing=diff_routing, soft_routing=soft_routing,
                    mlp_ratio=mlp_ratios[i], mlp_dwconv=mlp_dwconv,
                    side_dwconv=side_dwconv,
                    before_attn_dwconv=before_attn_dwconv,
                    pre_norm=pre_norm, auto_pad=auto_pad,
                    use_ttrm=use_ttrm_stage, text_dim=text_dim,
                    cross_attn_module=TextCrossAttention(
                        visual_dim=embed_dim[i], text_dim=text_dim,
                        num_heads=nheads[i]) if use_ca else None)
                  for j in range(depth[i])])
            self.stages.append(stage)
            cur += depth[i]

        # ---- extra norms for dense prediction ----
        from timm.models.layers import LayerNorm2d
        self.extra_norms = nn.ModuleList()
        for i in range(4):
            self.extra_norms.append(LayerNorm2d(embed_dim[i]))

        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    def init_weights(self, pretrained=None):
        if isinstance(pretrained, str):
            load_checkpoint(self, pretrained, strict=False)

    def forward_features(self, x, category_prototypes=None):
        out = []
        for i in range(4):
            x = self.downsample_layers[i](x)
            # 手动展开 stage，向每个 Block 透传 category_prototypes
            for block in self.stages[i]:
                x = block(x, category_prototypes=category_prototypes)
            out.append(self.extra_norms[i](x))
        return tuple(out)

    def forward(self, x, category_prototypes=None):
        return self.forward_features(x, category_prototypes=category_prototypes)


# =========================================================================
#  mmseg-registered wrapper
# =========================================================================

@MODELS.register_module()
class BiFormer_standalone(BiFormer):
    """BiFormer backbone registered in mmseg for segmentation comparison.

    Config example (BiFormer-S):
        backbone = dict(
            type='BiFormer_standalone',
            depth=[4, 4, 18, 4],
            embed_dim=[64, 128, 256, 512],
            mlp_ratios=[3, 3, 3, 3],
            n_win=8,
            kv_downsample_mode='identity',
            kv_per_wins=[-1, -1, -1, -1],
            topks=[1, 4, 16, -2],
            side_dwconv=5,
            before_attn_dwconv=3,
            layer_scale_init_value=-1,
            qk_dims=[64, 128, 256, 512],
            head_dim=32,
            param_routing=False, diff_routing=False, soft_routing=False,
            pre_norm=True, pe=None,
            auto_pad=True,
            drop_path_rate=0.3)
    """

    def __init__(self, pretrained=None, **kwargs):
        super().__init__(**kwargs)
        self.init_weights(pretrained)
        nn.SyncBatchNorm.convert_sync_batchnorm(self)

    def init_weights(self, pretrained=None):
        if isinstance(pretrained, str):
            load_checkpoint(self, pretrained, strict=False)


@MODELS.register_module()
class BiFormer_fusion_clip(BiFormer):
    """CLIP 路径的 BiFormer Attention backbone 包装。

    与 BiFormer_fusion（topp 版）平行存在，区别仅是注意力主路径用标准
    BiLevelRoutingAttention 而非 ToppAttention；其余完全一致：
      - 继承 BiFormer（已含 TTRM 路由级文本注入 + TextCrossAttention 特征级注入）
      - 额外加跨层自门控融合（conv11 + bn + sigmoid + 上采样），供解码头多尺度解码
      - forward(x, category_prototypes) 返回 (feats_tuple, category_prototypes)，
        与 CLIPEncoderDecoder.extract_feat 接口对齐
    """

    def __init__(self, pretrained=None, **kwargs):
        super().__init__(**kwargs)
        # 父类 BiFormer 已建好 self.extra_norms（4 个 LayerNorm2d），这里直接复用，
        # 只补跨层自门控融合所需的 bn / conv11（仅前 3 个 stage 用到）
        self.bn = nn.ModuleList()
        self.conv11 = nn.ModuleList()
        for i in range(3):
            self.bn.append(nn.BatchNorm2d(self.embed_dim[i]))
            self.conv11.append(nn.Conv2d(self.embed_dim[i + 1], self.embed_dim[i], 1, 1, 0))

        self.init_weights(pretrained)
        nn.SyncBatchNorm.convert_sync_batchnorm(self)
        self.sigmoid = nn.Sigmoid()

    def init_weights(self, pretrained=None):
        if isinstance(pretrained, str):
            print(f'Loading pretrained weights from {pretrained}')
            load_checkpoint(self, pretrained, strict=False)

    def forward_features(self, x, category_prototypes=None):
        """返回 (feats_tuple, category_prototypes)，与 CLIPEncoderDecoder 接口对齐。"""
        out = []
        stage_features = []
        for i in range(4):
            x = self.downsample_layers[i](x)
            for block in self.stages[i]:
                x = block(x, category_prototypes=category_prototypes)
            stage_features.append(x)
        # 跨层自门控融合：深层特征上采样后调制浅层
        for i in range(3):
            gate_visual = self.sigmoid(
                self.bn[i](self.conv11[i](stage_features[i + 1])))
            target_size = stage_features[i].shape[2:]
            gate_up = F.interpolate(gate_visual, size=target_size, mode='bilinear', align_corners=False)
            stage_features[i] = stage_features[i] + gate_up * stage_features[i]
        for i in range(4):
            out.append(self.extra_norms[i](stage_features[i]))
        return tuple(out), category_prototypes

    def forward(self, x, category_prototypes=None):
        feats, prototypes = self.forward_features(
            x, category_prototypes=category_prototypes)
        if category_prototypes is None:
            return feats
        return feats, prototypes
