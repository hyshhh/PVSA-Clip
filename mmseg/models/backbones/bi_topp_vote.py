import math
from collections import OrderedDict

import torch
import torch.nn as nn
from torch.nn.utils.fusion import fuse_conv_bn_eval

from fairscale.nn.checkpoint import checkpoint_wrapper
from timm.models.layers import DropPath, trunc_normal_
from ..utils.common import DWConv
from ..utils.top_p_bra import ToppAttention
from mmseg.registry import MODELS


def _normalize_topp_backend(backend):
    if backend is None:
        return None
    backend = str(backend).strip().lower()
    if backend in ('', 'none', 'false', 'off'):
        return None
    return backend


def _fuse_conv_bn(conv, bn):
    if not isinstance(bn, nn.modules.batchnorm._BatchNorm):
        return conv, bn
    if conv.training or bn.training:
        return conv, bn
    fused = fuse_conv_bn_eval(conv, bn)
    return fused, nn.Identity()


def _fuse_sequential_conv_bn(module):
    for child in module.children():
        _fuse_sequential_conv_bn(child)
    if not isinstance(module, nn.Sequential):
        return
    children = list(module.children())
    i = 0
    while i + 1 < len(children):
        if isinstance(children[i], nn.Conv2d) and isinstance(
                children[i + 1], nn.modules.batchnorm._BatchNorm):
            children[i], children[i + 1] = _fuse_conv_bn(
                children[i], children[i + 1])
            i += 2
        else:
            i += 1
    module._modules.clear()
    for idx, child in enumerate(children):
        module.add_module(str(idx), child)



def get_pe_layer(emb_dim, pe_dim=None, name='none'):
    if name == 'none':
        return nn.Identity()
    else:
        raise ValueError(f'PE name {name} is not supported!')

class Block(nn.Module):
    def __init__(self, dim, drop_path=0., layer_scale_init_value=-1,
                 num_heads=8, n_win=7, qk_dim=None, qk_scale=None,
                 kv_per_win=4, kv_downsample_ratio=4, kv_downsample_kernel=None, kv_downsample_mode='ada_avgpool',
                 topk=4, param_attention="qkvo", param_routing=False, diff_routing=False, soft_routing=False,
                 mlp_ratio=4, mlp_dwconv=False,
                 side_dwconv=5, before_attn_dwconv=3, pre_norm=True, auto_pad=False, W=False,
                 topp_flash_block_windows=64,
                 topp_flash_backend=None,
                 topp_route_configs=None,
                 attn_vis_config=None,
                 debug_route=False,
                 topp_flash_debug=False,
                 use_route_mask=False,
                 use_ttrm=False,
                 soft_kv_weight=0.5,
                 route_pooling='avg',
                 use_plain_attn=False,
                 cross_attn_module=None):
        super().__init__()
        qk_dim = qk_dim or dim
        self.W = W

        if before_attn_dwconv > 0:
            self.pos_embed = nn.Conv2d(dim, dim, kernel_size=before_attn_dwconv, padding=1, groups=dim)
        else:
            self.pos_embed = lambda x: 0

        if topk > 0 and not use_plain_attn:
            self.PA = ToppAttention(dim=dim, num_heads=num_heads, n_win=n_win, qk_dim=qk_dim,
                                    qk_scale=qk_scale, kv_per_win=kv_per_win,
                                    kv_downsample_ratio=kv_downsample_ratio,
                                    kv_downsample_kernel=kv_downsample_kernel,
                                    kv_downsample_mode=kv_downsample_mode,
                                    topk=topk, param_attention=param_attention, param_routing=param_routing,
                                    diff_routing=diff_routing, soft_routing=soft_routing,
                                    side_dwconv=side_dwconv,
                                    auto_pad=auto_pad, W=self.W,
                                    topp_flash_block_windows=topp_flash_block_windows,
                                    topp_flash_backend=topp_flash_backend,
                                    topp_route_configs=topp_route_configs,
                                    attn_vis_config=attn_vis_config,
                                    debug_route=debug_route,
                                    topp_flash_debug=topp_flash_debug,
                                    use_route_mask=use_route_mask,
                                    use_ttrm=use_ttrm,
                                    soft_kv_weight=soft_kv_weight,
                                    route_pooling=route_pooling)
            self._use_plain_attn = False
        elif topk > 0 and use_plain_attn:
            self.PA = Attention(dim=dim, num_heads=num_heads)
            self._use_plain_attn = True

        # Cross-attention module (for stages 2-3, shared across blocks in same stage)
        self.cross_attn = cross_attn_module

        self.norm3 = nn.LayerNorm(dim, eps=1e-6)
        self.norm4 = nn.LayerNorm(dim, eps=1e-6)
        self.mlp2 = nn.Sequential(nn.Linear(dim, int(mlp_ratio * dim)),
                                 DWConv(int(mlp_ratio * dim)) if mlp_dwconv else nn.Identity(),
                                 nn.GELU(),
                                 nn.Linear(int(mlp_ratio * dim), dim))
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.pre_norm = pre_norm

    def forward(self, x, category_prototypes=None):
        x = x + self.pos_embed(x)
        x = x.permute(0, 2, 3, 1)
        if self._use_plain_attn:
            PA = self.PA(self.norm3(x))
        else:
            PA = self.PA(self.norm3(x), None, category_prototypes=category_prototypes)
        if self.pre_norm:
            x = x + self.drop_path(PA)
            # Cross-attention: visual features attend to text prototypes
            if self.cross_attn is not None and category_prototypes is not None:
                x = self.cross_attn(x, category_prototypes)
            x = x + self.drop_path(self.mlp2(self.norm4(x)))
        x = x.permute(0, 3, 1, 2)
        return x
class FeatureAlignmentModule(nn.Module):
    def __init__(self, dim, reduction=1, lambda_c=.5, lambda_s=.5):
        super(FeatureAlignmentModule, self).__init__()
        self.lambda_c = lambda_c
        self.lambda_s = lambda_s
        self.channel_weights = ChannelWeights(dim=dim, reduction=reduction)
        self.spatial_weights = SpatialWeights(dim=dim, reduction=reduction)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)
        elif isinstance(m, nn.Conv2d):
            fan_out = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
            fan_out //= m.groups
            m.weight.data.normal_(0, math.sqrt(2.0 / fan_out))
            if m.bias is not None:
                m.bias.data.zero_()
    
    def forward(self, x1, x2):
        channel_weights = self.channel_weights(x1, x2)
        spatial_weights = self.spatial_weights(x1, x2)
        out_x1 = x1 + 0.5 * channel_weights[1] * x2 + 0.5 * spatial_weights[1] * x2
        out_x2 = x2 + 0.5* channel_weights[0] * x1 + 0.5 * spatial_weights[0] * x1
        return out_x1, out_x2

class DepthWiseConvModule(nn.Module):
    def __init__(self,
                 embed_dims,
                 feedforward_channels,
                 output_channels,
                 kernel_size=3,
                 stride=1,
                 padding=1,
                 drop_rate=0.,
                 dilation=1):
        super(DepthWiseConvModule, self).__init__()
        
        # 1. 自动计算 Padding，保证 stride=1 时尺寸不变
        # 考虑到 dilation 的情况: padding = dilation * (kernel_size - 1) // 2
        padding = dilation * (kernel_size - 1) // 2

        # 2. 第一个点卷积 (1x1 Conv): 升维 (Expansion)
        self.fc1 = nn.Conv2d(embed_dims, feedforward_channels, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(feedforward_channels) # 加上 BN
        # 3. 深度卷积 (Depthwise Conv)
        self.pe_conv = nn.Conv2d(
            in_channels=feedforward_channels,
            out_channels=feedforward_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            dilation=dilation,
            groups=feedforward_channels, # 关键：Groups = Channels
            bias=False)
        self.bn2 = nn.BatchNorm2d(feedforward_channels) # 加上 BN
        self.activate = nn.GELU() # 或者是 build_activation_layer(act_cfg)
        # 4. 第二个点卷积 (1x1 Conv): 降维 (Projection)
        self.fc2 = nn.Conv2d(feedforward_channels, output_channels, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(output_channels) # 加上 BN
        self.drop = nn.Dropout(drop_rate)
        # 处理残差连接时的维度/步长不匹配问题
        self.downsample = None
        if stride != 1 or embed_dims != output_channels:
            self.downsample = nn.Sequential(
                nn.Conv2d(embed_dims, output_channels, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(output_channels)
            )
    def forward(self, x):
        identity = x

        # 典型的结构：Conv -> BN -> Act -> Conv -> BN -> Act ...
        # 这里采用类似 MobileNetV2/SegFormer 的顺序
        out = self.fc1(x)
        out = self.bn1(out)
        out = self.activate(out) # 升维后激活
        out = self.pe_conv(out)
        out = self.bn2(out)
        out = self.activate(out) # 深度卷积后激活
        out = self.fc2(out)
        out = self.bn3(out)
        # 最后通常不激活，直接做 Dropout 和 Add
        out = self.drop(out)
        # 残差连接
        if self.downsample is not None:
            identity = self.downsample(x)
        return out + identity

    def fuse_for_inference(self):
        if self.training:
            return
        self.fc1, self.bn1 = _fuse_conv_bn(self.fc1, self.bn1)
        self.pe_conv, self.bn2 = _fuse_conv_bn(self.pe_conv, self.bn2)
        self.fc2, self.bn3 = _fuse_conv_bn(self.fc2, self.bn3)
        if self.downsample is not None:
            _fuse_sequential_conv_bn(self.downsample)

class ChannelWeights(nn.Module):
    def __init__(self, dim, reduction=1):
        super(ChannelWeights, self).__init__()
        self.dim = dim
        self.avg_pool = nn.AdaptiveAvgPool2d(1)#自适应平均池化，(B, 96, 256, 256) → (B, 96, 1, 1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)
        self.mlp_avg = nn.Sequential(
                    nn.Linear(self.dim, self.dim),#如果我的输入向量是96，但是全连接层在
                    nn.ReLU(inplace=True),
                    nn.Linear(self.dim, 2))
        self.mlp_max = nn.Sequential(
                    nn.Linear(self.dim, self.dim),
                    nn.ReLU(inplace=True),
                    nn.Linear(self.dim, 2))
        self.mlp = nn.Sequential(
                    nn.Linear(self.dim, self.dim),
                    nn.ReLU(inplace=True),
                    nn.Linear(self.dim, self.dim),
                    nn.Sigmoid())

    def forward(self, x1, x2):
        B, C, H, W = x1.shape
        x = torch.cat((x1, x2), dim=1)

        avg = self.avg_pool(x).view(B, 2 * C)
        avg_attn = self.mlp_avg(avg).softmax(dim=-1)
        avg_x1, avg_x2 = (avg_attn.view(B, 2, 1) * avg.view(B, 2, C)).chunk(2, dim=1)
        avg_x = (avg_x1 + avg_x2).view(B, C)

        # Max. Adaptive normalization
        max = self.max_pool(x).view(B, 2 * C)
        max_attn = self.mlp_max(max).softmax(dim=-1)
        max_x1, max_x2 = (max_attn.view(B, 2, 1) * max.view(B, 2, C)).chunk(2, dim=1)
        max_x = (max_x1 + max_x2).view(B, C)

        y = torch.cat((avg_x, max_x), dim=1)
        y = self.mlp(y).view(B, self.dim, 1)
        channel_weights = y.reshape(B, 2, C, 1, 1).permute(1, 0, 2, 3, 4)
        return channel_weights

class SpatialWeights(nn.Module):
    def __init__(self, dim, reduction=1):
        super(SpatialWeights, self).__init__()
        self.dim = dim
        self.mlp = nn.Sequential(
                    nn.Conv2d(self.dim, self.dim // reduction, kernel_size=1),
                    nn.ReLU(inplace=True),
                    nn.Conv2d(self.dim // reduction, 2, kernel_size=1), 
                    nn.Sigmoid())

    def forward(self, x1, x2):
        B, _, H, W = x1.shape
        x = torch.cat((x1, x2), dim=1)
        spatial_weights = self.mlp(x).reshape(B, 2, 1, H, W).permute(1, 0, 2, 3, 4)
        return spatial_weights
@MODELS.register_module()
class VTFormer(nn.Module):
    def __init__(self, depth=[3, 4, 8, 3], in_chans=3, num_classes=1000, embed_dim=[64, 128, 320, 512],
                 head_dim=64, qk_scale=None, representation_size=None,
                 drop_path_rate=0., drop_rate=0.,
                 use_checkpoint_stages=[],
                 ########
                 n_win=7,
                 kv_downsample_mode='ada_avgpool',
                 kv_per_wins=[2, 2, -1, -1],
                 topks=[8, 8, -1, -1],
                 side_dwconv=5,
                 layer_scale_init_value=-1,
                 qk_dims=[None, None, None, None],
                 param_routing=False, diff_routing=False, soft_routing=False,
                 pre_norm=True,
                 pe=None,
                 pe_stages=[0],
                 before_attn_dwconv=3,
                 auto_pad=False,
                 # -----------------------
                 kv_downsample_kernels=[4, 2, 1, 1],
                 kv_downsample_ratios=[4, 2, 1, 1],  # -> kv_per_win = [2, 2, 2, 1]
                 mlp_ratios=[4, 4, 4, 4],
                 param_attention='qkvo',
                 mlp_dwconv=False,
                 norm_eval=False,
                 W=False,
                 topp_flash_backend=None,
                 topp_flash_block_windows=64,
                 topp_flash_debug=False,
                 topp_route_configs=None,
                 attn_vis_config=None,
                 debug_route=False,
                 use_route_mask=False,
                 route_pooling='avg',
                 use_plain_attn_last_stage=False,
                 fam_reduction=4,
                 cnn_dwconv_layers=[2, 1, 2, 1],
                 feature_vis_config=None,
                 use_ttrm=False,
                 ttrm_stages=[0, 1, 2, 3],
                 cross_attn_stages=[2, 3],
                 remove_cnn_branch=False,
                 soft_kv_weight=0.5,
                 **kwargs):

        super().__init__()
        self.W = W
        self.remove_cnn_branch = remove_cnn_branch
        self.topp_flash_backend = _normalize_topp_backend(topp_flash_backend)
        self.topp_flash_block_windows = topp_flash_block_windows
        self.topp_flash_debug = topp_flash_debug
        self.topp_route_configs = topp_route_configs
        self.attn_vis_config = attn_vis_config
        self.debug_route = debug_route
        self.use_route_mask = use_route_mask
        self.route_pooling = route_pooling
        self.use_plain_attn_last_stage = use_plain_attn_last_stage
        self.feature_vis_config = feature_vis_config or {}
        self._inference_fused = False
        self._disable_inference_fusion = False
        self.num_classes = num_classes
        self.num_features = self.embed_dim = embed_dim  # num_features for consistency with other models
        self.norm_eval = norm_eval
        ############ downsample layers (patch embeddings) ######################
        self.downsample_layers = nn.ModuleList()
        if not remove_cnn_branch:
            self.downsample_layers2 = nn.ModuleList()
            self.FAM = nn.ModuleList()
        else:
            self.downsample_layers2 = None
            self.FAM = None



        # NOTE: uniformer uses two 3*3 conv, while in many other transformers this is one 7*7 conv
        stem = nn.Sequential(
            nn.Conv2d(in_chans, embed_dim[0] // 2, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1)),
            nn.BatchNorm2d(embed_dim[0] // 2),
            nn.GELU(),
            nn.Conv2d(embed_dim[0] // 2, embed_dim[0], kernel_size=(3, 3), stride=(2, 2), padding=(1, 1)),
            nn.BatchNorm2d(embed_dim[0]),
        )

        if (pe is not None) and 0 in pe_stages:
            stem.append(get_pe_layer(emb_dim=embed_dim[0], name=pe))
        if use_checkpoint_stages:
            stem = checkpoint_wrapper(stem)
        self.downsample_layers.append(stem)

        if not remove_cnn_branch:
            stem2_layers = [
                nn.Conv2d(in_chans, embed_dim[0] // 2, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1)),
                nn.BatchNorm2d(embed_dim[0] // 2),
                nn.GELU(),
                nn.Conv2d(embed_dim[0] // 2, embed_dim[0], kernel_size=(3, 3), stride=(2, 2), padding=(1, 1)),
                nn.BatchNorm2d(embed_dim[0]),
            ]
            stem2_layers.extend([
                DepthWiseConvModule(embed_dim[0], 4*embed_dim[0], embed_dim[0], 3, 1, 1)
                for _ in range(cnn_dwconv_layers[0])
            ])
            stem2 = nn.Sequential(*stem2_layers)
            if (pe is not None) and 0 in pe_stages:
                stem2.append(get_pe_layer(emb_dim=embed_dim[0], name=pe))
            if use_checkpoint_stages:
                stem2 = checkpoint_wrapper(stem2)
            self.downsample_layers2.append(stem2)
            self.FAM.append(FeatureAlignmentModule(dim=2*embed_dim[0], reduction=fam_reduction))
            self.fusion = nn.ModuleList()
            self.sigmoid = nn.Sigmoid()
            self.fusion.append(
                nn.Conv2d(2*embed_dim[0], embed_dim[0], kernel_size=(1,1), stride=(1, 1), padding=(0, 0),bias=True)
            )
        else:
            self.fusion = nn.ModuleList()
            self.sigmoid = nn.Sigmoid()

        for i in range(3):
            downsample_layer = nn.Sequential(
                nn.Conv2d(embed_dim[i], embed_dim[i + 1], kernel_size=(3, 3), stride=(2, 2), padding=(1, 1)),
                nn.BatchNorm2d(embed_dim[i + 1])
            )
            if (pe is not None) and i + 1 in pe_stages:
                downsample_layer.append(get_pe_layer(emb_dim=embed_dim[i + 1], name=pe))
            if use_checkpoint_stages:
                downsample_layer = checkpoint_wrapper(downsample_layer)
            self.downsample_layers.append(downsample_layer)

            if not remove_cnn_branch:
                layers = [
                    nn.Conv2d(embed_dim[i], embed_dim[i + 1], kernel_size=(3, 3), stride=(2, 2), padding=(1, 1)),
                    nn.BatchNorm2d(embed_dim[i + 1])
                ]
                layers.extend([
                    DepthWiseConvModule(embed_dim[i + 1], 4 * embed_dim[i + 1], embed_dim[i + 1], 3, 1, 1)
                    for _ in range(cnn_dwconv_layers[i + 1])
                ])
                downsample_layer2 = nn.Sequential(*layers)
                if (pe is not None) and i + 1 in pe_stages:
                    downsample_layer2.append(get_pe_layer(emb_dim=embed_dim[i + 1], name=pe))
                if use_checkpoint_stages:
                    downsample_layer2 = checkpoint_wrapper(downsample_layer2)
                self.downsample_layers2.append(downsample_layer2)
                self.fusion.append(
                    nn.Conv2d(2*embed_dim[i + 1], embed_dim[i + 1], kernel_size=(1,1), stride=(1, 1), padding=(0, 0),bias=True)
                )
                self.FAM.append(FeatureAlignmentModule(dim=2*embed_dim[i + 1], reduction=fam_reduction))

        ##########################################################################

        self.stages = nn.ModuleList()
        nheads = [dim // head_dim for dim in qk_dims]
        dp_rates = [x.item() for x in torch.linspace(0, drop_path_rate, sum(depth))]
        cur = 0
        self.use_ttrm = use_ttrm
        self.ttrm_stages = ttrm_stages
        self.cross_attn_stages = cross_attn_stages

        from ..utils.text_cross_attn import TextCrossAttention

        for i in range(4):
            # Each block gets its own cross-attention instance (no sharing)
            use_ca = i in cross_attn_stages
            stage = nn.Sequential(
                *[Block(dim=embed_dim[i], drop_path=dp_rates[cur + j],
                        layer_scale_init_value=layer_scale_init_value,
                        topk=topks[i],
                        num_heads=nheads[i],
                        n_win=n_win,
                        qk_dim=qk_dims[i],
                        qk_scale=qk_scale,
                        kv_per_win=kv_per_wins[i],
                        kv_downsample_ratio=kv_downsample_ratios[i],
                        kv_downsample_kernel=kv_downsample_kernels[i],
                        kv_downsample_mode=kv_downsample_mode,
                        param_attention=param_attention,
                        param_routing=param_routing,
                        diff_routing=diff_routing,
                        soft_routing=soft_routing,
                        mlp_ratio=mlp_ratios[i],
                        mlp_dwconv=mlp_dwconv,
                        side_dwconv=side_dwconv,
                        before_attn_dwconv=before_attn_dwconv,
                        pre_norm=pre_norm,
                        auto_pad=auto_pad,
                        W=self.W,
                        topp_flash_block_windows=self.topp_flash_block_windows,
                        topp_flash_backend=self.topp_flash_backend,
                        topp_route_configs=self.topp_route_configs,
                        attn_vis_config=self.attn_vis_config,
                        debug_route=self.debug_route,
                        topp_flash_debug=self.topp_flash_debug,
                        use_route_mask=self.use_route_mask,
                        use_ttrm=(use_ttrm and i in ttrm_stages),
                        soft_kv_weight=soft_kv_weight,
                        route_pooling=self.route_pooling,
                        use_plain_attn=(self.use_plain_attn_last_stage and i == 3),
                        cross_attn_module=TextCrossAttention(
                            visual_dim=embed_dim[i], text_dim=512,
                            num_heads=nheads[i]) if use_ca else None
                        ) for j in range(depth[i])],
            )
            if i in use_checkpoint_stages:
                stage = checkpoint_wrapper(stage)
            self.stages.append(stage)
            cur += depth[i]

        ##########################################################################
        self.norm = nn.BatchNorm2d(embed_dim[-1])
        if representation_size:
            self.num_features = representation_size
            self.pre_logits = nn.Sequential(OrderedDict([
                ('fc', nn.Linear(embed_dim[-1], representation_size)),
                ('act', nn.Tanh())
            ]))
        else:
            self.pre_logits = nn.Identity()

        self.head = nn.Linear(embed_dim[-1], num_classes) if num_classes > 0 else nn.Identity()
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    @torch.jit.ignore
    def no_weight_decay(self):
        return {'pos_embed', 'cls_token'}

    def get_classifier(self):
        return self.head

    def reset_classifier(self, num_classes, global_pool=''):
        self.num_classes = num_classes
        self.head = nn.Linear(self.embed_dim, num_classes) if num_classes > 0 else nn.Identity()

    def optimize_for_inference(self):
        if (self.training or self._inference_fused
                or self._disable_inference_fusion):
            return
        for layer in self.downsample_layers:
            _fuse_sequential_conv_bn(layer)
        if self.downsample_layers2 is not None:
            for layer in self.downsample_layers2:
                _fuse_sequential_conv_bn(layer)
        for module in self.modules():
            if isinstance(module, DepthWiseConvModule):
                module.fuse_for_inference()
        self._inference_fused = True

    def forward_features(self, x, category_prototypes=None):
        for i in range(4):
            x = self.downsample_layers[i](x)  # res = (56, 28, 14, 7), wins = (64, 16, 4, 1)
            # Iterate through blocks manually to pass category_prototypes
            for block in self.stages[i]:
                x = block(x, category_prototypes=category_prototypes)
        x = self.norm(x)
        x = self.pre_logits(x)
        return x

    def forward(self, x):
        x = self.forward_features(x)
        x = x.flatten(2).mean(-1)
        return x

    def train(self, mode=True):
        super(VTFormer, self).train(mode)
        if mode and self.norm_eval:
            for m in self.modules():
                if isinstance(m, torch.nn.BatchNorm2d):
                    m.eval()

#################### model variants #######################
