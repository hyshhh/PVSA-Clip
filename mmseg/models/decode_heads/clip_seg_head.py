# Copyright (c) OpenMMLab. All rights reserved.
import torch
import torch.nn as nn
import torch.nn.functional as F
from mmcv.cnn import ConvModule

from mmseg.models.decode_heads.decode_head import BaseDecodeHead
from mmseg.registry import MODELS
from ..utils import resize


@MODELS.register_module()
class CLIPSegHead(BaseDecodeHead):
    """CLIP-based segmentation head with contrastive classification.

    Combines multi-scale feature fusion (SegformerHead style) with
    BNContrastiveHead-style cosine similarity classification against
    text category prototypes.

    Args:
        embed_dim (int): Text embedding dimension. Default: 512
        interpolate_mode (str): Upsample mode. Default: 'bilinear'
    """

    def __init__(self, embed_dim=512, interpolate_mode='bilinear', **kwargs):
        super().__init__(input_transform='multiple_select', **kwargs)

        self.embed_dim = embed_dim
        self.interpolate_mode = interpolate_mode
        num_inputs = len(self.in_channels)

        assert num_inputs == len(self.in_index)

        self.convs = nn.ModuleList()
        for i in range(num_inputs):
            self.convs.append(
                ConvModule(
                    in_channels=self.in_channels[i],
                    out_channels=self.channels,
                    kernel_size=1,
                    stride=1,
                    norm_cfg=self.norm_cfg,
                    act_cfg=self.act_cfg))

        self.fusion_conv = ConvModule(
            in_channels=self.channels * num_inputs,
            out_channels=self.channels,
            kernel_size=1,
            norm_cfg=self.norm_cfg)

        self.proj_norm = nn.BatchNorm2d(self.channels)
        self.proj = nn.Conv2d(self.channels, embed_dim, kernel_size=1)

        # Temperature and bias (from YOLOE BNContrastiveHead)
        self.logit_scale = nn.Parameter(-1.0 * torch.ones([]))
        self.bias = nn.Parameter(torch.tensor([-10.0]))

        # Fallback: standard classification for non-CLIP mode
        self.cls_seg = nn.Conv2d(embed_dim, self.num_classes, kernel_size=1)

        # Inference prototypes buffer
        self._inference_prototypes = None

    def set_category_prototypes(self, prototypes):
        """Set category prototypes for inference."""
        self._inference_prototypes = prototypes

    def predict(self, inputs, batch_img_metas, test_cfg,
                category_prototypes=None):
        """Predict segmentation results."""
        if category_prototypes is not None:
            self._inference_prototypes = category_prototypes
        return self(inputs)

    def forward(self, inputs, category_prototypes=None):
        """Forward pass."""
        inputs = self._transform_inputs(inputs)
        outs = []
        for idx in range(len(inputs)):
            x = inputs[idx]
            conv = self.convs[idx]
            outs.append(
                resize(
                    input=conv(x),
                    size=inputs[0].shape[2:],
                    mode=self.interpolate_mode,
                    align_corners=self.align_corners))

        out = self.fusion_conv(torch.cat(outs, dim=1))

        out = self.proj_norm(out)
        out = self.proj(out)  # [B, embed_dim, H, W]

        prototypes = category_prototypes
        if prototypes is None:
            prototypes = self._inference_prototypes

        if prototypes is not None:
            # BN + dot product (same as YOLOE BNContrastiveHead)
            # BN handles normalization, dot product with fixed prototypes
            # This is fusible into a single Conv2d for deployment
            w = prototypes.unsqueeze(0)  # [1, K, D]
            seg_logits = torch.einsum("bchw,bkc->bkhw", out, w)
            seg_logits = seg_logits * self.logit_scale.exp() + self.bias
        else:
            seg_logits = self.cls_seg(out)

        return seg_logits

    def fuse_for_deployment(self, category_prototypes):
        """Fuse BN + proj + cosine + scale + bias into single Conv2d."""
        with torch.no_grad():
            w = category_prototypes  # [K, D]
            scale = self.logit_scale.exp()

            proj_weight = self.proj.weight.data  # [D, C, 1, 1]
            proj_bias = self.proj.bias.data  # [D]

            bn = self.proj_norm
            running_mean = bn.running_mean  # [C]
            running_var = bn.running_var  # [C]
            bn_weight = bn.weight.data  # [C]
            bn_bias = bn.bias.data  # [C]
            eps = bn.eps

            # BN before Conv: y = Conv(BN(x))
            # BN(x) = a*x + c where a=gamma/sigma, c=beta-gamma*mu/sigma
            a = bn_weight / torch.sqrt(running_var + eps)  # [C]
            c = bn_bias - running_mean * a  # [C]

            # Fused: y = (W*a)*x + (W*c + b)
            # proj_weight: [D, C, 1, 1], a: [C]
            fused_weight = proj_weight * a.view(1, -1, 1, 1)  # [D, C, 1, 1]
            fused_bias = proj_bias + (proj_weight.squeeze(-1).squeeze(-1) @ c)  # [D]

            # Contrastive fusion: w @ fused_weight
            fused_w_2d = fused_weight.squeeze(-1).squeeze(-1)  # [D, C]
            contrastive_weight = torch.matmul(w * scale, fused_w_2d)  # [K, C]
            contrastive_bias = torch.matmul(
                w * scale, fused_bias.unsqueeze(-1)
            ).squeeze(-1) + self.bias.data  # [K]

            new_conv = nn.Conv2d(self.channels, category_prototypes.shape[0], kernel_size=1)
            new_conv.weight.data.copy_(contrastive_weight.unsqueeze(-1).unsqueeze(-1))
            new_conv.bias.data.copy_(contrastive_bias)

            self.cls_seg = new_conv
            self.proj = nn.Identity()
            self.proj_norm = nn.Identity()
