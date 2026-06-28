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

    def __init__(self, embed_dim=512, interpolate_mode='bilinear',
                 cls_loss_weight=0.5, **kwargs):
        super().__init__(input_transform='multiple_select', **kwargs)

        self.embed_dim = embed_dim
        self.interpolate_mode = interpolate_mode
        self.cls_loss_weight = cls_loss_weight
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
        """Predict segmentation results.

        Args:
            inputs: list of stage features
            batch_img_metas: list of image meta info
            test_cfg: test config
            category_prototypes: [K, D] text embeddings (optional)

        Returns:
            seg_logits: [B, num_classes, H, W]
        """
        if category_prototypes is not None:
            self._inference_prototypes = category_prototypes
        return self(inputs)

    def forward(self, inputs, category_prototypes=None):
        """Forward pass.

        Args:
            inputs: list of 4 stage feature maps from backbone
            category_prototypes: [K, D] text embeddings for contrastive
                                classification. If None, uses cls_seg fallback.

        Returns:
            seg_logits: [B, num_classes, H, W] segmentation logits
        """
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

        # Project to embedding space
        out = self.proj_norm(out)
        out = self.proj(out)  # [B, embed_dim, H, W]

        # Use provided prototypes, or stored inference prototypes, or fallback
        prototypes = category_prototypes
        if prototypes is None:
            prototypes = self._inference_prototypes

        if prototypes is not None:
            w = prototypes.unsqueeze(0)  # [1, K, D]
            seg_logits = torch.einsum("bchw,bkc->bkhw", out, w)
            seg_logits = seg_logits * self.logit_scale.exp() + self.bias
        else:
            seg_logits = self.cls_seg(out)

        # Store visual features for L_cls computation
        self._visual_features = out
        return seg_logits

    def loss_by_feat_with_cls(self, seg_logits, batch_data_samples,
                               visual_features, prototypes):
        """Compute both L_seg and L_cls losses.

        Args:
            seg_logits: [B, K, H, W] classification logits (scaled + biased)
            batch_data_samples: GT labels
            visual_features: [B, D, H, W] raw projected features (before scale/bias)
            prototypes: [K, D] text prototypes

        Returns:
            dict with loss_seg and loss_cls
        """
        # L_seg: standard CrossEntropyLoss on classification logits
        seg_label = self._stack_batch_gt(batch_data_samples)
        loss_seg = self.loss_decode(seg_logits, seg_label,
                                     ignore_index=self.ignore_index)

        # L_cls: pixel-text cosine similarity loss
        # Normalize features and prototypes
        feat_norm = F.normalize(visual_features, dim=1)  # [B, D, H, W]
        proto_norm = F.normalize(prototypes, dim=-1)  # [K, D]
        # Raw cosine similarity: [B, K, H, W]
        cosine_sim = torch.einsum("bchw,bkc->bkhw", feat_norm,
                                   proto_norm.unsqueeze(0))
        # CrossEntropyLoss on raw cosine similarity
        loss_cls = F.cross_entropy(cosine_sim, seg_label,
                                    ignore_index=self.ignore_index)

        return dict(
            loss_seg=loss_seg * self.cls_loss_weight,
            loss_cls=loss_cls * (1 - self.cls_loss_weight))

    def _stack_batch_gt(self, batch_data_samples):
        """Stack GT semantic seg labels."""
        gt_semantic_segs = [
            data_sample.gt_sem_seg.data for data_sample in batch_data_samples
        ]
        return torch.stack(gt_semantic_segs, dim=0)

    def fuse_for_deployment(self, category_prototypes):
        """Fuse contrastive classification into Conv2d for deployment.

        Args:
            category_prototypes: [K, D] frozen text embeddings
        """
        with torch.no_grad():
            w = category_prototypes  # [K, D]
            scale = self.logit_scale.exp()

            # Fuse proj + contrastive into single Conv2d
            proj_weight = self.proj.weight.data  # [embed_dim, channels, 1, 1]
            proj_bias = self.proj.bias.data  # [embed_dim]

            # BN fusion
            bn = self.proj_norm
            running_mean = bn.running_mean
            running_var = bn.running_var
            bn_weight = bn.weight.data
            bn_bias = bn.bias.data
            eps = bn.eps

            # Fused BN scale and shift
            bn_scale = bn_weight / torch.sqrt(running_var + eps)
            bn_shift = bn_bias - running_mean * bn_scale

            # Apply BN to proj weight and bias
            fused_weight = proj_weight * bn_scale.view(-1, 1, 1, 1)
            fused_bias = proj_bias * bn_scale + bn_shift

            # Apply contrastive: w @ fused_weight
            # w: [K, D], fused_weight: [D, C, 1, 1]
            contrastive_weight = torch.matmul(
                w * scale,  # [K, D]
                fused_weight.squeeze(-1).squeeze(-1)  # [D, C]
            )  # [K, C]

            contrastive_bias = torch.matmul(
                w * scale,  # [K, D]
                fused_bias.unsqueeze(-1)  # [D, 1]
            ).squeeze(-1) + self.bias.data  # [K]

            # Create new Conv2d
            new_conv = nn.Conv2d(
                self.channels, category_prototypes.shape[0], kernel_size=1)
            new_conv.weight.data.copy_(
                contrastive_weight.unsqueeze(-1).unsqueeze(-1))
            new_conv.bias.data.copy_(contrastive_bias)

            self.cls_seg = new_conv
            self.proj = nn.Identity()
            self.proj_norm = nn.Identity()
