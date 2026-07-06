# Copyright (c) OpenMMLab. All rights reserved.
import torch
import torch.nn as nn
import torch.nn.functional as F
from mmcv.cnn import ConvModule

from mmseg.models.decode_heads.decode_head import BaseDecodeHead
from mmseg.registry import MODELS
from ..utils import resize


@MODELS.register_module()
class CLIPSegHeadV2(BaseDecodeHead):
    """Activation-prompt CLIP segmentation head.

    The head keeps a normal visual segmentation branch for stable in-domain
    accuracy, then uses its class activations to build per-class visual
    prompts. Those prompts lightly adapt the corresponding text prototypes
    before BN contrastive classification.
    """

    def __init__(self,
                 embed_dim=512,
                 interpolate_mode='bilinear',
                 visual_prompt_mode='class_activation',
                 clip_logit_weight_init=0.1,
                 text_delta_scale_init=0.1,
                 visual_prompt_temperature=0.5,
                 visual_prompt_topk_ratio=0.25,
                 base_loss_weight=0.4,
                 clip_loss_weight=0.4,
                 text_drift_loss_weight=0.02,
                 **kwargs):
        super().__init__(input_transform='multiple_select', **kwargs)

        if visual_prompt_mode != 'class_activation':
            raise ValueError(
                'CLIPSegHeadV2 only supports '
                'visual_prompt_mode="class_activation" for now.')

        self.embed_dim = embed_dim
        self.interpolate_mode = interpolate_mode
        self.visual_prompt_mode = visual_prompt_mode
        self.visual_prompt_temperature = visual_prompt_temperature
        self.visual_prompt_topk_ratio = visual_prompt_topk_ratio
        self.base_loss_weight = base_loss_weight
        self.clip_loss_weight = clip_loss_weight
        self.text_drift_loss_weight = text_drift_loss_weight

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

        self.visual_proj_norm = nn.BatchNorm2d(self.channels)
        self.visual_proj = nn.Conv2d(self.channels, embed_dim, kernel_size=1)
        self.contrast_norm = nn.BatchNorm2d(embed_dim)

        # YOLOE-style BN contrastive classifier parameters.
        self.clip_logit_scale = nn.Parameter(torch.tensor(-1.0))
        self.clip_bias = nn.Parameter(torch.tensor(-10.0))

        # Fused logits = base logits + positive_gamma * clip logits.
        # Positive gates keep the text branch from being canceled by a negative
        # residual while still allowing the optimizer to shrink its influence.
        self.clip_logit_weight_raw = nn.Parameter(
            self._inverse_softplus(float(clip_logit_weight_init)))
        self.text_delta_scale_raw = nn.Parameter(
            self._inverse_softplus(float(text_delta_scale_init)))

    @staticmethod
    def _inverse_softplus(value):
        value = torch.tensor(float(value)).clamp_min(1e-6)
        return torch.log(torch.expm1(value))

    @property
    def clip_logit_weight(self):
        return F.softplus(self.clip_logit_weight_raw)

    @property
    def text_delta_scale(self):
        return F.softplus(self.text_delta_scale_raw)

    def extract_fusion_feat(self, inputs):
        """Build the fused multi-scale visual feature."""
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

        return self.fusion_conv(torch.cat(outs, dim=1))

    def _make_visual_prompt(self, visual_feat, base_logits):
        """Aggregate projected visual features by per-class activations."""
        activation = F.softmax(
            base_logits / self.visual_prompt_temperature, dim=1)  # [B, C, H, W]
        topk_ratio = self.visual_prompt_topk_ratio
        if 0 < topk_ratio < 1:
            B, C, H, W = activation.shape
            keep = max(1, int(H * W * topk_ratio))
            flat = activation.flatten(2)
            threshold = flat.topk(keep, dim=-1).values[..., -1:]
            mask = (flat >= threshold).view(B, C, H, W)
            activation = activation * mask
        denom = activation.flatten(2).sum(dim=-1).clamp_min(1e-6)
        visual_prompt = torch.einsum(
            'bkhw,bdhw->bkd', activation, visual_feat)
        visual_prompt = visual_prompt / denom.unsqueeze(-1)
        return visual_prompt

    def forward_with_text(self, inputs, text_encoder):
        """Forward with visual-condition text prototype adaptation."""
        fusion_feat = self.extract_fusion_feat(inputs)
        base_logits = self.cls_seg(fusion_feat)

        visual_feat = self.visual_proj_norm(fusion_feat)
        visual_feat = self.visual_proj(visual_feat)               # [B, D, H, W]
        visual_prompt = self._make_visual_prompt(
            visual_feat, base_logits)                             # [B, C, D]

        adapted_proto, base_proto = text_encoder.adapt_with_visual_prompt(
            visual_prompt, self.text_delta_scale)

        contrast_feat = self.contrast_norm(visual_feat)
        clip_logits = torch.einsum(
            'bdhw,bkd->bkhw', contrast_feat, adapted_proto)
        clip_logits = clip_logits * self.clip_logit_scale.exp()
        clip_logits = clip_logits + self.clip_bias

        logits = base_logits + self.clip_logit_weight * clip_logits
        return dict(
            logits=logits,
            base_logits=base_logits,
            clip_logits=clip_logits,
            visual_prompt=visual_prompt,
            adapted_proto=adapted_proto,
            base_proto=base_proto)

    def forward(self, inputs, text_encoder=None):
        """Forward pass.

        Without a text encoder this falls back to the visual branch, which
        keeps tool-side shape probing robust.
        """
        if text_encoder is None:
            return self.cls_seg(self.extract_fusion_feat(inputs))
        return self.forward_with_text(inputs, text_encoder)['logits']

    def predict_with_text(self, inputs, batch_img_metas, test_cfg,
                          text_encoder):
        """Predict segmentation results with V2 text adaptation."""
        outputs = self.forward_with_text(inputs, text_encoder)
        return self.predict_by_feat(outputs['logits'], batch_img_metas)

    def loss_with_text(self, inputs, batch_data_samples, text_encoder):
        """Compute fused, base, clip and text-drift losses."""
        outputs = self.forward_with_text(inputs, text_encoder)

        losses = self.loss_by_feat(outputs['logits'], batch_data_samples)
        losses.update(
            self._loss_branch(
                outputs['base_logits'], batch_data_samples, 'base',
                self.base_loss_weight))
        losses.update(
            self._loss_branch(
                outputs['clip_logits'], batch_data_samples, 'clip',
                self.clip_loss_weight))

        base_proto = outputs['base_proto'].unsqueeze(0)
        drift = 1 - F.cosine_similarity(
            outputs['adapted_proto'], base_proto, dim=-1)
        losses['loss_text_drift'] = (
            drift.mean() * self.text_drift_loss_weight)
        return losses

    def _loss_branch(self, logits, batch_data_samples, name, weight):
        """Compute an auxiliary CE loss and rename its log keys."""
        raw_losses = self.loss_by_feat(logits, batch_data_samples)
        branch_losses = {}
        for key, value in raw_losses.items():
            if key.startswith('loss_'):
                branch_losses[f'loss_{name}_{key[5:]}'] = value * weight
            elif key == 'acc_seg':
                branch_losses[f'acc_{name}_seg'] = value
            else:
                branch_losses[f'{name}_{key}'] = value
        return branch_losses
