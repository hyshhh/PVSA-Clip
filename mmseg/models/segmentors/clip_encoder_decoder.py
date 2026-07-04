# Copyright (c) OpenMMLab. All rights reserved.
import logging
import os
from typing import List, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from mmengine.logging import print_log
from torch import Tensor

from mmseg.registry import MODELS
from mmseg.utils import (ConfigType, OptConfigType, OptMultiConfig,
                         OptSampleList, SampleList, add_prefix)
from .encoder_decoder import EncoderDecoder
from ..utils.text_encoder import TextEncoder
from ..utils.text_refiner import TextRefiner


@MODELS.register_module()
class CLIPEncoderDecoder(EncoderDecoder):
    """CLIP-enhanced Encoder-Decoder segmentor.

    Extends EncoderDecoder with a TextEncoder that produces category
    prototypes from a prompt bank. These prototypes are passed to the
    backbone (TTRM routing + TextCrossAttention) and decode head
    (contrastive classification).

    During inference, frozen prototypes are loaded and the TextEncoder
    is removed (zero text overhead).

    Args:
        text_encoder (dict): Config for TextEncoder.
    """

    def __init__(self, text_encoder: dict, text_refiner: dict = None,
                 image_query_proj: dict = None, **kwargs):
        super().__init__(**kwargs)
        self.text_encoder_cfg = text_encoder

        embed_dim = text_encoder.get('embed_dim', 512)
        num_categories = text_encoder.get('num_categories', 3)

        # Build TextEncoder
        self.text_encoder = TextEncoder(
            embed_dim=embed_dim,
            num_categories=num_categories,
            prompts_per_category=text_encoder.get('prompts_per_category', 10),
            use_reprta=text_encoder.get('use_reprta', True),
            reprta_ffn_type=text_encoder.get('reprta_ffn_type', 'swiglu'),
            reprta_zero_init=text_encoder.get('reprta_zero_init', True))

        # Load prompt bank
        prompt_bank_path = text_encoder.get('prompt_bank_path', None)
        if prompt_bank_path:
            if os.path.exists(prompt_bank_path):
                self.text_encoder.load_prompt_bank(prompt_bank_path)
            else:
                raise FileNotFoundError(
                    f'Prompt bank not found: {prompt_bank_path}. '
                    f'Run: python tools/generate_water_prompt_bank.py '
                    f'--output {prompt_bank_path}')

        # TextRefiner: backbone 注入前的文本重构（固定 30 条，不接图像）
        self.text_refiner = None
        if text_refiner is not None:
            self.text_refiner = TextRefiner(
                in_dim=text_refiner.get('in_dim', embed_dim),
                hidden_mult=text_refiner.get('hidden_mult', 4))

        # image_query_proj: 骨干多 stage 特征 -> 图相关 query [B, C, D]
        # stage 通道由 decode_head.in_channels 决定（与 backbone embed_dim 对齐）
        self.image_query_proj = None
        if image_query_proj is not None:
            stage_channels = image_query_proj.get(
                'stage_channels',
                getattr(self.decode_head, 'in_channels', None))
            if stage_channels is None:
                raise ValueError(
                    'image_query_proj.stage_channels not given and '
                    'decode_head.in_channels unavailable')
            in_dim = sum(stage_channels)
            hidden_dim = image_query_proj.get('hidden_dim', embed_dim)
            self.image_query_proj = nn.Sequential(
                nn.Linear(in_dim, hidden_dim),
                nn.GELU(),
                nn.Linear(hidden_dim, num_categories * embed_dim))

        # Frozen prototypes for inference (loaded from .pt)
        self.register_buffer(
            'frozen_prototypes',
            torch.zeros(num_categories, embed_dim))
        self._prototypes_frozen = False

        # Frozen backbone text (固定 30 条) caching 分支：部署融合后使用，
        # 推理走 self.get_backbone_text() 返回此 buffer，零文本开销。
        num_prompts = text_encoder.get('prompts_per_category', 10)
        self.register_buffer(
            'frozen_backbone_text',
            torch.zeros(num_categories * num_prompts, embed_dim))
        self._backbone_text_frozen = False

    def freeze_prototypes(self, save_path=None):
        """Freeze and save original CLIP prototypes for deployment.

        Head always uses original CLIP embeddings. CPFM only affects
        backbone feature learning during training.

        Args:
            save_path: Path to save .pt file. If None, uses default.
        """
        with torch.no_grad():
            self.frozen_prototypes.copy_(self.text_encoder())
            self._prototypes_frozen = True

        if save_path:
            torch.save({
                'prototypes': self.frozen_prototypes,
                'embed_dim': self.text_encoder.embed_dim,
                'num_categories': self.text_encoder.num_categories,
            }, save_path)
            print_log(
                f'Saved frozen prototypes to {save_path}',
                logger='current')

    def get_backbone_text(self):
        """Return the text tensor to inject into the backbone.

        Always fixed [N, D] (N = num_categories * prompts_per_category),
        derived from the frozen prompt bank optionally refined by TextRefiner.
        Does NOT depend on the input image, so the backbone injection is
        per-image invariant and stays deployable (caches as fixed K/V).

        During training this is recomputed each forward so TextRefiner (if
        present) receives gradients; during fused inference the cached
        _frozen_backbone_text buffer is returned instead.
        """
        if self._backbone_text_frozen:
            return self.frozen_backbone_text
        raw = self.text_encoder.prompt_bank_tensor().reshape(-1, self.text_encoder.embed_dim)
        if self.text_refiner is not None:
            raw = self.text_refiner(raw)
        return raw

    def _make_image_query(self, feats):
        """Pool multi-stage features -> per-image query [B, C, D].

        Each stage is globally pooled, concatenated, then mapped by the
        image_query_proj MLP to [B, C*D] -> [B, C, D]. Combined with the
        learnable attn_pool_query prior (each normalized then summed) to
        form the fused query fed to TextEncoder.pool_with_query. If the
        projection module is absent, falls back to broadcasting the
        learnable prior (no image conditioning).
        """
        B = feats[0].shape[0]
        D = self.text_encoder.embed_dim
        C = self.text_encoder.num_categories
        if self.image_query_proj is not None:
            pooled = [F.adaptive_avg_pool2d(f, 1).flatten(1) for f in feats]
            cat = torch.cat(pooled, dim=1)               # [B, sum_channels]
            img_q = self.image_query_proj(cat)            # [B, C*D]
            img_q = img_q.view(B, C, D)
        else:
            # 无图像投影：退化为仅用可学先验（仍走图相关接口，但 q 与图像无关）
            img_q = self.text_encoder.attn_pool_query.squeeze(1)        # [C, D]
            img_q = img_q.unsqueeze(0).expand(B, -1, -1)                # [B, C, D]
        # Learnable prior
        prior = self.text_encoder.attn_pool_query.squeeze(1)         # [C, D]
        prior = prior.unsqueeze(0).expand(B, -1, -1)                # [B, C, D]
        # Fuse: each L2 normalized then added
        fused = F.normalize(img_q, dim=-1, p=2) + F.normalize(prior, dim=-1, p=2)
        return fused

    def extract_feat(self, inputs: Tensor):
        """Extract features from images and produce text prototypes.

        Args:
            inputs: [B, 3, H, W] input images

        Returns:
            feats: tuple of stage features (4 tensors)
            category_prototypes: [B, C, D] per-image prototypes (image-
                conditioned pooling when image_query_proj is present);
                [C, D] for the frozen-deployment branch.
        """
        backbone_text = self.get_backbone_text()                  # [N, D] 固定

        # backbone 仅注入固定 backbone_text（与每图原型解耦）；feats 同一来源只取一次
        backbone_out = self.backbone(inputs, category_prototypes=backbone_text)
        if isinstance(backbone_out, tuple) and len(backbone_out) == 2:
            feats = backbone_out[0]
        else:
            feats = backbone_out

        if self._prototypes_frozen:
            # 部署近似分支：用冻结原型作 head 分类（不再算图相关）
            category_prototypes = self.frozen_prototypes        # [C, D]
        else:
            # 图相关分支：用骨干特征算 query 并池化得 per-image 原型
            fused_q = self._make_image_query(feats)             # [B, C, D]
            category_prototypes = self.text_encoder.pool_with_query(fused_q)

        return feats, category_prototypes

    def loss(self, inputs: Tensor, data_samples: SampleList) -> dict:
        """Calculate losses from a batch of inputs and data samples.

        Args:
            inputs: [B, 3, H, W] input images
            data_samples: list of SegDataSample

        Returns:
            dict[str, Tensor]: a dictionary of loss components
        """
        feats, category_prototypes = self.extract_feat(inputs)

        losses = dict()
        loss_decode = self._decode_head_forward_train(
            feats, data_samples, category_prototypes=category_prototypes)
        losses.update(loss_decode)

        if self.with_auxiliary_head:
            loss_aux = self._auxiliary_head_forward_train(
                feats, data_samples)
            losses.update(loss_aux)

        return losses

    def _decode_head_forward_train(self, inputs, data_samples,
                                   category_prototypes=None, **kwargs):
        """Run forward of decode head and compute loss."""
        losses = dict()
        if hasattr(self.decode_head, 'set_category_prototypes'):
            self.decode_head.set_category_prototypes(category_prototypes)
        loss_decode = self.decode_head.loss(
            inputs, data_samples, self.train_cfg, **kwargs)
        losses.update(add_prefix(loss_decode, 'decode'))
        return losses

    def _decode_head_supports_prototypes(self) -> bool:
        """Whether current decode head consumes CLIP category prototypes."""
        return hasattr(self.decode_head, 'set_category_prototypes')

    def predict(self, inputs: Tensor,
                data_samples: Optional[SampleList] = None) -> SampleList:
        """Predict segmentation results.

        Args:
            inputs: [B, 3, H, W] input images
            data_samples: optional list of SegDataSample

        Returns:
            list of SegDataSample with pred_sem_seg
        """
        if data_samples is not None:
            batch_img_metas = [
                data_sample.metainfo for data_sample in data_samples
            ]
        else:
            batch_img_metas = [
                dict(
                    ori_shape=inputs.shape[2:],
                    img_shape=inputs.shape[2:],
                    pad_shape=inputs.shape[2:],
                    padding_size=[0, 0, 0, 0])
            ] * inputs.shape[0]

        feats, category_prototypes = self.extract_feat(inputs)
        seg_logits = self.inference(feats, batch_img_metas,
                                    category_prototypes=category_prototypes)
        return self.postprocess_result(seg_logits, data_samples)

    def inference(self, feats, batch_img_metas,
                  category_prototypes=None, rescale=True):
        """Inference with augmented test time."""
        if self._decode_head_supports_prototypes():
            seg_logits = self.decode_head.predict(
                feats, batch_img_metas, self.test_cfg,
                category_prototypes=category_prototypes)
        else:
            seg_logits = self.decode_head.predict(
                feats, batch_img_metas, self.test_cfg)

        return seg_logits

    def _forward(self, inputs: Tensor,
                 data_samples: Optional[SampleList] = None) -> Tensor:
        """Network forward process."""
        feats, category_prototypes = self.extract_feat(inputs)
        if self._decode_head_supports_prototypes():
            return self.decode_head.forward(
                feats, category_prototypes=category_prototypes)
        return self.decode_head.forward(feats)

    @torch.no_grad()
    def fuse_for_deployment(self, fuse_head: bool = False):
        """Fuse CLIP modules for deployment.

        与图相关版本（image_query_proj 存在）配套的部署语义：
        - backbone 段：注入的固定 [N, D] 文本经 TextRefiner 重构后冻结成
          ``frozen_backbone_text`` buffer，TTRM / TextCrossAttention 把
          "refiner + norm + text_proj"链跑一遍缓存进各自的 _frozen_k/v，
          推理零文本开销。✅ 保留。
        - head 段：图相关原型 ``[B, C, D]`` 每图不同，对比分类融不了单
          Conv2d（数学必然）。默认 ``fuse_head=False``，跳过 head 融合，
          推理实时跑 image_query_proj + pool_with_query + einsum（精度无
          损，开销可忽略）。
        - 旧固定原型路径（无 image_query_proj）：若 ``fuse_head=True``，
          用 ``frozen_prototypes`` 近似融合 head 成单 Conv2d，沿用旧行为
          （有损：每图图相关选择被平均掉）。

        Call this AFTER loading checkpoint and BEFORE inference.
        """
        # 1. Freeze backbone text：把重构后的固定 30 条缓存进 buffer，
        #    extract_feat.get_backbone_text() 之后再返回这份缓存。
        with torch.no_grad():
            if self.text_refiner is not None:
                self.text_refiner.eval()
            refined = self.get_backbone_text()              # [N, D] 实时算一次
            self.frozen_backbone_text.copy_(refined)
            self._backbone_text_frozen = True

        # 2. Freeze backbone TTRM and TextCrossAttention（注入源用 frozen_backbone_text）
        for name, module in self.backbone.named_modules():
            if hasattr(module, 'freeze_for_deployment'):
                module.freeze_for_deployment(self.frozen_backbone_text)

        # 3. Head 段融合：仅当显式请求 fuse_head（旧固定原型场景）
        if fuse_head:
            self.text_encoder.eval()
            with torch.no_grad():
                # 用 attn_pool_query 先验单独产固定原型近似融合
                category_prototypes = self.text_encoder()
            self.frozen_prototypes.copy_(category_prototypes)
            self._prototypes_frozen = True
            if hasattr(self.decode_head, 'fuse_for_deployment'):
                self.decode_head.fuse_for_deployment(category_prototypes)
            # 走近似路径后，image_query_proj 不再参与 head（用固定原型）
        else:
            # 图相关路径：保留实时 image_query_proj + pool_with_query
            # frozen_prototypes 不启用，extract_feat 继续走图相关分支
            pass

        # 4. TextEncoder 与 TextRefiner 保留（head 实时池化仍需调用）；
        #    若 fuse_head=True 走近似 + 无图相关，可再视情况删除——这里保守保留。
        return self
