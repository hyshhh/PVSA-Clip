# Copyright (c) OpenMMLab. All rights reserved.
import os
from typing import Optional

import torch
from torch import Tensor

from mmseg.registry import MODELS
from mmseg.utils import SampleList, add_prefix
from .encoder_decoder import EncoderDecoder
from ..utils.text_encoder import TextEncoder
from ..utils.text_refiner import TextRefiner


@MODELS.register_module()
class CLIPEncoderDecoder(EncoderDecoder):
    """CLIP-enhanced Encoder-Decoder segmentor.

    Extends EncoderDecoder with a TextEncoder that provides prompt-bank
    prototypes to CLIPSegHeadV2. The V2 head builds image-conditioned visual
    prompts at runtime and adapts text prototypes per image.

    Args:
        text_encoder (dict): Config for TextEncoder.
    """

    def __init__(self, text_encoder: dict, text_refiner: dict = None,
                 use_backbone_text_injection: bool = True, **kwargs):
        super().__init__(**kwargs)
        self.text_encoder_cfg = text_encoder
        self.use_backbone_text_injection = use_backbone_text_injection

        embed_dim = text_encoder.get('embed_dim', 512)
        num_categories = text_encoder.get('num_categories', 3)
        self.use_activation_prompt_head = hasattr(
            self.decode_head, 'loss_with_text')

        # Build TextEncoder
        self.text_encoder = TextEncoder(
            embed_dim=embed_dim,
            num_categories=num_categories,
            prompts_per_category=text_encoder.get('prompts_per_category', 10),
            use_reprta=text_encoder.get('use_reprta', True),
            reprta_ffn_type=text_encoder.get('reprta_ffn_type', 'swiglu'),
            reprta_zero_init=text_encoder.get('reprta_zero_init', True),
            use_visual_delta=text_encoder.get(
                'use_visual_delta', self.use_activation_prompt_head))

        # Load prompt bank
        prompt_bank_path = text_encoder.get('prompt_bank_path', None)
        if prompt_bank_path:
            if os.path.exists(prompt_bank_path):
                self.text_encoder.load_prompt_bank(
                    prompt_bank_path,
                    category_order=text_encoder.get(
                        'prompt_category_order', None))
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

        # Frozen backbone text (固定 30 条) caching 分支：部署融合后使用，
        # 推理走 self.get_backbone_text() 返回此 buffer，零文本开销。
        num_prompts = text_encoder.get('prompts_per_category', 10)
        self.register_buffer(
            'frozen_backbone_text',
            torch.zeros(num_categories * num_prompts, embed_dim))
        self._backbone_text_frozen = False

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
        if not self.use_backbone_text_injection:
            return None
        if self._backbone_text_frozen:
            return self.frozen_backbone_text
        raw = self.text_encoder.prompt_bank_tensor().reshape(-1, self.text_encoder.embed_dim)
        if self.text_refiner is not None:
            raw = self.text_refiner(raw)
        return raw

    def _extract_backbone_feats(self, inputs: Tensor):
        """Extract backbone features with optional fixed text injection."""
        backbone_text = self.get_backbone_text()                  # [N, D] 固定或 None

        # backbone 仅注入固定 backbone_text（与每图原型解耦）；feats 同一来源只取一次
        backbone_out = self.backbone(inputs, category_prototypes=backbone_text)
        if isinstance(backbone_out, tuple) and len(backbone_out) == 2:
            return backbone_out[0]
        return backbone_out

    def extract_feat(self, inputs: Tensor):
        """Extract image features.

        Args:
            inputs: [B, 3, H, W] input images

        Returns:
            tuple: Backbone stage features and a reserved ``None`` placeholder
            kept for EncoderDecoder call-site compatibility.
        """
        feats = self._extract_backbone_feats(inputs)
        return feats, None

    def loss(self, inputs: Tensor, data_samples: SampleList) -> dict:
        """Calculate losses from a batch of inputs and data samples.

        Args:
            inputs: [B, 3, H, W] input images
            data_samples: list of SegDataSample

        Returns:
            dict[str, Tensor]: a dictionary of loss components
        """
        feats, _ = self.extract_feat(inputs)

        losses = dict()
        if self.use_activation_prompt_head:
            loss_decode = self.decode_head.loss_with_text(
                feats, data_samples, self.text_encoder)
            loss_decode = add_prefix(loss_decode, 'decode')
        else:
            loss_decode = self._decode_head_forward_train(
                feats, data_samples)
        losses.update(loss_decode)

        if self.with_auxiliary_head:
            loss_aux = self._auxiliary_head_forward_train(
                feats, data_samples)
            losses.update(loss_aux)

        return losses

    def _decode_head_forward_train(self, inputs, data_samples, **kwargs):
        """Run forward of decode head and compute loss."""
        losses = dict()
        loss_decode = self.decode_head.loss(
            inputs, data_samples, self.train_cfg, **kwargs)
        losses.update(add_prefix(loss_decode, 'decode'))
        return losses

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

        feats, _ = self.extract_feat(inputs)
        seg_logits = self.inference(feats, batch_img_metas)
        return self.postprocess_result(seg_logits, data_samples)

    def inference(self, feats, batch_img_metas, rescale=True):
        """Inference with augmented test time."""
        if self.use_activation_prompt_head:
            seg_logits = self.decode_head.predict_with_text(
                feats, batch_img_metas, self.test_cfg, self.text_encoder)
        else:
            seg_logits = self.decode_head.predict(
                feats, batch_img_metas, self.test_cfg)

        return seg_logits

    def _forward(self, inputs: Tensor,
                 data_samples: Optional[SampleList] = None) -> Tensor:
        """Network forward process."""
        feats, _ = self.extract_feat(inputs)
        if self.use_activation_prompt_head:
            return self.decode_head.forward(feats, text_encoder=self.text_encoder)
        return self.decode_head.forward(feats)

    @torch.no_grad()
    def fuse_for_deployment(self):
        """Freeze deployable backbone text caches.

        CLIPSegHeadV2 keeps image-conditioned visual prompts at runtime, so
        the decode head is intentionally not fused into a fixed Conv2d.
        """
        # 先把 TextEncoder 的 prompt 投影预算入 buffer，使推理零投影开销。
        self.text_encoder.freeze_for_deployment()
        if self.use_backbone_text_injection:
            with torch.no_grad():
                if self.text_refiner is not None:
                    self.text_refiner.eval()
                refined = self.get_backbone_text()
                self.frozen_backbone_text.copy_(refined)
                self._backbone_text_frozen = True

            for name, module in self.backbone.named_modules():
                if hasattr(module, 'freeze_for_deployment'):
                    module.freeze_for_deployment(self.frozen_backbone_text)
        return self
