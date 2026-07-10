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
    """CLIP-enhanced Encoder-Decoder segmentor."""

    _VALID_BACKBONE_TEXT_MODES = {'prompts', 'reprta'}

    def __init__(self,
                 text_encoder: dict,
                 backbone_text_encoder: Optional[dict] = None,
                 text_refiner: Optional[dict] = None,
                 use_backbone_text_injection: bool = True,
                 backbone_route_text_mode: str = 'prompts',
                 backbone_align_text_mode: str = 'prompts',
                 **kwargs):
        super().__init__(**kwargs)
        self.text_encoder_cfg = text_encoder
        self.backbone_text_encoder_cfg = backbone_text_encoder
        self.use_backbone_text_injection = use_backbone_text_injection
        self.backbone_route_text_mode = self._validate_backbone_text_mode(
            backbone_route_text_mode, 'backbone_route_text_mode')
        self.backbone_align_text_mode = self._validate_backbone_text_mode(
            backbone_align_text_mode, 'backbone_align_text_mode')
        self.use_activation_prompt_head = hasattr(
            self.decode_head, 'loss_with_text')

        self.text_encoder = self._build_text_encoder(
            text_encoder, use_visual_delta_default=self.use_activation_prompt_head)

        self.backbone_text_encoder = None
        self.text_refiner = None
        self._backbone_text_frozen = False

        route_frozen_shape = (0, )
        align_frozen_shape = (0, )
        if self.use_backbone_text_injection:
            backbone_text_encoder = backbone_text_encoder or text_encoder
            self.backbone_text_encoder_cfg = backbone_text_encoder
            self.backbone_text_encoder = self._build_text_encoder(
                backbone_text_encoder, use_visual_delta_default=False)

            backbone_embed_dim = self.backbone_text_encoder.embed_dim
            num_categories = self.backbone_text_encoder.num_categories
            num_prompts = self.backbone_text_encoder.prompts_per_category
            route_frozen_shape = (
                self._get_backbone_text_length(
                    self.backbone_route_text_mode, num_categories, num_prompts),
                backbone_embed_dim)
            align_frozen_shape = (
                self._get_backbone_text_length(
                    self.backbone_align_text_mode, num_categories, num_prompts),
                backbone_embed_dim)

            if text_refiner is not None:
                self.text_refiner = TextRefiner(
                    in_dim=text_refiner.get('in_dim', backbone_embed_dim),
                    hidden_mult=text_refiner.get('hidden_mult', 4))

        self.register_buffer(
            'frozen_backbone_route_text',
            torch.zeros(*route_frozen_shape))
        self.register_buffer(
            'frozen_backbone_align_text',
            torch.zeros(*align_frozen_shape))

    @classmethod
    def _validate_backbone_text_mode(cls, text_mode: str, field_name: str) -> str:
        if text_mode not in cls._VALID_BACKBONE_TEXT_MODES:
            raise ValueError(
                f'{field_name} must be one of '
                f'{tuple(sorted(cls._VALID_BACKBONE_TEXT_MODES))}, got {text_mode!r}.')
        return text_mode

    @staticmethod
    def _get_backbone_text_length(text_mode: str, num_categories: int,
                                  prompts_per_category: int) -> int:
        if text_mode == 'prompts':
            return num_categories * prompts_per_category
        return num_categories

    @staticmethod
    def _load_prompt_bank(text_encoder: TextEncoder, text_encoder_cfg: dict):
        prompt_bank_path = text_encoder_cfg.get('prompt_bank_path', None)
        if not prompt_bank_path:
            return
        if not os.path.exists(prompt_bank_path):
            raise FileNotFoundError(
                f'Prompt bank not found: {prompt_bank_path}. '
                f'Run: python tools/generate_water_prompt_bank.py '
                f'--output {prompt_bank_path}')
        text_encoder.load_prompt_bank(
            prompt_bank_path,
            category_order=text_encoder_cfg.get('prompt_category_order', None))

    def _build_text_encoder(self,
                            text_encoder_cfg: dict,
                            use_visual_delta_default: bool) -> TextEncoder:
        text_encoder = TextEncoder(
            embed_dim=text_encoder_cfg.get('embed_dim', 512),
            num_categories=text_encoder_cfg.get('num_categories', 3),
            prompts_per_category=text_encoder_cfg.get('prompts_per_category', 10),
            use_reprta=text_encoder_cfg.get('use_reprta', True),
            reprta_ffn_type=text_encoder_cfg.get('reprta_ffn_type', 'swiglu'),
            reprta_zero_init=text_encoder_cfg.get('reprta_zero_init', True),
            use_visual_delta=text_encoder_cfg.get(
                'use_visual_delta', use_visual_delta_default))
        self._load_prompt_bank(text_encoder, text_encoder_cfg)
        return text_encoder

    def _compute_backbone_text(self, text_mode: str):
        if not self.use_backbone_text_injection or self.backbone_text_encoder is None:
            return None
        if text_mode == 'prompts':
            raw = self.backbone_text_encoder.prompt_bank_tensor().reshape(
                -1, self.backbone_text_encoder.embed_dim)
            if self.text_refiner is not None:
                raw = self.text_refiner(raw)
            return raw
        if text_mode == 'reprta':
            return self.backbone_text_encoder.backbone_prototype_tensor()
        raise ValueError(f'Unsupported backbone text mode: {text_mode!r}')

    def get_backbone_text_inputs(self):
        if not self.use_backbone_text_injection:
            return None
        if self._backbone_text_frozen:
            return dict(
                route_text=self.frozen_backbone_route_text,
                align_text=self.frozen_backbone_align_text)

        text_cache = {}

        def _get_text(text_mode: str):
            if text_mode not in text_cache:
                text_cache[text_mode] = self._compute_backbone_text(text_mode)
            return text_cache[text_mode]

        route_text = _get_text(self.backbone_route_text_mode)
        align_text = _get_text(self.backbone_align_text_mode)
        if route_text is None and align_text is None:
            return None
        return dict(route_text=route_text, align_text=align_text)

    def get_backbone_route_text(self):
        backbone_text_inputs = self.get_backbone_text_inputs()
        if backbone_text_inputs is None:
            return None
        return backbone_text_inputs['route_text']

    def get_backbone_align_text(self):
        backbone_text_inputs = self.get_backbone_text_inputs()
        if backbone_text_inputs is None:
            return None
        return backbone_text_inputs['align_text']

    def _extract_backbone_feats(self, inputs: Tensor):
        """Extract backbone features with optional fixed text injection."""
        backbone_text_inputs = self.get_backbone_text_inputs()
        backbone_out = self.backbone(
            inputs, category_prototypes=backbone_text_inputs)
        if isinstance(backbone_out, tuple) and len(backbone_out) == 2:
            return backbone_out[0]
        return backbone_out

    def extract_feat(self, inputs: Tensor):
        """Extract image features."""
        feats = self._extract_backbone_feats(inputs)
        return feats, None

    def loss(self, inputs: Tensor, data_samples: SampleList) -> dict:
        """Calculate losses from a batch of inputs and data samples."""
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
        """Predict segmentation results."""
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
        """Freeze deployable backbone text caches."""
        self.text_encoder.freeze_for_deployment()
        if self.use_backbone_text_injection:
            if self.backbone_text_encoder is not None:
                self.backbone_text_encoder.freeze_for_deployment()
            if self.text_refiner is not None:
                self.text_refiner.eval()

            backbone_text_inputs = self.get_backbone_text_inputs()
            if backbone_text_inputs is not None:
                self.frozen_backbone_route_text.copy_(
                    backbone_text_inputs['route_text'])
                self.frozen_backbone_align_text.copy_(
                    backbone_text_inputs['align_text'])
                self._backbone_text_frozen = True
                frozen_backbone_text_inputs = dict(
                    route_text=self.frozen_backbone_route_text,
                    align_text=self.frozen_backbone_align_text)
                for module in self.backbone.modules():
                    if hasattr(module, 'freeze_for_deployment'):
                        module.freeze_for_deployment(frozen_backbone_text_inputs)
        return self
