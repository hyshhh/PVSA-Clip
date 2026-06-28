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


@MODELS.register_module()
class CLIPEncoderDecoder(EncoderDecoder):
    """CLIP-enhanced Encoder-Decoder segmentor.

    Extends EncoderDecoder with a TextEncoder that produces category
    prototypes from a prompt bank. These prototypes are passed to the
    backbone (TTRM routing + CPFM refinement) and decode head
    (contrastive classification).

    During training, CPFM refines text embeddings with visual context.
    After training, enhanced prototypes are saved as .pt for deployment.
    During inference, frozen prototypes are loaded and the TextEncoder
    is removed (zero text overhead).

    Args:
        text_encoder (dict): Config for TextEncoder.
    """

    def __init__(self, text_encoder: dict, **kwargs):
        super().__init__(**kwargs)
        self.text_encoder_cfg = text_encoder

        # Build TextEncoder
        self.text_encoder = TextEncoder(
            embed_dim=text_encoder.get('embed_dim', 512),
            num_categories=text_encoder.get('num_categories', 3),
            prompts_per_category=text_encoder.get('prompts_per_category', 10))

        # Load prompt bank if path provided
        prompt_bank_path = text_encoder.get('prompt_bank_path', None)
        if prompt_bank_path and os.path.exists(prompt_bank_path):
            self.text_encoder.load_prompt_bank(prompt_bank_path)

        # Frozen prototypes for inference (loaded from .pt)
        self.register_buffer(
            'frozen_prototypes',
            torch.zeros(text_encoder.get('num_categories', 3),
                        text_encoder.get('embed_dim', 512)))
        self._prototypes_frozen = False

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

    def extract_feat(self, inputs: Tensor):
        """Extract features from images and produce text prototypes.

        Args:
            inputs: [B, 3, H, W] input images

        Returns:
            tuple of stage features (4 tensors)
            category_prototypes: [K, D] original CLIP text embeddings
        """
        if self._prototypes_frozen:
            category_prototypes = self.frozen_prototypes
        else:
            category_prototypes = self.text_encoder()

        backbone_out = self.backbone(inputs, category_prototypes=category_prototypes)

        if isinstance(backbone_out, tuple) and len(backbone_out) == 2:
            feats, _ = backbone_out
        else:
            feats = backbone_out

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
        seg_logits = self.decode_head.predict(
            feats, batch_img_metas, self.test_cfg,
            category_prototypes=category_prototypes)

        return seg_logits

    def _forward(self, inputs: Tensor,
                 data_samples: Optional[SampleList] = None) -> Tensor:
        """Network forward process."""
        feats, category_prototypes = self.extract_feat(inputs)
        return self.decode_head.forward(
            feats, category_prototypes=category_prototypes)
