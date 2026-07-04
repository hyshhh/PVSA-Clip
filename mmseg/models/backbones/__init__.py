# Copyright (c) OpenMMLab. All rights reserved.
from .bi_topp_vote import VTFormer
from .biformer_fusion import BiFormer_fusion
from .bi_topp_vote_baseline import VTFormer as VTFormer_baseline
from .biformer_fusion_baseline import BiFormer_fusion_baseline
# 标准 mmseg backbone（用于对比实验）
from .resnet import ResNet
from .swin import SwinTransformer
from .mit import MixVisionTransformer
from .biformer import BiFormer_fusion_clip, BiFormer_standalone

__all__ = [
    'VTFormer', 'BiFormer_fusion', 'VTFormer_baseline', 'BiFormer_fusion_baseline',
    'ResNet', 'SwinTransformer', 'MixVisionTransformer', 'BiFormer_standalone',
    'BiFormer_fusion_clip',
]
