# Copyright (c) OpenMMLab. All rights reserved.
from .decode_head import BaseDecodeHead
from .segformer_head import SegformerHead
from .clip_seg_head import CLIPSegHeadV2
# 标准 mmseg head（用于对比实验）
from .fcn_head import FCNHead
from .aspp_head import ASPPHead
from .sep_aspp_head import DepthwiseSeparableASPPHead
from .psp_head import PSPHead
from .uper_head import UPerHead

__all__ = [
    'BaseDecodeHead', 'SegformerHead', 'CLIPSegHeadV2',
    'FCNHead', 'ASPPHead', 'DepthwiseSeparableASPPHead', 'PSPHead', 'UPerHead',
]
