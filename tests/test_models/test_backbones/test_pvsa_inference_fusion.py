from pathlib import Path

import torch
import torch.nn as nn
from torch.nn.utils.fusion import fuse_conv_bn_eval


def _load_depthwise_conv_module_class():
    root = Path(__file__).resolve().parents[3]
    path = root / 'mmseg' / 'models' / 'backbones' / 'bi_topp_vote.py'
    source = path.read_text(encoding='utf-8')
    helper_start = source.index('def _fuse_conv_bn')
    helper_end = source.index('\n\ndef get_pe_layer')
    class_start = source.index('class DepthWiseConvModule')
    class_end = source.index('\nclass ChannelWeights')
    namespace = {
        'nn': nn,
        'torch': torch,
        'fuse_conv_bn_eval': fuse_conv_bn_eval,
    }
    exec(source[helper_start:helper_end], namespace)
    exec(source[class_start:class_end], namespace)
    return namespace['DepthWiseConvModule']


def test_depthwise_conv_module_fuse_for_inference_matches_eval_output():
    DepthWiseConvModule = _load_depthwise_conv_module_class()
    torch.manual_seed(3)
    module = DepthWiseConvModule(8, 16, 8).eval()
    x = torch.randn(2, 8, 16, 16)

    with torch.no_grad():
        expected = module(x)
        module.fuse_for_inference()
        actual = module(x)

    torch.testing.assert_close(actual, expected, rtol=1e-5, atol=1e-5)
    assert isinstance(module.bn1, torch.nn.Identity)
    assert isinstance(module.bn2, torch.nn.Identity)
    assert isinstance(module.bn3, torch.nn.Identity)


def test_flops_analysis_disables_dynamic_inference_fusion():
    root = Path(__file__).resolve().parents[3]
    backbone = (
        root / 'mmseg' / 'models' / 'backbones' / 'bi_topp_vote.py'
    ).read_text(encoding='utf-8')
    fusion = (
        root / 'mmseg' / 'models' / 'backbones' / 'biformer_fusion.py'
    ).read_text(encoding='utf-8')
    flops = (
        root / 'tools' / 'analysis_tools' / 'get_flops.py'
    ).read_text(encoding='utf-8')

    assert 'self._disable_inference_fusion = False' in backbone
    assert 'or self._disable_inference_fusion' in backbone
    assert 'or self._disable_inference_fusion' in fusion
    assert "model.backbone._disable_inference_fusion = True" in flops
