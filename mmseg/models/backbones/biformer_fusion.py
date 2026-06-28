from mmseg.registry import MODELS
import torch
import torch.nn as nn
from torch.nn.utils.fusion import fuse_conv_bn_eval
from .bi_topp_vote import VTFormer
from ..utils.topp_flash_kernel import _load_cuda_extension
from ..utils.cpfm import CPFM
from timm.models.layers import LayerNorm2d
from mmengine.runner import load_checkpoint
import torch.nn.functional as F
@MODELS.register_module()
class BiFormer_fusion(VTFormer):
    def __init__(self, pretrained=None, cpfm_config=None, **kwargs):
        super().__init__(**kwargs)
        self.extra_norms = nn.ModuleList()
        self.bn = nn.ModuleList()
        self.conv11 = nn.ModuleList()
        for i in range(4):
            self.extra_norms.append(LayerNorm2d(self.embed_dim[i]))
            self.bn.append(nn.BatchNorm2d(self.embed_dim[i]))
        for i in range(3):
            self.conv11.append(nn.Conv2d(self.embed_dim[i + 1], self.embed_dim[i], 1, 1, 0))

        # CPFM modules (training only)
        self.cpfm_enabled = cpfm_config is not None
        self.cpfm_stages = cpfm_config.get('cpfm_stages', [2, 3]) if cpfm_config else []
        if self.cpfm_enabled:
            embed_dim = cpfm_config.get('embed_dim', 512)
            self.cpfm_modules = nn.ModuleDict()
            for stage_idx in self.cpfm_stages:
                self.cpfm_modules[str(stage_idx)] = CPFM(
                    embed_dim=embed_dim,
                    visual_dim=self.embed_dim[stage_idx],
                    num_heads=cpfm_config.get('num_heads', 8),
                    top_m=cpfm_config.get('top_m', 8))
            self.cpfm_agg = nn.Sequential(
                nn.Linear(embed_dim * len(self.cpfm_stages), embed_dim),
                nn.GELU(),
                nn.Linear(embed_dim, embed_dim))

        self.apply(self._init_weights)
        self.init_weights(pretrained=pretrained)
        nn.SyncBatchNorm.convert_sync_batchnorm(self)
        self.upsample2 = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False)
        self.sigmoid = nn.Sigmoid()

    def init_weights(self, pretrained=None):
        if isinstance(pretrained, str):
            print(f'Loading pretrained weights from {pretrained}')
            load_checkpoint(self, pretrained, strict=False)
        elif pretrained is None:
            for m in self.modules():
                if isinstance(m, nn.Linear):
                    nn.init.trunc_normal_(m.weight, std=.02)
                    if m.bias is not None:
                        nn.init.constant_(m.bias, 0)
                elif isinstance(m, nn.LayerNorm):
                    nn.init.constant_(m.bias, 0)
                    nn.init.constant_(m.weight, 1.0)
        else:
            raise TypeError(f'pretrained must be a str or None, but got {type(pretrained)}')

    def forward_features(self, x: torch.Tensor, category_prototypes=None):
        if not self.training:
            self.optimize_for_inference()

        out = []
        stage_features = []
        cpfm_outputs = []

        for i in range(4):
            x = self.downsample_layers[i](x)

            for block in self.stages[i]:
                x = block(x, category_prototypes=category_prototypes)

            if (self.cpfm_enabled and i in self.cpfm_stages
                    and category_prototypes is not None and self.training):
                cpfm_out = self.cpfm_modules[str(i)](
                    category_prototypes.unsqueeze(0), x).squeeze(0)
                cpfm_outputs.append(cpfm_out)

            stage_features.append(x)

        if cpfm_outputs:
            category_prototypes = self.cpfm_agg(
                torch.cat(cpfm_outputs, dim=-1))

        for i in range(3):
            gate_visual = self.sigmoid(
                self.bn[i](self.conv11[i](stage_features[i + 1])))
            stage_features[i] = stage_features[i] + \
                self.upsample2(gate_visual) * stage_features[i]

        for i in range(4):
            out.append(self.extra_norms[i](stage_features[i]))

        return tuple(out), category_prototypes

    def optimize_for_inference(self):
        if self.training:
            return
        if getattr(self, '_inference_fused', False):
            return
        super().optimize_for_inference()
        for idx in range(len(self.conv11)):
            bn = self.bn[idx]
            if isinstance(bn, nn.modules.batchnorm._BatchNorm) and not bn.training:
                self.conv11[idx] = fuse_conv_bn_eval(self.conv11[idx], bn)
                self.bn[idx] = nn.Identity()
        if getattr(self, 'topp_flash_backend', None) in ('cuda', 'cuda_forward'):
            _load_cuda_extension()

    def forward(self, x: torch.Tensor, category_prototypes=None):
        return self.forward_features(x, category_prototypes=category_prototypes)

    def train(self, mode=True):
        super(VTFormer, self).train(mode)
        if mode and self.norm_eval:
            for m in self.modules():
                if isinstance(m, torch.nn.BatchNorm2d):
                    m.eval()
