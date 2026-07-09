from mmseg.registry import MODELS
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
from torch.nn.utils.fusion import fuse_conv_bn_eval
from .bi_topp_vote_baseline import VTFormer
from ..utils.topp_flash_kernel import _load_cuda_extension
from timm.models.layers import LayerNorm2d
from mmengine.runner import load_checkpoint
import os
import numpy as np
import cv2
from PIL import Image
import torch.nn.functional as F
@MODELS.register_module()
class BiFormer_fusion_baseline(VTFormer):
    def __init__(self, pretrained=None, cross_stage_fusion_mode='gate',
                 fusion_type='conv1x1', **kwargs):
        valid_fusion_modes = {
            'none', 'gate', 'concat', 'gate_concat', 'cross_gate', 'cross_concat'
        }
        if cross_stage_fusion_mode not in valid_fusion_modes:
            raise ValueError(
                f'cross_stage_fusion_mode must be one of {valid_fusion_modes}, '
                f'but got {cross_stage_fusion_mode}')
        valid_fusion_types = {'conv1x1', 'conv1x1_bn_gelu', 'conv1x1_bn_gelu_dwconv'}
        if fusion_type not in valid_fusion_types:
            raise ValueError(
                f'fusion_type must be one of {valid_fusion_types}, '
                f'but got {fusion_type}')
        self.cross_stage_fusion_mode = cross_stage_fusion_mode
        self.fusion_type = fusion_type
        super().__init__(**kwargs)
        self.extra_norms = nn.ModuleList()
        self.trans_bn = nn.ModuleList()
        self.cnn_bn = nn.ModuleList()
        self.cnn_conv=nn.ModuleList()
        self.trans_conv=nn.ModuleList()
        self.trans_cross_stage_fusion = nn.ModuleList()
        self.cnn_cross_stage_fusion = nn.ModuleList()
        self.trans_gate_scale = nn.ParameterList()
        self.cnn_gate_scale = nn.ParameterList()
        for i in range(4):
            self.extra_norms.append(LayerNorm2d(self.embed_dim[i]))
            if i < 3:
                # trans_conv/cnn_conv 作用在高一层 (i+1) 的特征上做通道投影，
                # 输入通道应为 embed_dim[i+1]，而非 2*embed_dim[i]（此前二者恰好数值相等，
                # 是因为 embed_dim 逐级翻倍，换成非翻倍配置会直接报通道不匹配）。
                self.trans_bn.append(nn.BatchNorm2d(self.embed_dim[i]))
                self.cnn_bn.append(nn.BatchNorm2d(self.embed_dim[i]))
                self.cnn_conv.append(nn.Conv2d(self.embed_dim[i + 1], self.embed_dim[i], 1, 1, 0))
                self.trans_conv.append(nn.Conv2d(self.embed_dim[i + 1], self.embed_dim[i], 1, 1, 0))
                self.trans_cross_stage_fusion.append(
                    nn.Conv2d(2 * self.embed_dim[i], self.embed_dim[i], 1, 1, 0))
                self.cnn_cross_stage_fusion.append(
                    nn.Conv2d(2 * self.embed_dim[i], self.embed_dim[i], 1, 1, 0))
                self.trans_gate_scale.append(nn.Parameter(torch.tensor(0.0)))
                self.cnn_gate_scale.append(nn.Parameter(torch.tensor(0.0)))
            
            
        self.apply(self._init_weights)
        self.init_weights(pretrained=pretrained)
        nn.SyncBatchNorm.convert_sync_batchnorm(self)
        self.sigmoid = nn.Sigmoid()

    def _build_fusion_layer(self, channels):
        if self.fusion_type == 'conv1x1':
            return nn.Conv2d(2 * channels, channels, kernel_size=1, stride=1, padding=0, bias=True)

        layers = [
            nn.Conv2d(2 * channels, channels, kernel_size=1, stride=1, padding=0, bias=False),
            nn.BatchNorm2d(channels),
            nn.GELU(),
        ]
        if self.fusion_type == 'conv1x1_bn_gelu_dwconv':
            layers.extend([
                nn.Conv2d(
                    channels, channels, kernel_size=3, stride=1, padding=1,
                    groups=channels, bias=False),
                nn.BatchNorm2d(channels),
                nn.GELU(),
            ])
        return nn.Sequential(*layers)



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


    def forward_features(self, x: torch.Tensor):
        # NOTE: optimize_for_inference() removed from here.
        # Conv+BN fusing mutates model state_dict, causing key mismatch
        # when loading checkpoints after eval()+forward triggers fusion.
        # Call model.optimize_for_inference() explicitly AFTER loading
        # weights if you want inference fusion.

        vis_cfg = self.feature_vis_config
        vis_enabled = vis_cfg.get('enabled', False)
        vis_once = vis_cfg.get('once', True)
        vis_dir = vis_cfg.get('save_dir', 'cam/features_imgs4')
        vis_out_size = vis_cfg.get('out_size', 512)
        vis_reduce = vis_cfg.get('channel_reduce', 'mean')

        if vis_enabled and (not vis_once or not getattr(self, '_feature_vis_saved', False)):
            os.makedirs(vis_dir, exist_ok=True)

        out = []
        cnn_out = x
        trans_features=[]
        cnn_features=[]
        fused_features=[]

        # ── CNN 分支全零时：纯 Transformer 路径，跳过 CNN/FAM/Fusion ──
        if getattr(self, '_cnn_disabled', False):
            for i in range(4):
                x = self.trans_downsample_layers[i](x)
                x = self.stages[i](x)
                out.append(x)
            return out

        for i in range(4):
            if vis_enabled and (not vis_once or not getattr(self, '_feature_vis_saved', False)):
                self._save_feature_channel_as_image(x, f'{vis_dir}/stage{i}_xinput.png', vis_out_size, vis_reduce)
            cnn_out = self.cnn_downsample_layers[i](cnn_out)
            x = self.trans_downsample_layers[i](x)
            x = self.stages[i](x)
            if vis_enabled and (not vis_once or not getattr(self, '_feature_vis_saved', False)):
                self._save_feature_channel_as_image(x, f'{vis_dir}/stage{i}_before_FAM_x.png', vis_out_size, vis_reduce)
                self._save_feature_channel_as_image(cnn_out, f'{vis_dir}/stage{i}_before_FAM_cnn.png', vis_out_size, vis_reduce)

            if self.use_fam:
                x, cnn_out = self.FAM[i](x, cnn_out)
            trans_features.append(x)
            cnn_features.append(cnn_out)

            if vis_enabled and (not vis_once or not getattr(self, '_feature_vis_saved', False)):
                self._save_feature_channel_as_image(x, f'{vis_dir}/stage{i}_after_FAM_x.png', vis_out_size, vis_reduce)
                self._save_feature_channel_as_image(cnn_out, f'{vis_dir}/stage{i}_after_FAM_cnn.png', vis_out_size, vis_reduce)

        for i in range(3):
            trans_feat = trans_features[i]
            cnn_feat = cnn_features[i]
            if self.cross_stage_fusion_mode == 'none':
                fused_features.append(self.fusion[i](torch.cat((trans_feat, cnn_feat), dim=1)))
                continue
            target_size = trans_features[i].shape[2:]
            if self.cross_stage_fusion_mode in {'gate', 'gate_concat', 'cross_gate'}:
                trans_proj = self.trans_conv[i](trans_features[i + 1])
                cnn_proj = self.cnn_conv[i](cnn_features[i + 1])
                trans_mask = self.sigmoid(self.trans_bn[i](trans_proj))
                cnn_mask = self.sigmoid(self.cnn_bn[i](cnn_proj))
                trans_scale = 2.0 * self.trans_gate_scale[i].sigmoid()
                cnn_scale = 2.0 * self.cnn_gate_scale[i].sigmoid()
                trans_mask = F.interpolate(trans_mask, size=target_size, mode='bilinear', align_corners=False)
                cnn_mask = F.interpolate(cnn_mask, size=target_size, mode='bilinear', align_corners=False)
                if vis_enabled and (not vis_once or not getattr(self, '_feature_vis_saved', False)) and i == 0:
                    self._save_feature_channel_as_image(trans_mask, f'{vis_dir}/trans_mask.png', vis_out_size, vis_reduce)
                    self._save_feature_channel_as_image(cnn_mask, f'{vis_dir}/cnn_mask.png', vis_out_size, vis_reduce)
                if self.cross_stage_fusion_mode == 'cross_gate':
                    trans_feat = trans_feat * (1 + trans_scale * (cnn_mask - 0.5))
                    cnn_feat = cnn_feat * (1 + cnn_scale * (trans_mask - 0.5))
                else:
                    trans_feat = trans_feat * (1 + trans_scale * (trans_mask - 0.5))
                    cnn_feat = cnn_feat * (1 + cnn_scale * (cnn_mask - 0.5))
            if self.cross_stage_fusion_mode in {'concat', 'gate_concat', 'cross_concat'}:
                if self.cross_stage_fusion_mode in {'concat', 'cross_concat'}:
                    trans_proj = self.trans_conv[i](trans_features[i + 1])
                    cnn_proj = self.cnn_conv[i](cnn_features[i + 1])
                trans_high = F.interpolate(trans_proj, size=target_size, mode='bilinear', align_corners=False)
                cnn_high = F.interpolate(cnn_proj, size=target_size, mode='bilinear', align_corners=False)
                if self.cross_stage_fusion_mode == 'cross_concat':
                    trans_feat = self.trans_cross_stage_fusion[i](
                        torch.cat((trans_feat, cnn_high), dim=1))
                    cnn_feat = self.cnn_cross_stage_fusion[i](
                        torch.cat((cnn_feat, trans_high), dim=1))
                else:
                    trans_feat = self.trans_cross_stage_fusion[i](
                        torch.cat((trans_feat, trans_high), dim=1))
                    cnn_feat = self.cnn_cross_stage_fusion[i](
                        torch.cat((cnn_feat, cnn_high), dim=1))
            fused_features.append(self.fusion[i](torch.cat((trans_feat, cnn_feat), dim=1)))
        fused_features.append(self.fusion[3](torch.cat((trans_features[3], cnn_features[3]), dim=1)))

        for i in range(4):
            if vis_enabled and (not vis_once or not getattr(self, '_feature_vis_saved', False)):
                self._save_feature_channel_as_image(fused_features[i], f'{vis_dir}/stage{i}_after_channel.png', vis_out_size, vis_reduce)
            out.append(self.extra_norms[i](fused_features[i]))

        if vis_enabled:
            self._feature_vis_saved = True
        return tuple(out)



    def _save_feature_channel_as_image(
        self,
        feature_map,
        file_path,
        out_size=512,        # (H, W)，如 (512, 512)
        channel_reduce="mean" # "mean" | "max"
    ):
        """
        feature_map: [B, C, H, W] or [C, H, W]
        file_path: 保存路径
        out_size: 上采样到的空间尺寸 (H, W)，None 表示不变
        channel_reduce: 通道聚合方式
        """

        # ---------- 1. 维度统一 ----------
        if feature_map.dim() == 4:
            feature_map = feature_map[0]  # [C, H, W]

        assert feature_map.dim() == 3, "feature_map must be [C, H, W]"

        # ---------- 2. 通道聚合（深层特征必须做） ----------
        if channel_reduce == "mean":
            fmap = feature_map.mean(dim=0, keepdim=True)  # [1, H, W]
        elif channel_reduce == "max":
            fmap, _ = feature_map.max(dim=0, keepdim=True)
        else:
            raise ValueError(f"Unsupported channel_reduce: {channel_reduce}")

        fmap = fmap.unsqueeze(0)  # [1, 1, H, W]

        # ---------- 3. 上采样到目标分辨率 ----------
        if out_size is not None:
            fmap = F.interpolate(
                fmap,
                size=out_size,
                mode="bilinear",
                align_corners=False
            )
        # ---------- 4. 转 numpy ----------
        fmap = fmap[0, 0].detach().cpu().numpy()
        # ---------- 5. 归一化（用于可视化） ----------
        fmap = fmap - fmap.min()
        fmap = fmap / (fmap.max() + 1e-5)
        # ---------- 6. 轻度平滑（可选，但论文图更友好） ----------
        fmap = cv2.GaussianBlur(fmap, (3, 3), sigmaX=0.5, sigmaY=0.5)
        # ---------- 7. 映射为彩色热力图 ----------
        cmap = plt.get_cmap("viridis")
        img_color = (cmap(fmap)[:, :, :3] * 255).astype(np.uint8)
        # ---------- 8. 保存 ----------
        Image.fromarray(img_color).save(file_path)

    def optimize_for_inference(self):
        if self.training:
            return
        # Fallback guard: skip if parent already fused
        if getattr(self, '_inference_fused', False):
            return
        # Fuse parent (VTFormer) conv-bn layers
        super().optimize_for_inference()
        for idx in range(len(self.trans_conv)):
            trans_bn = self.trans_bn[idx]
            cnn_bn = self.cnn_bn[idx]
            if isinstance(trans_bn, nn.modules.batchnorm._BatchNorm) and not trans_bn.training:
                self.trans_conv[idx] = fuse_conv_bn_eval(self.trans_conv[idx], trans_bn)
                self.trans_bn[idx] = nn.Identity()
            if isinstance(cnn_bn, nn.modules.batchnorm._BatchNorm) and not cnn_bn.training:
                self.cnn_conv[idx] = fuse_conv_bn_eval(self.cnn_conv[idx], cnn_bn)
                self.cnn_bn[idx] = nn.Identity()
        if getattr(self, 'topp_flash_backend', None) in ('cuda', 'cuda_forward'):
            _load_cuda_extension()

    def forward(self, x: torch.Tensor):
        return self.forward_features(x)

    def train(self, mode=True):
        super(VTFormer, self).train(mode)
        if mode and self.norm_eval:
            for m in self.modules():
                if isinstance(m, torch.nn.BatchNorm2d):
                    m.eval()


@MODELS.register_module()
class BiFormer_sequential(BiFormer_fusion_baseline):
    """顺序双分支骨干：C+T / T+C。

    与并行版 BiFormer_fusion_baseline 的区别：
      - branch_order='cnn_first'  → C+T：CNN 先跑，输出投影后喂给 Transformer
      - branch_order='trans_first' → T+C：Transformer 先跑，输出投影后喂给 CNN
    第二分支的 stem 被跳过，改用 1×1 投影层匹配通道数。

    融合流程（FAM / VFM / self.fusion）与并行版完全一致。
    """

    def __init__(self, branch_order='cnn_first', **kwargs):
        super().__init__(**kwargs)
        self.branch_order = branch_order
        embed_dim = self.embed_dim  # [64, 128, 256, 512]

        # ── 投影层：将第一分支输出通道映射到第二分支 stage 期望的通道 ──
        if branch_order == 'cnn_first':
            # CNN → T：CNN stage 输出投影到 Transformer stage 输入通道
            self.cnn_to_trans_proj = nn.ModuleList([
                nn.Conv2d(embed_dim[i], embed_dim[i], kernel_size=1, bias=False)
                for i in range(4)
            ])
        else:
            # T → CNN：Transformer stage 输出投影到 CNN stage 输入通道
            self.trans_to_cnn_proj = nn.ModuleList([
                nn.Conv2d(embed_dim[i], embed_dim[i], kernel_size=1, bias=False)
                for i in range(4)
            ])

        self.apply(self._init_weights)

    # ── 顺序 forward ────────────────────────────────────────────────────────
    def forward_features(self, x: torch.Tensor):
        vis_cfg = self.feature_vis_config
        vis_enabled = vis_cfg.get('enabled', False)
        vis_once = vis_cfg.get('once', True)
        vis_dir = vis_cfg.get('save_dir', 'cam/features_imgs4')
        vis_out_size = vis_cfg.get('out_size', 512)
        vis_reduce = vis_cfg.get('channel_reduce', 'mean')

        if vis_enabled and (not vis_once or not getattr(self, '_feature_vis_saved', False)):
            os.makedirs(vis_dir, exist_ok=True)

        out = []
        trans_features = []
        cnn_features = []
        fused_features = []

        # ── CNN 分支全零时：纯 Transformer 路径 ──
        if getattr(self, '_cnn_disabled', False):
            for i in range(4):
                x = self.trans_downsample_layers[i](x)
                x = self.stages[i](x)
                out.append(x)
            return out

        if self.branch_order == 'cnn_first':
            # ══════════ C+T：CNN 先跑，投影后喂给 Transformer ══════════
            cnn_out = x
            for i in range(4):
                cnn_out = self.cnn_downsample_layers[i](cnn_out)
                # 投影 CNN 输出 → Transformer stage 期望的通道
                t_in = self.cnn_to_trans_proj[i](cnn_out)
                x = self.trans_downsample_layers[i](t_in)
                x = self.stages[i](x)

                if self.use_fam:
                    x, cnn_out = self.FAM[i](x, cnn_out)
                trans_features.append(x)
                cnn_features.append(cnn_out)

        else:
            # ══════════ T+C：Transformer 先跑，投影后喂给 CNN ══════════
            trans_out = x
            for i in range(4):
                trans_out = self.trans_downsample_layers[i](trans_out)
                trans_out = self.stages[i](trans_out)
                # 投影 Transformer 输出 → CNN stage 期望的通道
                c_in = self.trans_to_cnn_proj[i](trans_out)
                cnn_out = self.cnn_downsample_layers[i](c_in)

                if self.use_fam:
                    trans_out, cnn_out = self.FAM[i](trans_out, cnn_out)
                trans_features.append(trans_out)
                cnn_features.append(cnn_out)

        # ── 跨层融合 VFM（与并行版逻辑一致）──────────────────────────
        for i in range(3):
            trans_feat = trans_features[i]
            cnn_feat = cnn_features[i]
            if self.cross_stage_fusion_mode == 'none':
                fused_features.append(
                    self.fusion[i](torch.cat((trans_feat, cnn_feat), dim=1)))
                continue
            target_size = trans_features[i].shape[2:]
            if self.cross_stage_fusion_mode in {'gate', 'gate_concat', 'cross_gate'}:
                trans_proj = self.trans_conv[i](trans_features[i + 1])
                cnn_proj = self.cnn_conv[i](cnn_features[i + 1])
                trans_mask = self.sigmoid(self.trans_bn[i](trans_proj))
                cnn_mask = self.sigmoid(self.cnn_bn[i](cnn_proj))
                trans_scale = 2.0 * self.trans_gate_scale[i].sigmoid()
                cnn_scale = 2.0 * self.cnn_gate_scale[i].sigmoid()
                trans_mask = F.interpolate(trans_mask, size=target_size,
                                           mode='bilinear', align_corners=False)
                cnn_mask = F.interpolate(cnn_mask, size=target_size,
                                         mode='bilinear', align_corners=False)
                if self.cross_stage_fusion_mode == 'cross_gate':
                    trans_feat = trans_feat * (1 + trans_scale * (cnn_mask - 0.5))
                    cnn_feat = cnn_feat * (1 + cnn_scale * (trans_mask - 0.5))
                else:
                    trans_feat = trans_feat * (1 + trans_scale * (trans_mask - 0.5))
                    cnn_feat = cnn_feat * (1 + cnn_scale * (cnn_mask - 0.5))
            if self.cross_stage_fusion_mode in {'concat', 'gate_concat', 'cross_concat'}:
                if self.cross_stage_fusion_mode in {'concat', 'cross_concat'}:
                    trans_proj = self.trans_conv[i](trans_features[i + 1])
                    cnn_proj = self.cnn_conv[i](cnn_features[i + 1])
                trans_high = F.interpolate(trans_proj, size=target_size,
                                           mode='bilinear', align_corners=False)
                cnn_high = F.interpolate(cnn_proj, size=target_size,
                                         mode='bilinear', align_corners=False)
                if self.cross_stage_fusion_mode == 'cross_concat':
                    trans_feat = self.trans_cross_stage_fusion[i](
                        torch.cat((trans_feat, cnn_high), dim=1))
                    cnn_feat = self.cnn_cross_stage_fusion[i](
                        torch.cat((cnn_feat, trans_high), dim=1))
                else:
                    trans_feat = self.trans_cross_stage_fusion[i](
                        torch.cat((trans_feat, trans_high), dim=1))
                    cnn_feat = self.cnn_cross_stage_fusion[i](
                        torch.cat((cnn_feat, trans_high), dim=1))
            fused_features.append(
                self.fusion[i](torch.cat((trans_feat, cnn_feat), dim=1)))

        fused_features.append(
            self.fusion[3](torch.cat((trans_features[3], cnn_features[3]), dim=1)))

        for i in range(4):
            out.append(self.extra_norms[i](fused_features[i]))

        if vis_enabled:
            self._feature_vis_saved = True
        return tuple(out)

    def forward(self, x: torch.Tensor):
        return self.forward_features(x)
