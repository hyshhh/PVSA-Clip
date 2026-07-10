import torch
import torch.nn as nn
import torch.nn.functional as F

from .backbone_text import split_backbone_text_inputs


class TextCrossAttention(nn.Module):
    """Visual-text cross-attention for direct feature injection."""

    def __init__(self, visual_dim, text_dim, num_heads=8):
        super().__init__()
        self.visual_dim = visual_dim
        self.text_dim = text_dim
        self.num_heads = num_heads
        self.head_dim = visual_dim // num_heads

        self.text_proj_k = nn.Linear(text_dim, visual_dim)
        self.text_proj_v = nn.Linear(text_dim, visual_dim)
        self.visual_q = nn.Linear(visual_dim, visual_dim)
        self.out_proj = nn.Linear(visual_dim, visual_dim)
        self.norm = nn.LayerNorm(visual_dim)
        self.gate = nn.Parameter(torch.tensor([-2.0]))

    def forward(self, visual_features, text_prototypes):
        _, align_text = split_backbone_text_inputs(text_prototypes)
        text_prototypes = align_text
        if text_prototypes is None:
            return visual_features

        B, H, W, C = visual_features.shape
        K, _ = text_prototypes.shape
        v_tokens = visual_features.reshape(B, H * W, C)
        q = self.visual_q(v_tokens)

        has_frozen = hasattr(self, '_frozen_k') and self._frozen_k is not None
        if has_frozen:
            k = self._frozen_k
            v = self._frozen_v
        else:
            k = self.text_proj_k(text_prototypes)
            v = self.text_proj_v(text_prototypes)

        q = q.view(B, H * W, self.num_heads, self.head_dim).permute(0, 2, 1, 3)
        k = k.view(K, self.num_heads, self.head_dim).permute(1, 0, 2).unsqueeze(0).expand(B, -1, -1, -1)
        v = v.view(K, self.num_heads, self.head_dim).permute(1, 0, 2).unsqueeze(0).expand(B, -1, -1, -1)

        attn = torch.matmul(q, k.transpose(-2, -1)) * (self.head_dim ** -0.5)
        attn = F.softmax(attn, dim=-1)
        out = torch.matmul(attn, v)

        out = out.permute(0, 2, 1, 3).reshape(B, H * W, C)
        if not has_frozen:
            out = self.out_proj(out)

        gate = torch.sigmoid(self.gate)
        enhanced = self.norm(v_tokens + gate * out)
        return enhanced.reshape(B, H, W, C)

    @torch.no_grad()
    def freeze_for_deployment(self, text_prototypes):
        _, align_text = split_backbone_text_inputs(text_prototypes)
        if align_text is None:
            return
        k = self.text_proj_k(align_text)
        v = self.text_proj_v(align_text)
        v_weight = self.out_proj.weight.data
        v_bias = self.out_proj.bias.data
        self._frozen_k = k
        self._frozen_v = F.linear(v, v_weight, v_bias)
