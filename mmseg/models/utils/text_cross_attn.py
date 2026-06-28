import torch
import torch.nn as nn
import torch.nn.functional as F


class TextCrossAttention(nn.Module):
    """Visual-text cross-attention for direct feature injection.

    Visual features as Q, text prototypes as K/V.
    Produces text-aware visual features via gated residual.

    Args:
        visual_dim (int): Visual feature dimension.
        text_dim (int): Text embedding dimension. Default: 512
        num_heads (int): Number of attention heads. Default: 8
    """

    def __init__(self, visual_dim, text_dim=512, num_heads=8):
        super().__init__()
        self.visual_dim = visual_dim
        self.text_dim = text_dim
        self.num_heads = num_heads
        self.head_dim = visual_dim // num_heads

        # Project text to visual space for K, V
        self.text_proj_k = nn.Linear(text_dim, visual_dim)
        self.text_proj_v = nn.Linear(text_dim, visual_dim)

        # Visual Q projection
        self.visual_q = nn.Linear(visual_dim, visual_dim)

        # Output projection
        self.out_proj = nn.Linear(visual_dim, visual_dim)

        # LayerNorm
        self.norm = nn.LayerNorm(visual_dim)

        # Gate: sigmoid(-2) ≈ 0.12, starts with mostly original features
        self.gate = nn.Parameter(torch.tensor([-2.0]))

    def forward(self, visual_features, text_prototypes):
        """
        Args:
            visual_features: [B, H, W, C] backbone features (NHWC)
            text_prototypes: [K, D] category prototypes

        Returns:
            enhanced_visual: [B, H, W, C] text-aware visual features
        """
        B, H, W, C = visual_features.shape
        K, D = text_prototypes.shape

        # Reshape visual to tokens: [B, H*W, C]
        v_tokens = visual_features.reshape(B, H * W, C)

        # Project
        q = self.visual_q(v_tokens)  # [B, H*W, C]
        k = self.text_proj_k(text_prototypes)  # [K, C]
        v = self.text_proj_v(text_prototypes)  # [K, C]

        # Multi-head: [B, heads, N, head_dim]
        q = q.view(B, H * W, self.num_heads, self.head_dim).permute(0, 2, 1, 3)
        k = k.view(K, self.num_heads, self.head_dim).permute(1, 0, 2).unsqueeze(0).expand(B, -1, -1, -1)
        v = v.view(K, self.num_heads, self.head_dim).permute(1, 0, 2).unsqueeze(0).expand(B, -1, -1, -1)

        # Cross-attention
        attn = torch.matmul(q, k.transpose(-2, -1)) * (self.head_dim ** -0.5)
        attn = F.softmax(attn, dim=-1)
        out = torch.matmul(attn, v)  # [B, heads, H*W, head_dim]

        # Reshape back: [B, H*W, C]
        out = out.permute(0, 2, 1, 3).reshape(B, H * W, C)
        out = self.out_proj(out)

        # Gated residual
        gate = torch.sigmoid(self.gate)
        enhanced = self.norm(v_tokens + gate * out)

        return enhanced.reshape(B, H, W, C)
