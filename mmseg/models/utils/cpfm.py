import torch
import torch.nn as nn
import torch.nn.functional as F


class CPFM(nn.Module):
    """Category-aware Prompt Fusion Module (Training Only).

    Text embeddings as Q, visual features as K/V.
    Gated residual: original text preserved, scene-specific features injected.
    Enhanced embeddings saved as .pt after training, module removed at inference.

    Args:
        embed_dim (int): Text embedding dimension. Default: 512
        visual_dim (int): Visual feature dimension from backbone stage.
        num_heads (int): Number of attention heads. Default: 8
    """

    def __init__(self, embed_dim=512, visual_dim=256, num_heads=8):
        super().__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads

        # Visual -> text space projections for K, V
        self.visual_proj_k = nn.Linear(visual_dim, embed_dim)
        self.visual_proj_v = nn.Linear(visual_dim, embed_dim)

        # Text Q projection
        self.text_q = nn.Linear(embed_dim, embed_dim)

        # Output projection
        self.out_proj = nn.Linear(embed_dim, embed_dim)

        # LayerNorm
        self.norm_text = nn.LayerNorm(embed_dim)

        # Gate: controls how much scene-specific info to inject
        # sigmoid(-2) ≈ 0.12, starts with mostly original text
        self.gate = nn.Parameter(torch.tensor([-2.0]))

    def forward(self, text_embeds, visual_features):
        """
        Args:
            text_embeds: [K, D] category prototypes (original CLIP embeddings)
            visual_features: [B, C, H, W] backbone visual features
        Returns:
            enhanced_text: [K, D]
        """
        K, D = text_embeds.shape
        B, C, H, W = visual_features.shape

        # Average visual across batch -> [H*W, C]
        visual_avg = visual_features.flatten(2).mean(dim=0).transpose(0, 1)

        # Cross-attention: text Q attends to visual K/V
        k = self.visual_proj_k(visual_avg)  # [H*W, D]
        v = self.visual_proj_v(visual_avg)  # [H*W, D]
        q = self.text_q(self.norm_text(text_embeds))  # [K, D]

        # Multi-head
        q = q.view(K, self.num_heads, self.head_dim).permute(1, 0, 2)
        k = k.view(-1, self.num_heads, self.head_dim).permute(1, 0, 2)
        v = v.view(-1, self.num_heads, self.head_dim).permute(1, 0, 2)

        attn = torch.matmul(q, k.transpose(-2, -1)) * (self.head_dim ** -0.5)
        attn = F.softmax(attn, dim=-1)

        scene_info = torch.matmul(attn, v)  # [heads, K, head_dim]
        scene_info = scene_info.permute(1, 0, 2).reshape(K, D)
        scene_info = self.out_proj(scene_info)

        # Gated residual: original text + gated scene-specific info
        alpha = torch.sigmoid(self.gate)
        enhanced_text = text_embeds + alpha * scene_info

        return enhanced_text
