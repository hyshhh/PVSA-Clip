import torch
import torch.nn as nn
import torch.nn.functional as F


class CPFM(nn.Module):
    """Category-aware Prompt Fusion Module (Training Only).

    Uses text embeddings as Query, backbone visual features as Key/Value.
    Performs Cross-Attention to refine text embeddings with visual context.
    After training, enhanced embeddings are frozen and saved as .pt file;
    this module is removed during inference (zero overhead).

    Args:
        embed_dim (int): Text embedding dimension. Default: 512
        visual_dim (int): Visual feature dimension from backbone stage.
        num_heads (int): Number of attention heads. Default: 8
        top_m (int): Number of top visual windows to attend to. Default: 8
    """

    def __init__(self, embed_dim=512, visual_dim=256, num_heads=8, top_m=8):
        super().__init__()
        self.embed_dim = embed_dim
        self.visual_dim = visual_dim
        self.num_heads = num_heads
        self.top_m = top_m
        self.head_dim = embed_dim // num_heads

        # Project visual features to text embedding space for K, V
        self.visual_proj_k = nn.Linear(visual_dim, embed_dim)
        self.visual_proj_v = nn.Linear(visual_dim, embed_dim)

        # QKV projection for text (Q from text, K/V from visual)
        self.text_q = nn.Linear(embed_dim, embed_dim)

        # Output projection
        self.out_proj = nn.Linear(embed_dim, embed_dim)

        # LayerNorm
        self.norm_text = nn.LayerNorm(embed_dim)
        self.norm_out = nn.LayerNorm(embed_dim)

        # Gate for residual fusion (sigmoid(-5) ≈ 0.007, near-zero init)
        self.gate = nn.Parameter(torch.tensor([-5.0]))

    def forward(self, text_embeds, visual_features, route_indices=None):
        """Forward pass.

        Args:
            text_embeds: [B, K, D] category prototypes (K = num_categories * prompts)
            visual_features: [B, C, H, W] backbone visual features
            route_indices: [B, P, M] optional TopRouter indices to select
                          top-M visual windows (P = num_windows, M = top_m)

        Returns:
            enhanced_text: [B, K, D] visually-enhanced text embeddings
        """
        B, K, D = text_embeds.shape
        _, C, H, W = visual_features.shape

        # Reshape visual features to window format: [B, P, W2, C]
        # where P = n_win * n_win, W2 = (H/n_win) * (W/n_win)
        # For simplicity, treat each spatial position as a token
        visual_tokens = visual_features.flatten(2).transpose(1, 2)  # [B, H*W, C]

        # Project visual to text space
        k = self.visual_proj_k(visual_tokens)  # [B, H*W, D]
        v = self.visual_proj_v(visual_tokens)  # [B, H*W, D]

        # If route indices provided, select top-M visual tokens
        if route_indices is not None:
            # route_indices: [B, P, M] -> flatten to select tokens
            M = route_indices.shape[-1]
            # Use the first K route indices to select visual tokens
            idx = route_indices[:, :min(K, route_indices.shape[1]), :M]
            # Expand for gathering
            idx_k = idx.unsqueeze(-1).expand(-1, -1, -1, D)  # [B, K, M, D]
            k_selected = torch.gather(
                k.unsqueeze(1).expand(-1, K, -1, -1),  # [B, K, H*W, D]
                dim=2,
                index=idx_k
            )  # [B, K, M, D]
            v_selected = torch.gather(
                v.unsqueeze(1).expand(-1, K, -1, -1),
                dim=2,
                index=idx_k
            )  # [B, K, M, D]
            k = k_selected.reshape(B, K * M, D)
            v = v_selected.reshape(B, K * M, D)

        # Text Q
        q = self.text_q(self.norm_text(text_embeds))  # [B, K, D]

        # Multi-head attention
        q = q.reshape(B, K, self.num_heads, self.head_dim).transpose(1, 2)
        k = k.reshape(B, -1, self.num_heads, self.head_dim).transpose(1, 2)
        v = v.reshape(B, -1, self.num_heads, self.head_dim).transpose(1, 2)

        # Attention
        attn = torch.matmul(q, k.transpose(-2, -1)) * (self.head_dim ** -0.5)
        attn = F.softmax(attn, dim=-1)

        out = torch.matmul(attn, v)  # [B, num_heads, K, head_dim]
        out = out.transpose(1, 2).reshape(B, K, D)
        out = self.out_proj(out)

        # Gated residual
        enhanced_text = self.norm_out(
            text_embeds + torch.sigmoid(self.gate) * out)

        return enhanced_text
