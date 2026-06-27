import torch
import torch.nn as nn
import torch.nn.functional as F


class TextEncoder(nn.Module):
    """CLIP Text Encoder wrapper for category-aware prompt encoding.

    Encodes prompt bank into text embeddings, applies attention pooling
    to aggregate per-category prototypes, and optionally adds learnable
    RepRTA-style refinement.
    """

    def __init__(self, embed_dim=512, num_categories=3, prompts_per_category=10):
        super().__init__()
        self.embed_dim = embed_dim
        self.num_categories = num_categories
        self.prompts_per_category = prompts_per_category

        # Learnable attention pooling: query token for aggregating
        # multiple prompts within each category
        self.attn_pool_query = nn.Parameter(
            torch.randn(1, 1, embed_dim) * 0.02)

        # RepRTA-style refinement (SwiGLU FFN + residual)
        self.reprta_w12 = nn.Linear(embed_dim, 4 * embed_dim)
        self.reprta_w3 = nn.Linear(2 * embed_dim, embed_dim)
        nn.init.zeros_(self.reprta_w3.weight)
        nn.init.zeros_(self.reprta_w3.bias)

        self._fused = False

        # Precomputed prompt embeddings buffer (filled at init or loaded)
        self.register_buffer(
            'prompt_embeddings',
            torch.zeros(num_categories, prompts_per_category, embed_dim))

    def set_prompt_embeddings(self, embeddings):
        """Set precomputed CLIP prompt embeddings.

        Args:
            embeddings: [num_categories, prompts_per_category, embed_dim]
        """
        assert embeddings.shape == self.prompt_embeddings.shape
        self.prompt_embeddings.copy_(embeddings)

    def load_prompt_bank(self, path):
        """Load prompt embeddings from .pt file."""
        data = torch.load(path, map_location='cpu')
        if isinstance(data, dict):
            embeddings = data['embeddings']
        else:
            embeddings = data
        self.set_prompt_embeddings(embeddings)

    def forward(self):
        """Forward pass to produce category prototypes.

        Returns:
            category_prototypes: [num_categories, embed_dim]
        """
        # prompt_embeddings: [C, K, D]
        C, K, D = self.prompt_embeddings.shape

        prompts = self.prompt_embeddings

        # Training augmentation: random sample 2-3 prompts per category
        if self.training and K > 3:
            num_sample = torch.randint(2, 4, (1,)).item()
            indices = torch.randperm(K, device=prompts.device)[:num_sample]
            prompts = prompts[:, indices, :]  # [C, num_sample, D]

        # Training augmentation: Gaussian noise on embeddings
        if self.training:
            noise = torch.randn_like(prompts) * 0.01
            prompts = prompts + noise

        # Attention pooling across prompts within each category
        q = self.attn_pool_query.expand(C, -1, -1)  # [C, 1, D]
        k = v = prompts  # [C, K', D]

        # Attention: [C, 1, K']
        attn = torch.bmm(q, k.transpose(-2, -1)) * (D ** -0.5)
        attn = F.softmax(attn, dim=-1)

        # Weighted aggregation: [C, 1, D] -> [C, D]
        pooled = torch.bmm(attn, v).squeeze(1)

        # RepRTA refinement (SwiGLU + residual), skipped after fuse()
        if not self._fused:
            x12 = self.reprta_w12(pooled)
            x1, x2 = x12.chunk(2, dim=-1)
            hidden = F.silu(x1) * x2
            refined = self.reprta_w3(hidden)
            pooled = pooled + refined

        # L2 normalize
        category_prototypes = F.normalize(pooled, dim=-1, p=2)

        return category_prototypes

    def fuse(self):
        """Fuse RepRTA for deployment. After fusion, forward() skips RepRTA
        and returns attention-pooled embeddings directly."""
        self._fused = True
