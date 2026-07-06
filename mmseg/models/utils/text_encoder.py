import torch
import torch.nn as nn
import torch.nn.functional as F


class TextEncoder(nn.Module):
    """CLIP Text Encoder wrapper for category-aware prompt encoding.

    Encodes prompt bank into text embeddings, applies attention pooling
    to aggregate per-category prototypes, and optionally adds learnable
    RepRTA-style refinement.
    """

    def __init__(self, embed_dim=512, num_categories=3, prompts_per_category=10,
                 use_reprta=True, reprta_ffn_type='swiglu',
                 reprta_zero_init=True, use_visual_delta=False):
        super().__init__()
        self.embed_dim = embed_dim
        self.num_categories = num_categories
        self.prompts_per_category = prompts_per_category

        # Per-category learnable query for attention pooling
        self.attn_pool_query = nn.Parameter(
            torch.randn(num_categories, 1, embed_dim) * 0.02)

        # RepRTA-style refinement (FFN + residual)，受三开关控制以支持消融：
        #   use_reprta      : R0(=False 训期关闭) vs R2(=True 默认)
        #   reprta_ffn_type : 'swiglu'(门控，R2) vs 'gelu'(普通 FFN，R1)
        #   reprta_zero_init: w3 零初始化保护 CLIP 原型(R2=True) vs 随机初始化(R3=False)
        self.use_reprta = use_reprta
        self.reprta_ffn_type = reprta_ffn_type
        if use_reprta:
            if reprta_ffn_type == 'swiglu':
                # SwiGLU：升 4× 后切两半，w3 输入维 = (4/2)*D = 2D
                self.reprta_w12 = nn.Linear(embed_dim, 4 * embed_dim)
                self.reprta_w3 = nn.Linear(2 * embed_dim, embed_dim)
            elif reprta_ffn_type == 'gelu':
                # 普通 FFN：升 2× 后 GELU 再降回，参数量与 SwiGLU 对齐
                # （SwiGLU w12=4D, w3 入=2D; GELU w12=2D 直接 == 2D 故两者
                #   总参数量同为 D*2D + 2D*D，控制参数量后才能干净比门控贡献）
                self.reprta_w12 = nn.Linear(embed_dim, 2 * embed_dim)
                self.reprta_w3 = nn.Linear(2 * embed_dim, embed_dim)
            else:
                raise ValueError(f"Unsupported reprta_ffn_type: {reprta_ffn_type}")
            if reprta_zero_init:
                nn.init.zeros_(self.reprta_w3.weight)
                nn.init.zeros_(self.reprta_w3.bias)

        # 类激活视觉提示 -> 文本原型增量。只在 V2 中启用，避免旧消融
        # 统计到未使用参数；最后一层零初始化，初始等价于原文本原型。
        self.use_visual_delta = use_visual_delta
        if use_visual_delta:
            self.visual_delta_norm = nn.LayerNorm(embed_dim)
            self.visual_delta_proj = nn.Linear(embed_dim, embed_dim)
            nn.init.zeros_(self.visual_delta_proj.weight)
            nn.init.zeros_(self.visual_delta_proj.bias)

        self._fused = False

        # Precomputed prompt embeddings buffer (filled at init or loaded)
        self.register_buffer(
            'prompt_embeddings',
            torch.zeros(num_categories, prompts_per_category, embed_dim))

    def _reprta_refine(self, pooled):
        """RepRTA 残差精炼，受 use_reprta/reprta_ffn_type 控制；fuse() 后整体跳过。

        Args:
            pooled (Tensor): [..., D] 注意力池化结果，最后一维为 embed_dim。

        Returns:
            Tensor: [..., D] 精炼后的原型（未做 L2 归一化，由调用方处理）。
        """
        if not self.use_reprta or self._fused:
            return pooled
        x12 = self.reprta_w12(pooled)              # SwiGLU:[...,4D] / GELU:[...,2D]
        if self.reprta_ffn_type == 'swiglu':
            x1, x2 = x12.chunk(2, dim=-1)           # 各 [...,2D]
            hidden = F.silu(x1) * x2                # 门控
        else:  # 'gelu'
            hidden = F.gelu(x12)                    # 普通 FFN
        refined = self.reprta_w3(hidden)           # [...,D]
        return pooled + refined

    def load_prompt_bank(self, path, category_order=None):
        """Load prompt embeddings from .pt file.

        Automatically adapts attn_pool_query to match the loaded
        prompt bank's category count, so no manual config changes
        needed when switching datasets.
        """
        data = torch.load(path, map_location='cpu')
        if isinstance(data, dict):
            embeddings = data['embeddings']
            categories = data.get('categories', None)
        else:
            embeddings = data
            categories = None

        if category_order is not None:
            if categories is None:
                raise ValueError(
                    'prompt_category_order requires prompt bank metadata '
                    '"categories", but this prompt bank only stores a tensor.')
            missing = [cat for cat in category_order if cat not in categories]
            if missing:
                raise ValueError(
                    f'prompt_category_order contains categories not found in '
                    f'prompt bank: {missing}. Available: {categories}')
            order = [categories.index(cat) for cat in category_order]
            embeddings = embeddings[order]

        C, K, D = embeddings.shape
        # Auto-resize query if category count differs
        if C != self.attn_pool_query.shape[0]:
            self.attn_pool_query = nn.Parameter(
                torch.randn(C, 1, D) * 0.02)
            self.num_categories = C
        self.prompts_per_category = K
        self.register_buffer('prompt_embeddings', embeddings)

    def _maybe_augment_prompts(self, prompts):
        """Training-only prompt augmentation shared by both forward paths.

        - Randomly sample 1..K shared prompt positions across categories.
        - Add small Gaussian noise to the embeddings.

        Args:
            prompts (Tensor): [C, K, D].

        Returns:
            Tensor: [C, K', D] with K' <= K.
        """
        C, K, D = prompts.shape
        if self.training and K > 1:
            num_sample = torch.randint(1, K + 1, (1,)).item()
            indices = torch.randperm(K, device=prompts.device)[:num_sample]
            prompts = prompts[:, indices, :]        # [C, num_sample, D]
        if self.training:
            noise = torch.randn_like(prompts) * 0.01
            prompts = prompts + noise
        return prompts

    def forward(self):
        """Forward pass to produce category prototypes.

        Returns:
            category_prototypes: [num_categories, embed_dim]
        """
        # prompt_embeddings: [C, K, D]
        C, K, D = self.prompt_embeddings.shape

        prompts = self._maybe_augment_prompts(self.prompt_embeddings)

        # Attention pooling across prompts within each category
        q = self.attn_pool_query.expand(C, -1, -1)  # [C, 1, D]
        k = v = prompts                              # [C, K', D]

        # Attention: [C, 1, K']
        attn = torch.bmm(q, k.transpose(-2, -1)) * (D ** -0.5)
        attn = F.softmax(attn, dim=-1)

        # Weighted aggregation: [C, 1, D] -> [C, D]
        pooled = torch.bmm(attn, v).squeeze(1)

        # RepRTA refinement (FFN + residual), skipped after fuse() or when disabled
        pooled = self._reprta_refine(pooled)

        # L2 normalize
        category_prototypes = F.normalize(pooled, dim=-1, p=2)

        return category_prototypes

    def prompt_bank_tensor(self):
        """Return the raw frozen CLIP prompt embeddings for backbone injection.

        Returns:
            Tensor: [C, K, D] = [num_categories, prompts_per_category, embed_dim].
                This is the offline CLIP-encoded prompt bank, never re-encoded
                at runtime. backbone injects reshape(-1, D) = [C*K, D] of it.
        """
        return self.prompt_embeddings

    def adapt_with_visual_prompt(self, visual_prompt, delta_scale=0.1):
        """Use per-class visual prompts to lightly adapt text prototypes.

        Args:
            visual_prompt (Tensor): [B, C, D] class-wise visual summaries.
            delta_scale (float | Tensor): Scale for the visual residual.

        Returns:
            tuple:
                adapted_prototypes: [B, C, D]
                base_prototypes: [C, D]
        """
        if not self.use_visual_delta:
            raise RuntimeError(
                'TextEncoder.use_visual_delta must be True for '
                'adapt_with_visual_prompt().')
        base_prototypes = self.forward()                         # [C, D]
        delta = self.visual_delta_proj(
            self.visual_delta_norm(visual_prompt))               # [B, C, D]
        adapted = base_prototypes.unsqueeze(0) + delta_scale * delta
        adapted = F.normalize(adapted, dim=-1, p=2)
        return adapted, base_prototypes

    def fuse(self):
        """Fuse RepRTA for deployment. After fusion, forward() skips RepRTA
        and returns attention-pooled embeddings directly."""
        self._fused = True
