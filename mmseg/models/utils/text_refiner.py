import torch
import torch.nn as nn
import torch.nn.functional as F


class TextRefiner(nn.Module):
    """SwiGLU FFN + 残差，对原始 CLIP 文本嵌入做重构（供 backbone 注入）。

    与 TextEncoder 内的 RepRTA 区别：
    - RepRTA：对**图相关筛选后的 3 条池化原型** [B,3,512] 作用（head 路径）。
    - TextRefiner：对**原始固定 30 条 CLIP 嵌入** [30,512] 作用（backbone 路径），
      **不接图像**，输出固定 30 条，保证 backbone 段可融合成固定 K/V 缓存。

    Args:
        in_dim (int): 文本嵌入维度（与 CLIP embed_dim 对齐）。
        hidden_mult (int): SwiGLU 升维倍率；FFN 内部宽度 = hidden_mult * in_dim，
            切成两段各 (hidden_mult//2) * in_dim。
    """

    def __init__(self, in_dim=512, hidden_mult=4):
        super().__init__()
        self.in_dim = in_dim
        self.hidden_mult = hidden_mult
        self.w12 = nn.Linear(in_dim, hidden_mult * in_dim)
        # SwiGLU 把 hidden 维对半切，后段 Linear 的输入维 = (hidden_mult // 2) * in_dim
        self.w3 = nn.Linear((hidden_mult // 2) * in_dim, in_dim)
        # 零初始化 w3：起步等价于无残差修正，温和引入可学重构
        nn.init.zeros_(self.w3.weight)
        nn.init.zeros_(self.w3.bias)

    def forward(self, x):
        """对固定文本向量做 SwiGLU + 残差重构。

        Args:
            x (Tensor): [N, in_dim]，N 为文本向量数（backbone 用 N=30）。

        Returns:
            Tensor: [N, in_dim]，与输入同形，含残差。
        """
        x12 = self.w12(x)                       # [N, hidden*in_dim]
        x1, x2 = x12.chunk(2, dim=-1)           # 各 [(hidden/2)*in_dim]
        hidden = F.silu(x1) * x2                # SwiGLU 门控
        refined = self.w3(hidden)               # [N, in_dim]
        return x + refined                      # 残差
