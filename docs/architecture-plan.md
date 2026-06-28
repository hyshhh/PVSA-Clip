# 水域语义分割 — CLIP 文本注入 PVSA-Net 架构设计方案

---

## Core Architecture Design

针对水域语义分割任务，定义类别集合：water、ship、land

每个类别构建 **Category-aware Prompt Set**，用于增强语义表达能力与鲁棒性：
- Water：river、lake、sea、ocean、wave、water surface、water reflection、flood、stream、reservoir
- Ship：boat、vessel、cargo ship、fishing boat、yacht、sailboat、canoe、barge、ship、small boat
- Land：shore、coast、vegetation、road、bridge、building、sky、tree、sand、grass

**离线编码（只做一次）：** 30 条 prompt 文本通过冻结 CLIP Text Encoder 批量编码为 `[30, 512]` 的 L2 归一化嵌入，按类别整理为 `[3, 10, 512]` 张量保存为 `.pt` 文件。训练和推理时不再需要 CLIP 模型。

**部署阶段：** 完全移除 Text Encoder 与 Prompt 流程，仅保留 category prototypes 与分类权重 W_c，实现**零语言计算开销推理**。

---

## 模块创新

### 创新 1：可学习 Attention Pooling + RepRTA Prototype 聚合

**设计动机：** CLIP embedding 冻结后，如何把每个类别 10 个 prompt 聚合成 1 个代表性向量？

YOLOE 训练时每个类别随机采样 1 个 prompt 直接使用，无聚合过程。本方案引入可学习 attention pooling，每个类别 10 个 prompt 全部参与加权聚合，再经 SwiGLU FFN 精炼。

**Attention Pooling：**
```
q = learnable_query[1, 512]           # 可学习查询向量
k = v = prompt_embeddings[10, 512]    # 10 个 prompt 嵌入
attn = softmax(q @ k.T / sqrt(512))  # [1, 10] 注意力权重
prototype = attn @ v                  # [512] 加权聚合结果
```

可学习 query 自动学习每个 prompt 的重要性权重。例如 water 类中 "river" 和 "lake" 语义代表性强，权重高；"harbor" 兼具 water 和 land 语义，权重相对低。

**RepRTA 精炼：**
```
SwiGLU FFN：x12 = Linear(prototype) → chunk → SiLU(x1) × x2 → Linear → refined
残差：prototype = prototype + refined
L2 归一化 → category_prototype [512]
```

**训练增强：** 每次前向随机采样 2-3 个 prompt + 高斯噪声（σ=0.01），防止过拟合。

**从 Prototype 到卷积核：** 3 个类别 `[3, 512]` × head proj 权重 `[512, 256]` = Conv2d 权重 `[3, 256, 1, 1]`。推理时等价于固定 1×1 卷积。

---

### 创新 2：Text-guided Top-P Routing Module（TTRM）

**设计动机：** 原 PVSA-Net 的 Top-P Routing 仅基于视觉 token 相似度做路由，无法感知语义类别。TTRM 在路由器中注入文本亲和力信号，让语义相关的窗口优先被选中聚合。

**作用位置：** Backbone **全部 4 个阶段**的 ToppAttention 内部路由器

**原理：** 输入图像被切成 7×7=49 个窗口，路由器决定"每个窗口应该和哪些其他窗口做注意力"。原版只看视觉特征像不像，不知道"这个窗口是水还是地"。

**两层文本注入机制：**

**第一层 — 路由分数混合（选窗口）：**
```
attn_visual = q_norm @ k_norm.T / sqrt(D)     # [49, 49] 窗口间视觉相似度
attn_text = q_norm @ T_c.T / sqrt(D)         # [49, 3] 每个窗口与 3 个类别的相似度
attn_text_mean = attn_text.mean(dim=-1)        # [49] 平均文本亲和力标量
attn = (1-α) * attn_visual + α * attn_text_mean # [49, 49] 广播到所有 key
```

α = sigmoid(learnable_parameter)，初始值约 0.1。

**第二层 — Soft KV 加权（调强度）：**
```
原版：kv_selected = gather(kv, route_indices)
新版：kv_selected = (1 + soft_kv_weight * r_weight) * gather(kv, route_indices)
```

soft_kv_weight 控制路由分数对 KV 的调制强度（0=纯 gather，0.5=半强度，1=原版 soft routing）。残差形式保证退化安全。

**两层的区别：** 第一层影响"选哪些窗口"，第二层影响"选中的窗口多大程度参与注意力"。

**部署融合：** α 烘焙为常数，soft_kv_weight 固定为配置值。两者都不增加推理开销。

---

### 创新 3：Category-aware Prompt Fusion Module（CPFM）

**设计动机：** CLIP 原始 text embedding 仅在纯文本空间编码，缺乏视觉上下文。CPFM 在训练阶段用 backbone 视觉特征作为 KV 来精炼 text embedding，使 category prototype 携带视觉先验。

**作用位置：** Stage 3 和 Stage 4 的 ToppAttention 输出之后

**核心机制（Text-as-Query Cross-Attention）：**
- **Query = category prototype T_c** `[B, K, D]`
- **Key / Value = backbone 视觉特征 F_v** `[B, H*W, C]`
- Cross-Attention 输出视觉增强的 text embedding T_c'
- **Gated residual + LayerNorm：** `T_enhanced = LayerNorm(T_c + sigmoid(gate) * T_c')`（gate 初始化 sigmoid(-5)≈0）
- Stage 3 和 Stage 4 的 T_enhanced **concat 后经 MLP 聚合**为最终 category prototype

**训练完成后：** 固化 T_enhanced 为 `.pt`，移除 CPFM，推理零开销。

**参数量：** 每个 CPFM ~0.15M，Stage3+Stage4 共 ~0.3M。

---

## 训练策略与损失函数

### 端到端单阶段训练

- **冻结：** CLIP Text Encoder
- **训练：** Backbone + CPFM + TTRM(α) + SegHead 一次性端到端
- 训练完成后固化 CPFM 输出为离线 embedding，移除 CPFM 模块

### 损失函数

```
L = L_seg（= L_cls）
```

| 损失 | 惩罚什么 | 作用 |
|------|---------|------|
| **L_seg** | 像素分类是否正确 | 基础分割监督（CrossEntropyLoss） |
| **L_cls** | 正确类别相对排序是否最高 | 让视觉特征和文字 prototype 对齐（与 L_seg 合一） |

CPFM 的文本增强效果通过 L_cls 的任务损失隐式约束，无需显式对齐损失。

**L_cls 原理：** 用冻结的文字 prototype 替代传统 `nn.Linear` 分类头。每个像素 512 维视觉特征与 3 个类别 prototype 逐一点积，再过 CrossEntropyLoss。只更新图像侧参数，文字 prototype 零梯度。

```python
x = BN(visual_features)
logits = einsum("bchw,bkc->bkhw", x, prototypes) * scale + bias
loss = CrossEntropyLoss(logits, gt_labels)
```

### Prompt Bank 增强策略

- 训练时每个类别**随机采样 2-3 个 prompt**（非全部使用）
- 对 text embedding 施加**高斯噪声扰动**（σ=0.01）
- 语义分割场景所有像素都有明确类别标签，无需负样本 prompt bank

---

## 部署阶段重参数化

**Step 1：** Text Encoder 移除 — 所有 text embedding 预计算并固化

**Step 2：** RepRTA 跳过 — 设置 `_fused=True` 标志，forward 跳过 RepRTA 计算

**Step 3：** 分类头融合 — BN + einsum + scale + bias 融合进 1x1 Conv2d

**Step 4：** TTRM α 融合 — α 烘焙为常数

**Step 5：** CPFM 移除 — 删除 CPFM 模块

**最终模型等价于原始 PVSA-Net + 一个 1x1 Conv 分类头，零文本计算开销。**
