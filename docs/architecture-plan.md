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

YOLOE 训练时每个类别随机采样 1 个 prompt 直接使用，无聚合过程。本方案引入**per-category 可学习 attention pooling**，每个类别有独立的 query 向量，学习类别特定的 prompt 加权策略，再经 SwiGLU FFN 精炼。

**Attention Pooling：**
```
q = per_category_query[C, 1, 512]     # 每个类别独立的可学习查询向量
k = v = prompt_embeddings[C, 10, 512]  # 10 个 prompt 嵌入
attn = softmax(q @ k.T / sqrt(512))   # [C, 1, 10] 注意力权重
prototype = attn @ v                   # [C, 512] 加权聚合结果
```

每个类别有独立 query，water 的 query 学到"river、lake 权重高"，ship 的 query 学到"boat、vessel 权重高"，互不干扰。加载 prompt bank 时 query 数量自动适配类别数，换数据集无需改配置。

**训练增强（关键，防止 query 过拟合）：**

每次训练前向，对 prompt 做两步增强后再送入 attention pooling：

1. **随机采样 2-3 个 prompt**：从 10 个 prompt 中随机抽 2-3 个，迫使 query 学会对**任意子集**都能产出稳定的 prototype，不能依赖固定的 10 个 prompt 组合。效果类似 Dropout。
2. **高斯噪声扰动（σ=0.01）**：对采样后的 prompt embedding 加微小噪声，防止 query 过拟合到某个 prompt 的精确向量值，学到的是"方向"而不是"坐标"。

推理时使用全部 10 个 prompt，不加噪声，attention pooling 输出更稳定。

**RepRTA 精炼：**
```
SwiGLU FFN：x12 = Linear(prototype) → chunk → SiLU(x1) × x2 → Linear → refined
残差：prototype = prototype + refined
L2 归一化 → category_prototype [512]
```

RepRTA 权重零初始化，训练初期近似恒等映射，推理时设置 `_fused=True` 跳过计算。

**从 Prototype 到卷积核：** 3 个类别 `[3, 512]` × head proj 权重 `[512, 256]` = Conv2d 权重 `[3, 256, 1, 1]`。推理时等价于固定 1×1 卷积。

---

### 创新 2：Text-guided Top-P Routing Module（TTRM）

**设计动机：** 原 PVSA-Net 的 Top-P Routing 仅基于视觉 token 相似度做路由，无法感知语义类别。TTRM 在路由器中注入文本语义，让路由过程具备类别感知能力。

**作用位置：** Backbone **全部 4 个阶段**的 ToppAttention 内部路由器

**原理：** 输入图像被切成 7×7=49 个窗口，路由器决定"每个窗口应该和哪些其他窗口做注意力"。原版只看视觉特征像不像，不知道"这个窗口是水还是地"。

**核心机制 — 交叉注意力增强 Q：**

在路由计算之前，窗口池化后的视觉 Q 先与文本 prototypes 做一次轻量交叉注意力，将文本语义注入 Q 中，再用增强后的 Q 进行 Top-P 路由：

```
窗口池化后：
  Q_visual = 视觉窗口特征 [49, qk_dim]
  K_text = text_prototypes 投影 [3, qk_dim]     ← 3 个类别 prototype
  V_text = text_prototypes 投影 [3, qk_dim]

交叉注意力：
  cross_attn = softmax(Q_visual @ K_text.T / sqrt(D))  # [49, 3]
  text_info = cross_attn @ V_text                       # [49, qk_dim]

残差注入：
  Q_enriched = normalize(Q_visual + gate × text_info)   # [49, qk_dim]
  gate = sigmoid(learnable), 初始 ≈ 0.12

Top-P 路由：
  attn = Q_enriched @ K_visual.T                         # [49, 49]
  → Top-P 剪枝 → 选 top-k 窗口
```

**效果：** 经过交叉注意力后，水域窗口的 Q 携带了"这是水"的语义信息，路由时水域窗口之间更容易被选中聚合。

**与 SAM3 的区别：** SAM3 在编码器/解码器每层都做视觉-文本交叉注意力（6层×2），计算量大。TTRM 只在路由器的窗口池化后做一次轻量交叉注意力，且 gate 初始值很小（≈0.12），训练初期以原始视觉 Q 为主，逐步注入文本信息。

**部署融合：** TTRM 关闭，走纯视觉路由，零额外开销。

---

### 创新 3：渐进式文本注入 — TTRM + TextCrossAttention

**设计动机：** TTRM 只影响路由选择（哪些窗口被选中），不改变视觉特征本身。深层特征已经具备语义信息，可以直接和文本做交叉注意力来增强特征，让视觉特征本身携带"这是水"的语义。

**渐进式注入策略：**

| Stage | 注入方式 | 作用层面 | 理由 |
|-------|---------|---------|------|
| Stage 0-1（浅层） | TTRM | 路由级 | 浅层特征是纹理/边缘，语义弱，改路由就够了 |
| Stage 2-3（深层） | TTRM + TextCrossAttention | 路由级 + 特征级 | 深层特征有语义，直接融合文本能对齐 |

**TextCrossAttention 模块：**

```
Q = visual_features [B, H*W, C]     ← backbone 深层特征
K = text_prototypes 投影 [K, C]      ← 3 个类别 prototype
V = text_prototypes 投影 [K, C]

cross_attn = softmax(Q @ K.T / sqrt(D))   # [B, H*W, K]
text_info = cross_attn @ V                  # [B, H*W, C]

gate = sigmoid(learnable), 初始 ≈ 0.12
enhanced_visual = LayerNorm(visual + gate × text_info)
```

每个 Block 拥有独立的 TextCrossAttention 实例（不共享），插入在 ToppAttention 输出之后、MLP 之前。通过配置 `cross_attn_stages=[2, 3]` 控制作用阶段。

**与 SAM3 的区别：** SAM3 在编码器全部 6 层都做 cross-attention，计算量大。本方案只在 Stage 2、3 做（共 9 个 Block），且 gate 初始值很小（≈0.12），训练初期以原始视觉特征为主，逐步注入文本信息。

**与 TTRM 的区别：** TTRM 在路由器内部做，影响"选哪些窗口"；TextCrossAttention 在 ToppAttention 之后做，直接修改视觉特征本身。两者互补。

**部署融合：** TextCrossAttention 模块直接删除，走纯视觉前向，零额外开销。

---

### 创新 4：CLIPSegHead — 文本原型驱动的对比分类头

**设计动机：** 普通 SegformerHead 用 `nn.Conv2d(channels, num_classes)` 做分类，权重随机初始化，每个类别之间的关系完全由训练数据隐式学习。CLIPSegHead 用冻结的 CLIP text prototype 替代分类权重，将分类问题转化为"视觉特征和哪个文本原型最像"的对比问题，继承 CLIP 的跨模态语义先验。

**与普通 SegformerHead 的对比：**

| | SegformerHead | CLIPSegHead |
|---|---|---|
| 分类器 | `nn.Conv2d(256, 3)` 随机初始化 | cosine(visual_feat, text_prototypes) 冻结 |
| 类别关系 | 训练后隐式学到 | 由 CLIP 语义空间预定义 |
| 新增类别 | 重新训练分类头 | 只需添加 prompt embedding |
| 梯度 | 流过分类权重 + backbone | 只流过 backbone（text 嵌入冻结） |
| 推理 | 标准 Conv2d | 融合后等价 Conv2d |

**前向计算：**
```
4 stage 特征 → 各 1×1 Conv(→256) → 上采样 → concat → [B, 1024, 64, 64]
→ fusion_conv(1024→256) → BN → proj(256→512) → [B, 512, H, W]
→ einsum("bchw,bkc->bkhw", feat, prototypes[3,512]) × scale + bias → [B, 3, H, W]
```

**损失函数：** 单一 CrossEntropyLoss，logits 来自 cosine similarity × scale + bias。

**部署融合：** BN + proj + einsum + scale + bias 全部折叠进单个 `Conv2d(256, 3, 1×1)`，推理时等价于普通分类头。

---

### Prompt Bank 增强策略

- 训练时每个类别**随机采样 2-3 个 prompt**（非全部使用），防止 attention pooling query 过拟合
- 对 text embedding 施加**高斯噪声扰动**（σ=0.01），模拟 CLIP 编码器的不确定性
- 语义分割场景所有像素都有明确类别标签（water/ship/land），无需负样本 prompt bank

---

## 部署阶段重参数化

**Step 1：** Text Encoder 移除 — 所有 text embedding 预计算并固化

**Step 2：** RepRTA 跳过 — 设置 `_fused=True` 标志，forward 跳过 RepRTA 计算

**Step 3：** 分类头融合 — BN + einsum + scale + bias 融合进 1x1 Conv2d

**Step 4：** TTRM 关闭 + TextCrossAttention 删除 — 走纯视觉路由

**最终模型等价于原始 PVSA-Net + 一个 1x1 Conv 分类头，零文本计算开销。**
