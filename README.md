# PVSA-Net

基于 MMSegmentation 的语义分割项目，支持两条架构路径：

1. **PVSA-Net（原始）**：纯视觉 Top-P 路由注意力 + CNN 双流融合。
2. **PVSA-CLIP（新增）**：在 PVSA-Net 基础上注入 CLIP 文本语义，支持开放词汇分割，部署时零文本计算开销。

---

## 架构概览

### PVSA-CLIP 核心模块

```
┌─────────────────────────────────────────────────────────┐
│  CLIPEncoderDecoder                                      │
│  ┌──────────────┐   ┌─────────────────────────────────┐ │
│  │ TextEncoder   │   │  BiFormer_fusion (Backbone)     │ │
│  │  Prompt Bank  │──→│  ┌─────────────────────────┐    │ │
│  │  AttnPool     │   │  │ Stage 1-4               │    │ │
│  │  RepRTA       │   │  │  ┌──────────────────┐   │    │ │
│  └──────────────┘   │  │  │ TTRM (文本路由)   │   │    │ │
│                      │  │  │ α * attn_text     │   │    │ │
│                      │  │  │ + (1-α) * visual  │   │    │ │
│                      │  │  └──────────────────┘   │    │ │
│                      │  │  ┌──────────────────┐   │    │ │
│                      │  │  │ CPFM (Stage3/4)  │   │    │ │
│                      │  │  │ Text→Q, Vis→KV   │   │    │ │
│                      │  │  │ 训练后固化移除    │   │    │ │
│                      │  │  └──────────────────┘   │    │ │
│                      │  └─────────────────────────┘    │ │
│                      └─────────────────────────────────┘ │
│  ┌─────────────────────────────────────────────────────┐ │
│  │  CLIPSegHead                                        │ │
│  │  多尺度融合 → BN → proj → cosine(text_prototypes)   │ │
│  │  部署时融合为单个 Conv2d                             │ │
│  └─────────────────────────────────────────────────────┘ │
└─────────────────────────────────────────────────────────┘
```

| 模块 | 作用 | 推理开销 |
|------|------|---------|
| **TTRM** | 文本引导 Top-P 路由，语义相关 token 优先聚合 | α 融合进常数，零开销 |
| **CPFM** | 训练时用视觉特征增强 text embedding | 固化为 .pt 后移除 |
| **CLIPSegHead** | 对比分类，cosine 相似度替代 softmax | 融合为 Conv2d |
| **TextEncoder** | Prompt 编码 + attention pooling + RepRTA | 移除 |

---

## 损失函数设计

### 总损失

```
L = L_seg + λ_cls · L_cls + λ_align · L_align + λ_soft · L_soft
```

| 损失 | 权重 | 作用 | 相比原始 PVSA-Net 的改进 |
|------|------|------|------------------------|
| **L_seg** | 1.0 | 标准 CrossEntropyLoss 分割损失 | 无变化，保持原有分割监督 |
| **L_cls** | 内置 | CLIPSegHead 中 cosine similarity 的 BCE 损失 | **新增**：替代传统 softmax 分类，引入文本语义约束 |
| **L_align** | 0.1 | CPFM 增强后 prototype 与原始 CLIP embedding 的余弦相似度约束 | **新增**：防止 CPFM 过度偏移原始语义 |
| **L_soft** | 0.1 | MedCLIPSeg 风格的双向软对比损失 | **新增**：patch 级跨模态对齐 |

### L_cls — 对比分类损失

**原理：** CLIPSegHead 不使用传统的 `nn.Linear(C, K)` 分类头，而是将视觉特征投影到 512 维后，与 category prototype 计算 cosine similarity：

```python
# clip_seg_head.py
seg_logits = einsum("bchw,bkc->bkhw", visual_feat, text_prototypes)
seg_logits = seg_logits * exp(logit_scale) + bias
loss = CrossEntropyLoss(seg_logits, gt_labels)
```

**为什么用 cosine similarity 替代 softmax 分类：**
- 原始 PVSA-Net 的 `SegformerHead.cls_seg` 是纯视觉分类头，无法利用语义先验
- cosine similarity 将分类问题转化为视觉-文本对齐问题，类别间的关系由 CLIP 语义空间定义
- 新增类别时只需添加 prompt embedding，无需重新训练分类头
- `logit_scale`（初始值 exp(-1)≈0.37）和 `bias`（初始值 -10）参考 YOLOE 的 BNContrastiveHead，训练时自动学习

### L_align — 嵌入正则化损失

```python
# clip_encoder_decoder.py
original_prototypes = text_encoder()  # CLIP 原始 embedding
cos_sim = cosine_similarity(enhanced_prototypes, original_prototypes)
loss_align = 0.1 * (1 - cos_sim).mean()
```

**为什么需要：**
- CPFM 用视觉特征增强 text embedding，增强后的 prototype 可能偏离 CLIP 原始语义空间
- L_align 约束增强后的 embedding 与原始 CLIP embedding 保持方向一致性
- 类似于知识蒸馏中的 "teacher loss"，原始 CLIP embedding 充当教师信号
- 如果去掉 L_align，增强后的 prototype 可能过拟合训练集视觉特征，降低零样本泛化能力

### L_soft — 软对比损失

```python
# clip_encoder_decoder.py
patch_embed = avg_pool(feats[-1])  # [B, C]
text_embed = category_prototypes[:num_categories]  # [K, D]

# 文本相似度生成软目标（非 one-hot）
G = softmax(text_prototypes @ text_prototypes.T / τ)  # τ=0.2

# 双向软交叉熵
loss_t2v = -sum(G * log_softmax(similarity_v2t))
loss_v2t = -sum(G.T * log_softmax(similarity_t2v))
loss_soft = 0.1 * 0.5 * (loss_t2v + loss_v2t)
```

**为什么需要：**
- 传统对比学习用 hard target（one-hot），但水域类别间存在语义重叠（如 harbor 既是 water 又接近 land）
- 软目标根据文本相似度自动分配：语义相近的类别获得更均匀的监督信号
- 温度 τ=0.2 控制软目标的平滑程度（参考 MedCLIPSeg 论文）
- 双向对齐确保视觉→文本和文本→视觉两个方向都被约束

---

## 训练增强策略

### 1. Prompt 随机采样

```python
# text_encoder.py forward()
if self.training and K > 3:
    num_sample = randint(2, 4)  # 随机采样 2-3 个 prompt
    indices = randperm(K)[:num_sample]
    prompts = prompt_embeddings[:, indices, :]
```

**改进点：** 原始 YOLOE 使用固定 prompt 集合，每次前向使用全部 10 个 prompt。

**为什么改为随机采样：**
- 迫使 attention pooling 学会对不同子集都能产生稳定的 prototype，提升鲁棒性
- 类似 Dropout 的正则化效果，防止模型依赖特定 prompt 组合
- 实际效果：模型不会过拟合某几个 prompt 的语义表达，泛化到未见过的描述方式

### 2. 文本嵌入高斯噪声

```python
# text_encoder.py forward()
if self.training:
    noise = torch.randn_like(prompts) * 0.01
    prompts = prompts + noise
```

**改进点：** 原始 CLIP embedding 是确定性的，训练时无扰动。

**为什么加噪声：**
- CLIP Text Encoder 对 prompt 措辞敏感（"river" vs "a photo of river" 编码不同）
- 高斯噪声模拟 prompt 表述的不确定性，让模型学到更平滑的语义空间
- σ=0.01 较小，不会破坏语义结构，仅提供轻微正则化
- 类似于 label smoothing 的文本版本

### 3. 负样本 Prompt Bank

**改进点：** 原始方案只有 water/ship/land 三个正类别。

**为什么需要负样本：**
- 水域分割场景中存在大量"非水非船非地"的类别（天空、建筑、车辆等）
- 仅用正类别训练，模型倾向于把所有像素归为三个类别之一
- 负样本提供"都不是"的监督信号，降低误检率
- 参考 YOLOE 的 `global_grounding_neg_cat.json`（3265 个负样本类别）

### 4. 端到端单阶段训练

```python
# 配置：冻结 CLIP Text Encoder，其余全部联合训练
# Backbone + CPFM + TTRM(α) + SegHead 一次性端到端
```

**改进点：** 原始方案设计为两阶段训练（Stage1 冻结 backbone → Stage2 端到端）。

**为什么改为单阶段：**
- CPFM 和 TTRM 都是轻量模块（参数量 < backbone 的 3%），不需要预训练适配
- 单阶段训练更简单，超参更少，实验迭代更快
- TTRM 的 α 和 CPFM 的 gate 都从零开始学习，与 backbone 自然收敛
