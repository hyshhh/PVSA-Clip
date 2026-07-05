# 水域语义分割 — CLIP 文本注入 PVSA-Net 架构设计方案

---

## 核心设计

任务类别固定为 `water`、`ship`、`land`。每类维护 10 条类别相关提示词，用冻结的 CLIP 文本塔离线编码一次，得到 L2 归一化的提示词库 `[3, 10, 512]`，训练和推理阶段不再调用 CLIP 文本塔。

- Water：river、lake、sea、ocean、wave、water surface、water reflection、flood、stream、reservoir
- Ship：boat、vessel、cargo ship、fishing boat、yacht、sailboat、canoe、barge、ship、small boat
- Land：shore、coast、vegetation、road、bridge、building、sky、tree、sand、grass

整体采用**双路解耦文本注入**，避免“骨干需要文本原型、文本原型又依赖骨干特征”的循环依赖：

- **backbone 固定文本路**：原始 30 条 CLIP 提示词嵌入 `[30,512]` 经 `TextRefiner` 重构后，作为固定 `backbone_text` 注入 `TTRM` 与 `TextCrossAttention`。该路径与输入图像无关，可在部署时冻结成 K/V 缓存。
- **CLIPSegHead 动态原型路**：骨干多阶段特征先生成图相关 query，再按类别对提示词库做分组注意力池化，得到每张图不同的类别原型 `[B,3,512]`，供解码头逐像素分类。

最终推理形态：backbone 段文本注入可冻结为查表式 K/V；head 段保留图相关原型生成与实时点积分类，不默认折叠成单个 `Conv2d`。

---

## 1. Backbone：固定文本语义注入

backbone 只使用固定的 30 条文本向量，不接收 head 侧动态图相关原型。这样路由级文本注入、特征级文本注入、head 侧图相关原型生成三者互不阻塞。

### 1.1 TextRefiner：固定提示词重构

**作用对象**：原始提示词库 reshape 后的 `[30,512]`。

**动机**：CLIP 文本向量来自通用图文预训练，直接注入水域分割骨干可能存在域偏差。`TextRefiner` 用轻量残差前馈网络对固定文本向量做可学习重构，但不引入图像依赖，保持部署可缓存。

```
x12 = Linear(512 → 4*512)
x1, x2 = chunk(x12)
refined = Linear(SiLU(x1) * x2)
backbone_text = x + refined
```

`w3` 零初始化，训练初期严格近似恒等映射；训练后学到对水域分割更友好的文本方向。该模块和 head 路的 `RepRTA` 结构相似，但参数独立、作用对象不同。

### 1.2 TTRM：路由级文本引导

**作用位置**：Stage 0-2 的 `ToppAttention` 路由器。当前配置 `ttrm_stages=[0,1,2]`，Stage 3 通过 `use_plain_attn_last_stage=True` 改为普通 self-attention，因此最后阶段不使用 Top-P 路由，也不加 TTRM。

**设计动机**：原 PVSA-Net 的 Top-P 路由只看视觉窗口相似度，不知道窗口语义。TTRM 在路由计算前把固定文本语义注入窗口级 Q/K，使“水域窗口更倾向于路由到水域窗口”。

```
窗口池化后：
  Q_visual, K_visual = window tokens            # [49, qk_dim]
  K_text, V_text = proj(backbone_text)          # [30, qk_dim]

文本交叉注意力：
  q_text_attn = softmax(Q_visual @ K_text.T / sqrt(D))
  k_text_attn = softmax(K_visual @ K_text.T / sqrt(D))
  q_text = out_proj(q_text_attn @ V_text)
  k_text = out_proj(k_text_attn @ V_text)

门控残差：
  Q_enriched = Q_visual + sigmoid(gate) * q_text
  K_enriched = K_visual + sigmoid(gate) * k_text

Top-P 路由：
  route_score = Q_enriched @ K_enriched.T
  route_score → Top-P 剪枝 → top-k 窗口
```

`gate` 初始为 `sigmoid(-2)≈0.12`，训练早期以视觉路由为主，逐步引入文本语义。相比在每层做重型视觉-文本交叉注意力，TTRM 只在窗口池化后的路由 token 上计算，开销较小。

### 1.3 TextCrossAttention：特征级文本融合

**作用位置**：Stage 2-3，配置为 `cross_attn_stages=[2,3]`。每个 block 拥有独立的 `TextCrossAttention`，插入在注意力模块输出之后、MLP 之前。

**设计动机**：TTRM 改变“哪些窗口被选中”，但不直接修改视觉特征内容。深层特征已有更强语义表达，适合直接与固定文本做特征级融合。

```
Q = visual_features                    # [B, H*W, C]
K = text_proj_k(backbone_text)          # [30, C]
V = text_proj_v(backbone_text)          # [30, C]

attn = softmax(Q @ K.T / sqrt(D))       # [B, H*W, 30]
text_info = out_proj(attn @ V)          # [B, H*W, C]
enhanced = LayerNorm(visual + sigmoid(gate) * text_info)
```

Stage 2 同时有 TTRM 与 TextCrossAttention：既让路由选择带文本语义，也让深层特征显式吸收文本信息。Stage 3 不再做 TTRM，只保留普通 self-attention + TextCrossAttention，用于最终全局语义整合。

### 1.4 Backbone 部署缓存

`fuse_for_deployment(fuse_head=False)` 中，backbone 段可完全冻结文本计算：

1. `TextRefiner` 对 30 条提示词跑一次，结果写入 `frozen_backbone_text`。
2. `TextCrossAttention.freeze_for_deployment` 预计算 `_frozen_k/_frozen_v [30,C]`，并把 `out_proj` 融入 V。
3. Stage 0-2 的 `TopkRouting` 预计算或首次推理 lazy 注册 `_frozen_tc_k/_frozen_tc_v [30,qk_dim]`。
4. 推理时 backbone 不再执行文本塔或文本重构，只查缓存 K/V。

---

## 2. CLIPSegHead：动态图相关原型解码

CLIPSegHead 侧负责“生成图相关类别文本原型”并“用这些原型做像素分类”。这里的文本原型每张图都不同，因此和 backbone 固定文本路完全解耦。

### 2.1 图相关分组注意力池化

**设计动机**：每类 10 条提示词代表不同场景语义，例如 water 下有 river、lake、wave 等。固定平均会损失场景适应性；图相关 query 能让每张图自己选择更相关的提示词。

```
多阶段骨干特征：
  feats = [B,64,H/4,W/4], [B,128,...], [B,256,...], [B,512,...]

生成图相关 query：
  pooled = concat(global_avg_pool(feats))        # [B, 960]
  img_q = image_query_proj(pooled)               # [B, 3*512] → [B,3,512]
  prior = attn_pool_query                        # [3,1,512] → [B,3,512]
  fused_q = L2norm(img_q) + L2norm(prior)         # [B,3,512]

按类别分组池化：
  q = fused_q.unsqueeze(2)                       # [B,3,1,512]
  k = v = prompt_embeddings[:, sample_idx]       # [3,K_eff,512] → [B,3,K_eff,512]
  attn = softmax(q @ k.T / sqrt(512))            # [B,3,1,K_eff]
  prototype = (attn @ v).squeeze(2)              # [B,3,512]
```

其中 `K_eff` 是当前前向实际参与池化的提示词数量：训练时先随机 `K_eff ∈ [1,K]`，再随机采样对应提示词位置；当前水域配置 `K=10`，即每次随机使用 1-10 条。推理时 `K_eff=10`，使用完整提示词集合。关键点是**分组注意力**：water 的 query 只看 water 的 `K_eff` 条提示词，ship 与 land 同理，不把 30 条文本展平成一个混合池。这样类别边界稳定，场景选择又保持图相关。

### 2.2 提示词增强与 RepRTA 原型精炼

训练时，动态原型池化前会对提示词做轻量增强：

- 先随机采样数量 `K_eff ∈ [1,K]`，再随机采样对应提示词位置；当前 `K=10`，即随机 1-10 条。所有类别共享同一组位置索引，避免 query 依赖固定提示词组合。
- 对采样后的 embedding 加高斯噪声 `σ=0.01`，降低对精确坐标的过拟合。
- 推理时使用全部 10 条提示词，不加噪声。

池化得到 `[B,3,512]` 后，进入 `RepRTA` 精炼。这里的 `RepRTA` 可理解为 **Residual Text Adapter**：一个零初始化末层的 SwiGLU 残差适配器，用来微调动态图相关文本原型。

```
x12 = Linear(prototype)
x1, x2 = chunk(x12)
refined = Linear(SiLU(x1) * x2)
prototype = L2norm(prototype + refined)
```

`RepRTA` 位于 `TextEncoder` 内，不在 `CLIPSegHead` 类本体里；但它属于 head 侧动态原型路径，作用对象是 `[B,3,512]`，不是 backbone 的固定 `[30,512]`。

### 2.3 文本原型驱动的解码分类头

`CLIPSegHead` 将四阶段视觉特征融合后投影到 512 维文本空间，再与动态类别原型做逐像素点积分类。

```
# 输入来自 backbone 的 4 个 stage，空间分辨率以 stage0 为对齐目标
x0, x1, x2, x3 = feats
# x0: [B,  64, H0, W0]
# x1: [B, 128, H1, W1]
# x2: [B, 256, H2, W2]
# x3: [B, 512, H3, W3]

# 每个 stage 先通过独立 1×1 ConvModule 映射到统一通道数 256
y0 = convs[0](x0)                                # [B,256,H0,W0]
y1 = resize(convs[1](x1), size=(H0,W0))          # [B,256,H0,W0]
y2 = resize(convs[2](x2), size=(H0,W0))          # [B,256,H0,W0]
y3 = resize(convs[3](x3), size=(H0,W0))          # [B,256,H0,W0]

# 多尺度融合
y = concat([y0,y1,y2,y3], dim=channel)           # [B,1024,H0,W0]
y = fusion_conv(y)                               # [B,256,H0,W0]

# 投影到 CLIP 文本空间
feat = proj_norm(y)                              # [B,256,H0,W0]
feat = proj(feat)                                # [B,512,H0,W0]

# prototype 主路径为图相关 [B,3,512]；旧 fallback 可为 [3,512]
w = prototype
if w.dim() == 2:
    w = w.unsqueeze(0)                           # [1,3,512]，按 batch 广播

# 严格余弦消融分支
if normalize_visual:
    feat = L2norm(feat, dim=channel)             # [B,512,H0,W0]

# 逐像素文本原型分类
logits = einsum("bchw,bkc->bkhw", feat, w)       # [B,3,H0,W0]
# 等价理解：对每张图 b，w[b,k,c] × feat[b,c,h,w] 沿 c=512 求和，
# 即 [K,C] × [C,H,W] → [K,H,W]，相当于逐图动态 1×1 分类卷积。
logits = logits * exp(logit_scale) + bias

# CLIPSegHead.forward 到这里返回的是 stage0 分辨率 logits。
# 训练时 BaseDecodeHead.loss_by_feat 会 resize 到 gt_sem_seg 尺寸；
# 推理时 BaseDecodeHead.predict_by_feat 会 resize 到 img_shape/pad_shape。
# 当前 backbone 的 stage0 通常是输入的 1/4，因此常见情况是再上采样 4 倍。
seg_logits = resize(logits, target_size)         # [B,3,H_img,W_img]
```

**参考：SAM3/DETR 式 cross-attention decoder 思路**

当前 `einsum` 只做相似度分类，可理解为 `QK^T`：像素视觉特征作为 `Q`，类别文本原型作为 `K`，直接得到每个像素对每个类别的分数。若改成 cross-attention decoder，则会多出 `softmax(QK^T)V` 的信息聚合步骤，先让视觉 token 和文本/查询 token 交互，再输出 mask 或类别 logits。

一种简化的像素级 cross-attention 写法如下：

```
# 视觉 token：来自 head 投影后的 feat
X = flatten(feat)                                # [B,H0*W0,512]

# 文本 token：来自动态图相关类别原型
T = prototype                                    # [B,3,512]
K_text = text_k(T)                               # [B,3,512]
V_text = text_v(T)                               # [B,3,512]

# 像素作为 query 去读取文本 value
Q_pix = pix_q(X)                                 # [B,H0*W0,512]
A = softmax(Q_pix @ K_text.T / sqrt(512), dim=class)
text_ctx = A @ V_text                            # [B,H0*W0,512]

# 文本上下文回写到像素特征
X_enhanced = LayerNorm(X + gate * text_ctx)
X_enhanced = FFN(X_enhanced) + X_enhanced

# 分类方式 A：继续使用文本原型点积分类（分类权重来自 T，不是独立 Linear 参数）
logits = X_enhanced @ T.T                        # [B,H0*W0,3]

# 分类方式 B：也可改成普通线性分类头（分类权重为可学习参数 W_cls）
# logits = Linear(512, num_classes)(X_enhanced)  # [B,H0*W0,3]
logits = reshape(logits, [B,3,H0,W0])
```

更接近 DETR/SAM3 detector 的写法会引入一组 object/mask queries：

```
image_tokens = pixel_features.flatten()           # [B, H0*W0, 512]
text_tokens  = prototype                          # [B, 3, 512]
Q_obj = learned_or_prompt_queries                 # [B, Nq, 512]，Nq 远小于像素数
Q_obj = SelfAttention(Q_obj)                      # [B, Nq, 512]，query 间通信，学分工
Q_obj = CrossAttention(Q_obj, image_tokens)       # [B, Nq, 512]，图像作 KV，query 圈定区域
Q_obj = CrossAttention(Q_obj, text_tokens)        # [B, Nq, 512]，文本作 KV，query 对齐语义
mask_embed = MLP(Q_obj)                           # [B, Nq, 512]
mask_logits = mask_embed @ pixel_features         # [B, Nq, H0, W0]，每 query 出一张 mask
class_or_presence = Head(Q_obj)                   # [B, Nq, num_classes(+1)]，query 级分类
```

区别在于：当前方案没有 decoder query 迭代，也没有 `V` 聚合，计算更轻，输出直接是 3 类语义 logits；cross-attention decoder 更强但更重，适合开放词汇、多实例或需要 object query 的场景。

当前默认 `normalize_visual=False`，文本原型已 L2 归一化，但视觉特征不归一化：

```
logits = ||visual_feat|| × cos(theta) × scale + bias
```

这保留了视觉特征范数作为置信度。若做严格余弦消融，配置 `normalize_visual=True`，head 会先对 `feat` 做通道维 L2 归一化，再与文本原型点积。对应消融配置：

- `configs-h/clip/waterseg.py`：默认点积形式
- `configs-h/clip/waterseg_cosine.py`：严格余弦形式
- `configs-h/clip/attn_waterseg.py`：BRG 注意力消融 + 默认点积
- `configs-h/clip/attn_waterseg_cosine.py`：BRG 注意力消融 + 严格余弦

### 2.4 Head 部署边界

动态图相关原型 `[B,3,512]` 每张图不同，因此默认部署不把 head 折叠成单个固定 `Conv2d`。推理仍实时执行：

```
image_query_proj + pool_with_query + RepRTA + einsum
```

这部分相对 backbone 开销较小，且保留图相关选择能力。只有旧固定原型 fallback 才允许 `fuse_head=True`，把固定 `[3,512]` 原型与 BN、proj、scale、bias 近似融合成 `Conv2d(256,3,1×1)`；该路径会丢失图相关原型选择。若 `normalize_visual=True`，由于逐像素 L2 归一化是非线性操作，不能融合成单个卷积。

---

## 训练与部署摘要

---

## 3. CLIPSegHeadV2：类激活引导的视觉条件文本原型

当前旧版 `CLIPSegHead` 保留为 `v1` 消融对照；新的主路线为 `CLIPSegHeadV2`。它不再用全局图像向量生成图相关 query，而是先让普通视觉分割头产生每类空间激活，再用每类激活区域生成对应类别的视觉提示，从而让文本原型知道“这一类应该看图中哪里”。

### 3.1 设计目标

- 同域 `KAKA` 精度不低于纯视觉头。
- 文本分支不只增加参数，而是真正参与分类。
- 优先观察 `gqy`、`GBA` 泛化是否超过纯视觉 `Q4-no-text`。
- 不整套搬 `YOLOE` 检测头，只借鉴三点：视觉特征归一化/标定、文本原型残差精炼、训练后可解释为文本分类核。

### 3.2 前向流程

```
feats
→ SegFormer 式多尺度融合
→ fusion_feat                         # [B,256,H/4,W/4]

base_logits = Conv(fusion_feat)        # [B,3,H/4,W/4]
activation = softmax(base_logits)      # [B,3,H/4,W/4]

visual_feat = Conv(BN(fusion_feat))    # [B,512,H/4,W/4]
visual_prompt = class_weighted_avg(
    visual_feat, activation)           # [B,3,512]

base_proto = TextEncoder()             # [3,512]
delta = Linear(LayerNorm(visual_prompt))
adapted_proto = L2norm(
    base_proto + lambda * delta)        # [B,3,512]

clip_logits = BNContrastive(
    visual_feat, adapted_proto)         # [B,3,H/4,W/4]

logits = base_logits + gamma * clip_logits
```

其中 `gamma` 为可学习标量，初始值为 `0.1`；`lambda` 也是可学习标量，初始值为 `0.1`。视觉增量投影的最后一层零初始化，因此训练初期 `adapted_proto` 基本等价于原始文本原型，避免一开始破坏 CLIP 语义。

### 3.3 文本增量接口

`TextEncoder` 新增 `adapt_with_visual_prompt(visual_prompt, delta_scale)`：

```
base_proto = TextEncoder.forward()                  # [3,512]
delta = visual_delta_proj(visual_delta_norm(prompt))
adapted_proto = L2norm(base_proto + delta_scale * delta)
```

该接口只在 `CLIPSegHeadV2` 中启用。旧版 `CLIPSegHead` 和 `v1` 消融不会创建这部分参数，避免参数统计被未使用模块污染。

### 3.4 损失设计

训练时同时约束三路输出：

- 主损失：最终融合 `logits` 的交叉熵，权重 `1.0`。
- 辅助损失：`base_logits` 交叉熵，权重 `0.4`，保证同域精度稳定。
- 辅助损失：`clip_logits` 交叉熵，权重 `0.4`，强迫文本分支参与。
- 文本漂移约束：`1 - cos(adapted_proto, base_proto)`，权重 `0.02`，防止视觉提示把文本原型完全改成普通分类器。

### 3.5 配置与消融

模型配置新增：

```
clip_head_type = 'v1' | 'v2'
visual_prompt_mode = 'class_activation'
clip_logit_weight_init = 0.1
text_delta_scale_init = 0.1
```

默认主实验使用 `clip_head_type='v2'`。`v1` 仍保留旧的 `decode_fusion + separate` 路线，用于对照。

`ablation/clip_head.py` 的核心三组为：

- `brg-query-Q4-no-text`：纯视觉基线。
- `clip-v1-best`：旧版 `decode_fusion + separate`。
- `clip-v2-actprompt`：新版类激活视觉提示文本原型。

泛化测试继续输出简表：

```
dataset,mIoU,IoU_background,IoU_boat,IoU_free_space,mACC,ablation,status
```

**训练前准备**

1. 使用冻结 CLIP 文本塔生成 `tools/prompt_bank_water.pt`，形状为 `[3,10,512]`。
2. `backbone_text` 始终来自完整 30 条提示词，不做随机增强。
3. head 动态原型池化训练时启用提示词采样与噪声增强。

**训练前向**

```
输入图像
→ backbone(inputs, category_prototypes=backbone_text)
→ 多阶段视觉特征 feats
→ image_query_proj(feats) 得 fused_q
→ pool_with_query(fused_q) 得 prototype [B,3,512]
→ CLIPSegHead(feats, prototype) 得 logits [B,3,H,W]
→ CrossEntropyLoss
```

**部署前向**

```
backbone 段：TextRefiner / TTRM / TextCrossAttention 文本投影全部缓存
head 段：动态图相关原型实时生成，默认不融合成固定 Conv2d
CLIP 文本塔：全程离线，不进入训练或推理图
```
