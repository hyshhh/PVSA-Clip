# 水域语义分割 - CLIP 文本注入 PVSA-Net 架构设计方案

## 核心设计

任务类别固定为 `water`、`ship`、`land`。每类维护 10 条类别相关提示词，用冻结的 CLIP 文本塔离线编码一次，得到 L2 归一化的提示词库 `[3, 10, 512]`。训练和推理阶段不再调用 CLIP 文本塔，只加载 `tools/prompt_bank_water.pt`。

- Water：river、lake、sea、ocean、wave、water surface、water reflection、flood、stream、reservoir
- Ship：boat、vessel、cargo ship、fishing boat、yacht、sailboat、canoe、barge、ship、small boat
- Land：shore、coast、vegetation、road、bridge、building、sky、tree、sand、grass

当前代码主线已经只保留 `CLIPSegHeadV2`。旧版 `CLIPSegHead`、`image_query_proj`、`pool_with_query` 图相关 query 原型池化、`frozen_prototypes` 固定原型部署分支均已从运行链路移除。

当前 CLIP 路径分成两层：

- **backbone 固定文本路**：可选开启。原始 30 条 CLIP 提示词嵌入 `[30,512]` 可经 `TextRefiner` 重构后作为固定 `backbone_text` 注入 `TTRM` 与 `TextCrossAttention`。该路径与输入图像无关，可在部署时冻结成 K/V 缓存。当前默认 `use_backbone_text_injection=False`，也就是默认不启用 backbone 文本注入。
- **CLIPSegHeadV2 图像条件文本路**：默认主线。解码头先融合多阶段视觉特征，再用普通视觉分支产生 `base_logits`，由类激活区域聚合每类 `visual_prompt`，再用 `visual_prompt` 对 CLIP 文本原型做每图残差适配，最终用文本分支 `clip_logits` 与视觉分支融合。

最终推理形态：CLIP 文本塔全程离线；`CLIPSegHeadV2` 保留实时图像条件视觉提示和文本原型适配，不折叠为固定 `Conv2d`。

---

## 1. Backbone：固定文本语义注入

当前配置文件 [clip-topp.py](../configs-h/_base_/models/clip-topp.py) 中：

```python
use_backbone_text_injection = False
```

因此默认训练命令下，backbone 不接收文本向量；只使用 CLIP 路径同构的视觉骨干和 `CLIPSegHeadV2`。下面的固定文本注入是可选能力，只有把 `use_backbone_text_injection=True` 后才生效。

### 1.1 TextRefiner

**作用对象**：原始提示词库 reshape 后的 `[30,512]`。

**作用**：对固定 CLIP 文本向量做轻量残差重构，但不引入图像依赖，保持部署可缓存。

```python
x12 = Linear(512 -> 4 * 512)
x1, x2 = chunk(x12)
refined = Linear(SiLU(x1) * x2)
backbone_text = x + refined
```

`w3` 零初始化，训练初期接近恒等映射。该模块和 `TextEncoder` 内的 `RepRTA` 参数独立，作用对象也不同：`TextRefiner` 处理固定 `[30,512]`，`RepRTA` 处理类别原型 `[3,512]` 或 `[B,3,512]`。

### 1.2 TTRM

**作用位置**：启用文本注入时，Stage 0-2 的 Top-P 路由器。

```python
Q_visual, K_visual = window_tokens
K_text, V_text = proj(backbone_text)

q_text = Attention(Q_visual, K_text, V_text)
k_text = Attention(K_visual, K_text, V_text)

Q_enriched = Q_visual + sigmoid(gate) * q_text
K_enriched = K_visual + sigmoid(gate) * k_text

route_score = Q_enriched @ K_enriched.T
```

`gate` 是模块内部可学习标量，初始化为 `-2`，经 `sigmoid` 后约为 `0.12`，训练初期仍以视觉路由为主。

### 1.3 TextCrossAttention

**作用位置**：启用文本注入时，Stage 2-3 的深层 block。

```python
visual = reshape(visual_features)
Q = visual_q(visual)
K = text_proj_k(backbone_text)
V = text_proj_v(backbone_text)

text_info = Attention(Q, K, V)
enhanced = LayerNorm(visual + sigmoid(gate) * text_info)
```

这里的 `gate` 同样是每个 `TextCrossAttention` 模块内部的单个可学习标量。

### 1.4 部署缓存

[CLIPEncoderDecoder.fuse_for_deployment](../mmseg/models/segmentors/clip_encoder_decoder.py) 现在只冻结 backbone 可部署文本缓存：

1. 若启用 `use_backbone_text_injection`，先计算一次 `backbone_text`。
2. 写入 `frozen_backbone_text`。
3. 调用 backbone 中支持 `freeze_for_deployment` 的模块，冻结文本 K/V。
4. `CLIPSegHeadV2` 不融合成固定卷积，仍保持动态图像条件提示。

---

## 2. TextEncoder：CLIPSegHeadV2 的文本原型子分支

[TextEncoder](../mmseg/models/utils/text_encoder.py) 在代码上是 `CLIPEncoderDecoder` 持有的独立模块，不写在 `CLIPSegHeadV2` 类内部；但在功能上，它服务于 `CLIPSegHeadV2` 的文本原型分支。也就是说，`RepRTA`、提示词注意力池化、视觉增量适配可以看成 `CLIPSegHeadV2` 链路的一部分，只是为了复用和配置管理，具体实现放在 `TextEncoder` 里。

这条文本原型子分支的层级可以理解为：

```text
CLIPSegHeadV2
-> base_logits 生成 visual_prompt
-> TextEncoder 文本原型子分支
   -> prompt bank 加载与类别重排
   -> 类内注意力池化
   -> RepRTA 原型精炼
   -> visual_delta 根据 visual_prompt 做每图适配
-> clip_logits
-> logits
```

`TextEncoder` 当前提供两类接口：

- `forward()`：从 `[3,10,512]` 提示词库生成基础类别原型 `[3,512]`。
- `adapt_with_visual_prompt(visual_prompt, delta_scale)`：用每图视觉提示生成适配后的类别原型 `[B,3,512]`。

### 2.1 Prompt Bank 加载与类别重排

`prompt_bank_water.pt` 原始类别顺序是：

```text
water / ship / land
```

不同数据集的标签通道顺序不同，因此通过 `prompt_category_order` 重排：

```text
KAKA: background / boat / free-space -> land  / ship  / water
gqy : water      / ground / object   -> water / land  / ship
GBA : object     / water  / ground   -> ship  / water / land
```

配置入口在 [clip-topp.py](../configs-h/_base_/models/clip-topp.py)：

```python
prompt_category_order = globals().get(
    'prompt_category_order', prompt_category_orders[prompt_dataset])
```

消融脚本会通过 `--cfg-options` 写入：

```text
model.text_encoder.prompt_category_order=['land', 'ship', 'water']
```

### 2.2 RepRTA 原型精炼

`TextEncoder.forward()` 会先用每类可学习 query 对同类 10 条提示词做注意力池化，再经过 `RepRTA` 残差精炼：

```python
q = attn_pool_query                  # [3,1,512]
k = v = prompt_embeddings            # [3,K,512]
attn = softmax(q @ k.T / sqrt(512))
pooled = attn @ v                    # [3,512]

x12 = Linear(pooled)
x1, x2 = chunk(x12)
refined = Linear(SiLU(x1) * x2)
prototype = L2norm(pooled + refined)
```

训练时会对 prompt embedding 做轻量增强：

- 随机采样 `K_eff` 条提示词，`K_eff` 在 `[1,K]` 内。
- 加高斯噪声 `0.01`。
- 推理时使用全部 10 条提示词，不加噪声。

这里的 `RepRTA` 不接收图像特征，作用是把离线 CLIP 提示词库先整理成更适合水域分割的基础类别原型。它和 `visual_delta` 是前后两步：`RepRTA` 先得到稳的基础原型，`visual_delta` 再根据当前图像做轻量偏移。

### 2.3 视觉条件文本适配

`CLIPSegHeadV2` 传入 `visual_prompt [B,3,512]` 后：

```python
base_proto = TextEncoder.forward()                  # [3,512]
delta = visual_delta_proj(LN(visual_prompt))        # [B,3,512]
adapted_proto = L2norm(base_proto[None] + s * delta)
```

其中 `s = text_delta_scale`，代码里是可学习正值门控：

```python
text_delta_scale = softplus(text_delta_scale_raw)
```

当前默认初值：

```python
text_delta_scale_init = 0.2
```

`use_visual_delta` 不需要在配置里手动写死。`CLIPEncoderDecoder` 会先判断当前解码头是否有 `loss_with_text`；如果是 `CLIPSegHeadV2`，就默认把 `TextEncoder.use_visual_delta` 打开。因此 `clip-v2-actprompt` 这条链路中，`adapt_with_visual_prompt()` 一定是可用的。

---

## 3. CLIPSegHeadV2：当前唯一 CLIP Head 主线

[CLIPSegHeadV2](../mmseg/models/decode_heads/clip_seg_head.py) 是当前唯一保留的 CLIP 解码头。它不是旧版“先全局池化图像，再生成 query 去读文本”的结构，而是以普通视觉分支的类别激活为锚点，生成每类视觉提示，再反过来调整 CLIP 文本原型。

完整前向可以压缩成一句话：

```text
fusion_feat -> base_logits -> visual_prompt -> adapted_proto -> clip_logits -> logits
```

其中 `base_logits` 不是最终要证明的 CLIP 增益本身，而是两个作用：一是提供稳定的视觉定位，二是告诉文本分支每个类别该从图像的哪些区域提取视觉提示。为了避免最终结果只依赖普通视觉头，训练时会额外监督 `clip_logits`，并把 `clip_loss_weight` 设得高于 `base_loss_weight`。

### 3.1 模块结构

输入是 backbone 四阶段特征：

```text
x0: [B,  64, H/4,  W/4]
x1: [B, 128, H/8,  W/8]
x2: [B, 256, H/16, W/16]
x3: [B, 512, H/32, W/32]
```

多尺度融合：

```python
y_i = ConvModule_1x1(x_i)
y_i = resize(y_i, size=x0.shape[2:])
fusion_feat = fusion_conv(concat(y_0, y_1, y_2, y_3))
```

得到：

```text
fusion_feat: [B,256,H/4,W/4]
```

### 3.2 普通视觉分支

```python
base_logits = self.cls_seg(fusion_feat)
```

形状：

```text
base_logits: [B,3,H/4,W/4]
```

这一路提供稳定的视觉分割能力，也为 `visual_prompt` 提供类别空间激活图。它相当于视觉定位器，不直接代表最终方法退化成普通分割头；后面的文本分支仍要独立输出 `clip_logits` 并接受辅助监督。

### 3.3 类激活视觉提示

当前实现不是简单全图平均，而是温度锐化后只聚合每类高响应区域：

```python
activation = softmax(base_logits / visual_prompt_temperature, dim=1)

flat = activation.flatten(2)
threshold = flat.topk(keep, dim=-1).values[..., -1:]
mask = flat >= threshold
activation = activation * mask

visual_prompt = einsum('bkhw,bdhw->bkd', activation, visual_feat)
visual_prompt = visual_prompt / activation.sum()
```

当前默认：

```python
visual_prompt_temperature = 0.5
visual_prompt_topk_ratio = 0.25
```

直观含义：

- `temperature=0.5` 会让类别激活更尖锐。
- `topk_ratio=0.25` 让每类只聚合响应最高的 25% 空间位置。
- 对小目标如 `boat/ship`，这能减少背景和水域低置信区域对视觉提示的稀释。

### 3.4 文本分支分类

视觉特征先投影到 CLIP 文本维度：

```python
visual_feat = visual_proj(BN(fusion_feat))        # [B,512,H/4,W/4]
```

文本原型根据当前图像适配。这里会先调用 `TextEncoder.forward()`，因此基础原型已经经过提示词注意力池化和 `RepRTA` 精炼；随后 `visual_delta_proj` 再根据 `visual_prompt` 给每张图、每个类别加一个轻量残差：

```python
adapted_proto, base_proto = text_encoder.adapt_with_visual_prompt(
    visual_prompt, text_delta_scale)
```

再做 BN 对比分类：

```python
contrast_feat = contrast_norm(visual_feat)
clip_logits = einsum('bdhw,bkd->bkhw', contrast_feat, adapted_proto)
clip_logits = clip_logits * exp(clip_logit_scale) + clip_bias
```

当前 `clip_logit_scale` 初始为 `-1.0`，`clip_bias` 初始为 `-10.0`。

### 3.5 正值融合门控

最终输出：

```python
logits = base_logits + clip_logit_weight * clip_logits
```

`clip_logit_weight` 不是任意正负标量，而是正值门控：

```python
clip_logit_weight = softplus(clip_logit_weight_raw)
```

当前默认初值：

```python
clip_logit_weight_init = 0.3
```

这样可以避免文本分支被训练成负残差直接抵消，同时仍允许优化器把文本分支影响调小。

### 3.6 损失设计

训练时同时监督三路输出：

```python
loss =
  CE(logits, label)
  + base_loss_weight * CE(base_logits, label)
  + clip_loss_weight * CE(clip_logits, label)
  + text_drift_loss_weight * mean(1 - cos(adapted_proto, base_proto))
```

当前默认：

```python
base_loss_weight = 0.2
clip_loss_weight = 0.6
text_drift_loss_weight = 0.01
```

含义：

- 主损失监督最终融合输出。
- `base` 辅助损失保证视觉定位分支稳定，但权重较低，避免模型退化成普通视觉头。
- `clip` 辅助损失要求文本分支单独具备分类能力。
- `text_drift` 防止视觉增量把 CLIP 原型拉得过远。

所以当前 `CLIPSegHeadV2` 的精度来源应该分两部分看：`base_logits` 负责把区域找准，`clip_logits` 负责让像素分类权重来自经过 CLIP 原型约束的文本方向。如果实验中 `clip_logits` 单独效果很弱，或者 `clip_logit_weight` 训练后接近 0，就说明文本分支贡献不足；如果二者都正常，最终提升才更像是真正来自文本原型适配，而不是单纯堆参数。

### 3.7 训练与推理入口

`CLIPEncoderDecoder` 通过是否存在 `loss_with_text` 判断当前 decode head 是否为 `CLIPSegHeadV2`：

```python
self.use_activation_prompt_head = hasattr(self.decode_head, 'loss_with_text')
```

训练：

```python
feats, _ = self.extract_feat(inputs)
loss_decode = self.decode_head.loss_with_text(
    feats, data_samples, self.text_encoder)
```

推理：

```python
feats, _ = self.extract_feat(inputs)
seg_logits = self.decode_head.predict_with_text(
    feats, batch_img_metas, self.test_cfg, self.text_encoder)
```

普通无文本对照头没有 `loss_with_text`，会退回标准 `decode_head.loss/predict`。

---

## 4. 消融入口

当前 `ablation/clip_head.py` 只保留三个有效变体：

```text
brg-query-Q4-no-text
brg-query-Q5-same-backbone-no-text
clip-v2-actprompt
```

含义：

- `brg-query-Q4-no-text`：旧无文本基线，`EncoderDecoder + BiFormer_standalone + SegformerHead`。
- `brg-query-Q5-same-backbone-no-text`：同构无文本基线，使用和 CLIP 路径相同的 `BiFormer_fusion_clip` 骨干，但不构建 `CLIPEncoderDecoder/TextEncoder/CLIPSegHeadV2`。
- `clip-v2-actprompt`：当前主线，`CLIPEncoderDecoder + BiFormer_fusion_clip + CLIPSegHeadV2`。

KAKA 默认命令：

```bash
CUDA_VISIBLE_DEVICES=0 python ablation/clip_head.py --work-dir-root ablation/clip_head_fix_prompt --shape 256 256 --variants clip-v2-actprompt --train-dataset kaka
CUDA_VISIBLE_DEVICES=0 python ablation/clip_head.py --work-dir-root ablation/clip_head_fix_prompt --shape 256 256 --variants brg-query-Q4-no-text brg-query-Q5-same-backbone-no-text clip-v2-actprompt --train-dataset kaka
```

---

## 5. 部署边界

当前 `CLIPSegHeadV2` 依赖当前图像的 `base_logits` 和 `visual_prompt`，因此不能融合成固定 `Conv2d`。

部署时可以缓存的只有 backbone 固定文本注入部分：

```text
TextRefiner / TTRM / TextCrossAttention 的固定文本 K/V
```

head 侧仍实时执行：

```text
fusion_feat
-> base_logits
-> visual_prompt
-> adapted_proto
-> clip_logits
-> logits
```

这正是当前真实代码的行为。
