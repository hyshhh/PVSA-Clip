# 对比实验命令

## 支持的对比模型

| 模型 | Backbone | Decode Head | 默认启用 |
|------|----------|-------------|---------|
| DeepLabV3+ | ResNet-50 | DepthwiseSeparableASPPHead | ✅ |
| Swin-T + UPerNet | SwinTransformer-Tiny | UPerHead | ❌ |
| SegFormer-B2 | MixVisionTransformer-B2 | SegformerHead | ❌ |

---

## 切换模型

编辑 `configs-h/biformer/baselines_compare.py`，取消注释你要用的模型块（`model` / `optim_wrapper` / `param_scheduler` 三项），注释掉其余的。

---

## 训练

```bash
# DeepLabV3+ (ResNet-50)
CUDA_VISIBLE_DEVICES=0 python tools/train.py \
    configs-h/biformer/baselines_compare.py \
    --work-dir work_dirs/deeplabv3plus_r50

# Swin-T + UPerNet（需先取消注释 Swin 块）
CUDA_VISIBLE_DEVICES=0 python tools/train.py \
    configs-h/biformer/baselines_compare.py \
    --work-dir work_dirs/swin_t_upernet

# SegFormer-B2（需先取消注释 SegFormer 块）
CUDA_VISIBLE_DEVICES=0 python tools/train.py \
    configs-h/biformer/baselines_compare.py \
    --work-dir work_dirs/segformer_b2
```

---

## 测试

```bash
# DeepLabV3+
CUDA_VISIBLE_DEVICES=0 python tools/test.py \
    configs-h/biformer/baselines_compare.py \
    work_dirs/deeplabv3plus_r50/best_mIoU.pth

# Swin-T
CUDA_VISIBLE_DEVICES=0 python tools/test.py \
    configs-h/biformer/baselines_compare.py \
    work_dirs/swin_t_upernet/best_mIoU.pth

# SegFormer-B2
CUDA_VISIBLE_DEVICES=0 python tools/test.py \
    configs-h/biformer/baselines_compare.py \
    work_dirs/segformer_b2/best_mIoU.pth
```

---

## 我们的模型（PVSA-Net）

```bash
# Baseline
CUDA_VISIBLE_DEVICES=0 python tools/train.py \
    configs-h/biformer/biformer_baseline_waterseg.py \
    --work-dir work_dirs/baseline

# CLIP
CUDA_VISIBLE_DEVICES=0 python tools/train.py \
    configs-h/biformer/biformer_clip_waterseg.py \
    --work-dir work_dirs/clip_waterseg
```

---

## 预训练权重

| 模型 | 预训练权重 | 下载方式 |
|------|-----------|---------|
| ResNet-50 | `resnet50_v1c` | mmseg 自动下载（`pretrained='open-mmlab://resnet50_v1c'`） |
| Swin-T | `swin_tiny_patch4_window7_224.pth` | [下载](https://download.openmmlab.com/mmpretrain/v1.0/swin/swin-tiny_3rdparty_16xb64_in1k_300e/swin-tiny_3rdparty_16xb64_in1k_300e_20230912_224855-e6a0c6bf.pth) |
| MiT-B2 | `mit_b2.pth` | [下载](https://download.openmmlab.com/mmsegmentation/v0.5/segformer/segformer_mit-b2_512x512_160k_ade20k/segformer_mit-b2_512x512_160k_ade20k_20220617_164113-83a7b3e6.pth) |

使用：在配置 `backbone` 中加 `init_cfg=dict(type='Pretrained', checkpoint='path/to/weight.pth')`。不加也能训练，但收敛慢。

---

## 训练设置对比

| 设置 | PVSA-Net | DeepLabV3+ | Swin-T | SegFormer-B2 |
|------|----------|------------|--------|-------------|
| 优化器 | AdamW | SGD | AdamW | AdamW |
| 学习率 | 6e-4 | 0.01 | 6e-4 | 6e-4 |
| 训练轮次 | 200 | 200 | 200 | 200 |
| Warmup | 10 epoch | 10 epoch | 10 epoch | 10 epoch |
| LR策略 | PolyLR(p=1.0) | PolyLR(p=0.9) | PolyLR(p=1.0) | PolyLR(p=1.0) |
| Batch Size | 16 | 16 | 16 | 16 |
| 输入尺寸 | 256×256 | 256×256 | 256×256 | 256×256 |
| 评估指标 | mIoU+mDice | mIoU+mDice | mIoU+mDice | mIoU+mDice |

---

## 注意事项

1. 切换模型后必须改 `--work-dir`，避免覆盖
2. num_classes=3 对应 gqy 数据集（water/ground/object）
3. 显存不够可调小 batch_size 并线性缩放学习率
4. Swin/SegFormer 预训练权重需手动下载到项目根目录
