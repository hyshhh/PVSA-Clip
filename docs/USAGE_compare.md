# 对比实验命令

## 支持的对比模型

| 模型 | Backbone | Decode Head | 默认启用 |
|------|----------|-------------|---------|
| DeepLabV3+ | ResNet-18 | DepthwiseSeparableASPPHead | ✅ |
| Swin-T + UPerNet | SwinTransformer-Tiny | UPerHead | ❌ |
| SegFormer-B2 | MixVisionTransformer-B2 | SegformerHead | ❌ |
| BiFormer-S | BiFormer-Small | SegformerHead | ❌ |
| BiFormer-B | BiFormer-Base | SegformerHead | ❌ |

切换模型：编辑 `configs-h/vision/baselines_compare.py`，取消注释对应的 `_base_` 行。

## 训练
```bash
CUDA_VISIBLE_DEVICES=0 python tools/train.py configs-h/vision/baselines_compare.py --work-dir work_dirs/<model_name>
# 示例
CUDA_VISIBLE_DEVICES=0 python tools/train.py configs-h/vision/baselines_compare.py --work-dir work_dirs/deeplabv3plus_r18
CUDA_VISIBLE_DEVICES=0 python tools/train.py configs-h/vision/baselines_compare.py --work-dir work_dirs/swin_t_upernet
CUDA_VISIBLE_DEVICES=0 python tools/train.py configs-h/vision/baselines_compare.py --work-dir work_dirs/segformer_b2
```

## 测试
```bash
CUDA_VISIBLE_DEVICES=0 python tools/test.py configs-h/vision/baselines_compare.py work_dirs/<model_name>/best_mIoU.pth
```

## 参数量与 FLOPs
```bash
python tools/analysis_tools/get_flops.py configs-h/vision/baselines_compare.py --shape 256 256
python tools/analysis_tools/pvsa_stage_complexity.py configs-h/vision/baselines_compare.py --shape 256 256
```

## 预训练权重
| 模型 | 权重 | 说明 |
|------|------|------|
| ResNet-18 | `resnet18_v1c` | mmseg 自动下载 |
| Swin-T | [下载](https://download.openmmlab.com/mmpretrain/v1.0/swin/swin-tiny_3rdparty_16xb64_in1k_300e/swin-tiny_3rdparty_16xb64_in1k_300e_20230912_224855-e6a0c6bf.pth) | 手动下载，backbone config 加 `init_cfg=dict(type='Pretrained', checkpoint='path')` |
| MiT-B2 | [下载](https://download.openmmlab.com/mmsegmentation/v0.5/segformer/segformer_mit-b2_512x512_160k_ade20k/segformer_mit-b2_512x512_160k_ade20k_20220617_164113-83a7b3e6.pth) | 同上 |

## 训练设置对比
| 设置 | PVSA-Net | DeepLabV3+ | Swin-T | SegFormer-B2 |
|------|----------|------------|--------|-------------|
| 优化器 | AdamW | SGD | AdamW | AdamW |
| 学习率 | 6e-4 | 0.01 | 6e-4 | 6e-4 |
| LR策略 | PolyLR(p=1.0) | PolyLR(p=0.9) | PolyLR(p=1.0) | PolyLR(p=1.0) |
| 轮次/Batch | 200/16 | 200/16 | 200/16 | 200/16 |
| 输入尺寸 | 256x256 | 256x256 | 256x256 | 256x256 |

## 注意事项
1. 切换模型后必须改 `--work-dir`，避免覆盖
2. num_classes=3 对应 gqy 数据集（water/ground/object）
3. 显存不够可调小 batch_size 并线性缩放学习率
4. Swin/SegFormer 预训练权重需手动下载
5. DeepLabV3+ 用 SGD，切换时需在 `baselines_compare.py` 中注释/取消注释对应优化器块
