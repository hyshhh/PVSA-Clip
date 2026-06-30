# PVSA-Clip 操作指南

## 环境安装
```bash
pip install -r requirements/mminstall.txt && pip install -r requirements/runtime.txt && pip install openai-clip
```

## Prompt Bank（CLIP 路径必需）
```bash
wget -O tools/ViT-B-32.pt https://openaipublic.azureedge.net/clip/models/40d365715913c9da98579312b702a82c18be219cc2a73407c4526f58eba950af/ViT-B-32.pt
python tools/generate_water_prompt_bank.py --output tools/prompt_bank_water.pt --model ViT-B/32 --model-path tools/ViT-B-32.pt
```

## 训练
```bash
# Baseline
CUDA_VISIBLE_DEVICES=0 python tools/train.py configs-h/biformer/biformer_baseline_waterseg.py --work-dir work_dirs/baseline

# CLIP
CUDA_VISIBLE_DEVICES=0 python tools/train.py configs-h/biformer/biformer_clip_waterseg.py --work-dir work_dirs/clip_waterseg
```

## 推理
```bash
# Baseline
CUDA_VISIBLE_DEVICES=0 python tools/analysis_tools/benchmark.py configs-h/biformer/biformer_baseline_waterseg.py work_dirs/baseline/epoch_200.pth --cfg-options model.backbone.topp_flash_backend=None

# CLIP
CUDA_VISIBLE_DEVICES=0 python tools/analysis_tools/benchmark.py configs-h/biformer/biformer_clip_waterseg.py work_dirs/clip_waterseg/epoch_200.pth --cfg-options model.backbone.topp_flash_backend=None

# 保存分割可视化结果
CUDA_VISIBLE_DEVICES=0 python tools/test.py configs-h/biformer/biformer_baseline_waterseg.py work_dirs/baseline/epoch_200.pth --show-dir vis_results/baseline/
CUDA_VISIBLE_DEVICES=0 python tools/test.py configs-h/biformer/biformer_clip_waterseg.py work_dirs/clip_waterseg/epoch_200.pth --show-dir vis_results/clip/
```

## 特征图和注意力可视化
```bash
# 非 CLIP 路径
CUDA_VISIBLE_DEVICES=0 python tools/visualize_pvsa.py configs-h/biformer/biformer_baseline_waterseg.py work_dirs/baseline/epoch_200.pth --image demo/demo.png --mode baseline --device cuda:0 --query-index 32
CUDA_VISIBLE_DEVICES=0 python tools/visualize_pvsa.py configs-h/biformer/biformer_baseline_waterseg.py work_dirs/baseline/epoch_200.pth 1 --mode baseline --device cuda:0 --query-index 32

# CLIP 路径
CUDA_VISIBLE_DEVICES=0 python tools/visualize_pvsa.py configs-h/biformer/biformer_clip_waterseg.py work_dirs/clip_waterseg/epoch_200.pth --image demo/demo.png --mode clip --device cuda:0 --query-index 32
CUDA_VISIBLE_DEVICES=0 python tools/visualize_pvsa.py configs-h/biformer/biformer_clip_waterseg.py work_dirs/clip_waterseg/epoch_200.pth 1 --mode clip --device cuda:0 --query-index 32
```
其中 `1` 表示配置文件中测试集的第一张图片；也可以写成 `--test-index 1`。保留 `--image` 时会优先使用手动指定的图片路径。
默认会保存所有 stage、所有 block 的 `top-p` 路由图；如果只想看单个路由块，可加 `--single-route --route-stage 0 --route-block 0`。

默认保存位置：
- 特征图：`demo/feathermap/<baseline|clip>/<图片名>/`
- 注意力图：`demo/attension_map/<baseline|clip>/<图片名>/`

注意力图包含三个子目录：
- `route_scores/`：原始一阶段路由的全量窗口权重，可视化为分块热力图，并在每个局部窗口标注 `softmax * energy` 后的得分。CLIP 路径下会进一步分成 `visual/`、`text/`、`merged/`，分别对应纯视觉得分、文本注入带来的分布变化量、合并后得分。
- `route_scores_topp/`：`top-k` 内温度软化、再经 `top-p` 截断后的软路由分数。训练时最终乘到 `KV` 上的是 `1 + soft_kv_weight * 分数`，并会截断到 `[1, 2]`。
- `top_p_mask/`：`top-p` 裁剪后实际选中的窗口掩码，未选区域会暗化。图中数字是最终乘到 `KV` 上的截断系数，开启软路由时范围为 `[1, 2]`。

CLIP 路径的特征图会额外保存 `text_injection/`：对启用文本注入的阶段保存注入前、注入后和差异图；没有文本注入的阶段保存在 `stage_outputs/`。

## 部署
```bash
python tools/deploy_clip_pvsa.py --config configs-h/biformer/biformer_clip_waterseg.py --checkpoint work_dirs/clip_waterseg/epoch_200.pth --output work_dirs/deployed/
```
部署后：TextEncoder 移除，Head 融合为 Conv2d，TTRM/Cross-Attn 使用预计算 frozen K/V。

## 复杂度分析
```bash
python tools/analysis_tools/get_flops.py configs-h/biformer/biformer_baseline_waterseg.py --shape 256 256
python tools/analysis_tools/pvsa_stage_complexity.py configs-h/biformer/biformer_baseline_waterseg.py --shape 256 256
python tools/analysis_tools/get_flops.py configs-h/biformer/biformer_clip_waterseg.py --shape 256 256
python tools/analysis_tools/clip_stage_complexity.py configs-h/biformer/biformer_clip_waterseg.py --shape 256 256
```

## CUDA 核加速推理
```bash
rm -rf ~/.cache/torch_extensions/py*/pvsa_topp_flash_cuda
export CC=/usr/bin/gcc-11 && export CXX=/usr/bin/g++-11
CUDA_VISIBLE_DEVICES=0 python tools/analysis_tools/benchmark.py configs-h/biformer/biformer_clip_waterseg.py work_dirs/clip_waterseg/epoch_200.pth --cfg-options model.backbone.topp_flash_backend=cuda model.backbone.topp_flash_debug=True
```
GPU 架构检测失败：`export PVSA_TOPP_FLASH_ARCH="8.6"`

---

## 对比实验

配置文件：`configs-h/biformer/baselines_compare.py`

### 支持的对比模型

| 模型 | Backbone | Decode Head | 默认启用 |
|------|----------|-------------|---------|
| DeepLabV3+ | ResNet-50 | DepthwiseSeparableASPPHead | ✅ |
| Swin-T + UPerNet | SwinTransformer-Tiny | UPerHead | ❌ |
| SegFormer-B2 | MixVisionTransformer-B2 | SegformerHead | ❌ |

### 切换模型

编辑 `baselines_compare.py`，取消注释你要用的模型块（`model` / `optim_wrapper` / `param_scheduler` 三项），注释掉其余模型的块。

### 训练命令

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

### 测试命令

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

### 预训练权重

| 模型 | 预训练权重 | 下载方式 |
|------|-----------|---------|
| ResNet-50 | `resnet50_v1c` | mmseg 自动下载（设置 `pretrained='open-mmlab://resnet50_v1c'`） |
| Swin-T | `swin_tiny_patch4_window7_224.pth` | [下载链接](https://download.openmmlab.com/mmpretrain/v1.0/swin/swin-tiny_3rdparty_16xb64_in1k_300e/swin-tiny_3rdparty_16xb64_in1k_300e_20230912_224855-e6a0c6bf.pth) |
| MiT-B2 | `mit_b2.pth` | [下载链接](https://download.openmmlab.com/mmsegmentation/v0.5/segformer/segformer_mit-b2_512x512_160k_ade20k/segformer_mit-b2_512x512_160k_ade20k_20220617_164113-83a7b3e6.pth) |

使用方式：在配置文件的 `backbone` 中添加 `init_cfg=dict(type='Pretrained', checkpoint='path/to/weight.pth')`。

**注意**：不加载预训练权重也能训练，但收敛会慢很多。

### 训练设置对比

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

### 注意事项

1. **切换模型后一定要修改 `--work-dir`**，避免覆盖其他模型的结果
2. **num_classes=3** 对应 gqy 数据集的 3 类（water/ground/object）
3. 如果显存不够跑 batch_size=16，调小 `train_dataloader.batch_size` 并相应调大学习率
4. Swin 和 SegFormer 预训练权重需要手动下载并放到项目根目录

---

## 配置文件
| 文件 | 说明 |
|------|------|
| `configs-h/_base_/models/VTFormer-s-baseline.py` | Baseline 模型 |
| `configs-h/_base_/models/VTFormer-clip.py` | CLIP 模型 |
| `configs-h/biformer/biformer_baseline_waterseg.py` | Baseline 训练 |
| `configs-h/biformer/biformer_clip_waterseg.py` | CLIP 训练 |
| `configs-h/biformer/baselines_compare.py` | 对比实验（DeepLabV3+/Swin/SegFormer） |

## 注意事项
- 非训练脚本运行前需设置：`export PYTHONPATH=/media/ddc/新加卷/hys/hysnew3/PVSA/PVSA-Clip:$PYTHONPATH`
- CUDA 核路径只面向推理，不用于训练。
- 调整 `energy`/`p`/`temperature`/`maxk` 修改配置中的 `topp_route_configs`。
- CLIP Text Encoder 训练时冻结，部署推理时移除。
- 部署后：TextEncoder 移除，Head 融合为 Conv2d，TTRM/Cross-Attn 使用预计算 frozen K/V 交互。
- 服务器训练前确保在 `pvsa-v3.0` 分支：`git checkout pvsa-v3.0 && git pull`
