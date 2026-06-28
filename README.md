# PVSA-Net

基于 MMSegmentation 的语义分割项目，支持三条路径：

1. **CLIP 增强路径**：注入 CLIP 文本语义，支持开放词汇分割，部署时零额外开销。
2. **Baseline 路径**：非 CLIP，soft routing，用于对比实验。
3. **原始路径**：原始 PVSA-Net，用于训练和普通推理。

## 环境

```bash
pip install -r requirements/mminstall.txt
pip install -r requirements/runtime.txt
pip install openai-clip
```

## Prompt Bank（仅 CLIP 路径需要）

```bash
# 本地下载权重后传到服务器
wget -O tools/ViT-B-32.pt https://openaipublic.azureedge.net/clip/models/40d365715913c9da98579312b702a82c18be219cc2a73407c4526f58eba950af/ViT-B-32.pt
python tools/generate_water_prompt_bank.py --output tools/prompt_bank_water.pt --model ViT-B/32 --model-path tools/ViT-B-32.pt
```

## 训练

CLIP 增强训练：
```bash
CUDA_VISIBLE_DEVICES=0 python tools/train.py \
  configs-h/biformer/biformer_clip_waterseg.py \
  --work-dir work_dirs/clip_waterseg
```

Baseline 训练（非 CLIP，soft routing）：
```bash
CUDA_VISIBLE_DEVICES=0 python tools/train.py \
  configs-h/biformer/biformer_baseline_waterseg.py \
  --work-dir work_dirs/baseline
```

原始 PVSA-Net 训练：
```bash
CUDA_VISIBLE_DEVICES=0 python tools/train.py \
  configs-h/biformer/biformer_mm-20k_chase_db1-512x512.py \
  --cfg-options model.backbone.topp_flash_backend=None \
  model.backbone.feature_vis_config.enabled=False \
  model.backbone.attn_vis_config.enabled=False \
  train_dataloader.batch_size=16 \
  --work-dir work_dirs/pvsa_baseline
```

## 原始路径推理

```bash
export PYTHONPATH=/media/ddc/新加卷/hys/hysnew3/PVSA/PVSA-Clip:$PYTHONPATH
CUDA_VISIBLE_DEVICES=0 python tools/analysis_tools/benchmark.py \
  configs-h/biformer/biformer_mm-20k_chase_db1-512x512.py \
  checkpoint.pth \
  --cfg-options model.backbone.topp_flash_backend=None
```

## CLIP 模型推理

```bash
export PYTHONPATH=/media/ddc/新加卷/hys/hysnew3/PVSA/PVSA-Clip:$PYTHONPATH
CUDA_VISIBLE_DEVICES=0 python tools/analysis_tools/benchmark.py \
  configs-h/biformer/biformer_clip_waterseg.py \
  work_dirs/clip_waterseg/best_mIoU_epoch.pth \
  --cfg-options model.backbone.topp_flash_backend=None
```

## 推理并保存分割结果

```bash
export PYTHONPATH=/media/ddc/新加卷/hys/hysnew3/PVSA/PVSA-Clip:$PYTHONPATH
CUDA_VISIBLE_DEVICES=0 python tools/test.py \
  configs-h/biformer/biformer_clip_waterseg.py \
  checkpoint.pth \
  --show-dir vis_results/
```

## 部署（CLIP 路径）

```bash
python tools/deploy_clip_pvsa.py \
  --config configs-h/biformer/biformer_clip_waterseg.py \
  --checkpoint work_dirs/clip_waterseg/best_mIoU_epoch.pth \
  --output work_dirs/deployed/
```

融合步骤：冻结 prototypes → BN+Conv2d 融合 → TTRM 文本投影预计算 → TextCrossAttention K/V 预计算 → 跳过 RepRTA → 输出零文本开销模型。

## 复杂度统计

```bash
export PYTHONPATH=/media/ddc/新加卷/hys/hysnew3/PVSA/PVSA-Clip:$PYTHONPATH
python tools/analysis_tools/get_flops.py \
  configs-h/biformer/biformer_clip_waterseg.py --shape 224 224
python tools/analysis_tools/pvsa_stage_complexity.py \
  configs-h/biformer/biformer_clip_waterseg.py --shape 224 224
```

## 自定义 CUDA 核推理

首次运行或修改 CUDA 源码后，建议先清理旧编译缓存：
```bash
rm -rf ~/.cache/torch_extensions/py*/pvsa_topp_flash_cuda
```

TopP 推理只保留两个开关：
- `model.backbone.topp_flash_backend=None` 或 `cuda`
- `model.backbone.topp_flash_debug=False` 或 `True`

推理模板：
```bash
export PYTHONPATH=/media/ddc/新加卷/hys/hysnew3/PVSA/PVSA-Clip:$PYTHONPATH
export CC=/usr/bin/gcc-11
export CXX=/usr/bin/g++-11
CUDA_VISIBLE_DEVICES=0 python tools/analysis_tools/benchmark.py \
  configs-h/biformer/biformer_clip_waterseg.py \
  checkpoint.pth \
  --cfg-options model.backbone.topp_flash_backend=cuda \
  model.backbone.topp_flash_debug=False
```

打印各环节时间时，只把 `model.backbone.topp_flash_debug` 改成 `True`。

如果服务器 GPU 架构自动检测失败，可以手动指定：
```bash
export PVSA_TOPP_FLASH_ARCH="8.6"
```

## 配置文件

| 文件 | 说明 |
|------|------|
| `configs-h/_base_/models/VTFormer-s-baseline.py` | Baseline 模型 |
| `configs-h/_base_/models/VTFormer-clip.py` | CLIP 增强模型 |
| `configs-h/biformer/biformer_baseline_waterseg.py` | Baseline 训练 |
| `configs-h/biformer/biformer_clip_waterseg.py` | CLIP 训练 |
| `configs-h/biformer/biformer_mm-20k_chase_db1-512x512.py` | 原始 PVSA-Net |

## 注意事项

- 自定义 CUDA 核路径只面向推理，不用于训练。
- 真实速度测试时不要打开调试日志。
- 如果需要调整 `energy`、`p`、`temperature`、`maxk`，请修改配置文件中的 `topp_route_configs`。
- CLIP Text Encoder 训练时冻结，CPFM 推理时移除。
- 部署后模型等价于原始 PVSA-Net + 1×1 Conv，零额外推理开销。
- 本地不具备 CUDA 编译环境，CUDA 编译和性能验证以服务器结果为准。
