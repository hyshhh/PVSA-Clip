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
CUDA_VISIBLE_DEVICES=0 python tools/analysis_tools/benchmark.py configs-h/biformer/biformer_baseline_waterseg.py work_dirs/baseline/best_mIoU_epoch.pth --cfg-options model.backbone.topp_flash_backend=None

# CLIP
CUDA_VISIBLE_DEVICES=0 python tools/analysis_tools/benchmark.py configs-h/biformer/biformer_clip_waterseg.py work_dirs/clip_waterseg/best_mIoU_epoch.pth --cfg-options model.backbone.topp_flash_backend=None

# 保存分割可视化结果
CUDA_VISIBLE_DEVICES=0 python tools/test.py configs-h/biformer/biformer_clip_waterseg.py checkpoint.pth --show-dir vis_results/
```

## 部署
```bash
python tools/deploy_clip_pvsa.py --config configs-h/biformer/biformer_clip_waterseg.py --checkpoint work_dirs/clip_waterseg/best_mIoU_epoch.pth --output work_dirs/deployed/
```

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
CUDA_VISIBLE_DEVICES=0 python tools/analysis_tools/benchmark.py configs-h/biformer/biformer_clip_waterseg.py checkpoint.pth --cfg-options model.backbone.topp_flash_backend=cuda model.backbone.topp_flash_debug=True
```
GPU 架构检测失败：`export PVSA_TOPP_FLASH_ARCH="8.6"`

## 配置文件
| 文件 | 说明 |
|------|------|
| `configs-h/_base_/models/VTFormer-s-baseline.py` | Baseline 模型 |
| `configs-h/_base_/models/VTFormer-clip.py` | CLIP 模型 |
| `configs-h/biformer/biformer_baseline_waterseg.py` | Baseline 训练 |
| `configs-h/biformer/biformer_clip_waterseg.py` | CLIP 训练 |

## 注意事项
- 非训练脚本运行前需设置：`export PYTHONPATH=/media/ddc/新加卷/hys/hysnew3/PVSA/PVSA-Clip:$PYTHONPATH`
- CUDA 核路径只面向推理，不用于训练。
- 调整 `energy`/`p`/`temperature`/`maxk` 修改配置中的 `topp_route_configs`。
- CLIP Text Encoder 训练时冻结，部署推理时移除。
- 部署后模型等价于原始 PVSA-Net + 1x1 Conv，零额外推理开销。
- 服务器训练前确保在 `pvsa-v3.0` 分支：`git checkout pvsa-v3.0 && git pull`
