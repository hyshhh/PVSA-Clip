# PVSA-Clip（CLIP 路径）操作命令

本文件汇总所有与 **CLIP 文本引导** 相关的命令（含训练 / 推理 / 可视化 / 部署 / 加速）。
纯视觉 baseline 命令见 [USAGE.md](USAGE.md)。

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
# 训练 - CLIP（gqy 数据集, 200 epoch）
CUDA_VISIBLE_DEVICES=0 python tools/train.py configs-h/clip/waterseg.py --work-dir work_dirs/clip_waterseg
# 训练 - CLIP 严格 cosine 分类头消融
CUDA_VISIBLE_DEVICES=0 python tools/train.py configs-h/clip/waterseg_cosine.py --work-dir work_dirs/clip_waterseg_cosine
# 训练 - PVSA-Net Baseline（移除全部文本信号，仅 ToppAttention/BiFormer block 切换，200 epoch）
CUDA_VISIBLE_DEVICES=0 python tools/train.py configs-h/vision/pvsa_net_baseline.py --work-dir work_dirs/pvsa_net_baseline
# 训练 - CLIP + 标准 BiFormer Attention（保留完整文本路径，仅替换注意力为 BRG, 200 epoch）
CUDA_VISIBLE_DEVICES=0 python tools/train.py configs-h/clip/attn_waterseg.py --work-dir work_dirs/clip_biformer_attn
# 训练 - CLIP + 标准 BiFormer Attention + 严格 cosine 分类头消融
CUDA_VISIBLE_DEVICES=0 python tools/train.py configs-h/clip/attn_waterseg_cosine.py --work-dir work_dirs/clip_biformer_attn_cosine
```

## 推理
```bash
# 推理 - CLIP
CUDA_VISIBLE_DEVICES=0 python tools/analysis_tools/benchmark.py configs-h/clip/waterseg.py work_dirs/clip_waterseg/epoch_200.pth
# 推理 - CLIP 严格 cosine 分类头消融
CUDA_VISIBLE_DEVICES=0 python tools/analysis_tools/benchmark.py configs-h/clip/waterseg_cosine.py work_dirs/clip_waterseg_cosine/epoch_200.pth
# 推理 - PVSA-Net Baseline
CUDA_VISIBLE_DEVICES=0 python tools/analysis_tools/benchmark.py configs-h/vision/pvsa_net_baseline.py work_dirs/pvsa_net_baseline/epoch_200.pth
# 推理 - CLIP + 标准 BiFormer Attention
CUDA_VISIBLE_DEVICES=0 python tools/analysis_tools/benchmark.py configs-h/clip/attn_waterseg.py work_dirs/clip_biformer_attn/epoch_200.pth
# 推理 - CLIP + 标准 BiFormer Attention + 严格 cosine 分类头消融
CUDA_VISIBLE_DEVICES=0 python tools/analysis_tools/benchmark.py configs-h/clip/attn_waterseg_cosine.py work_dirs/clip_biformer_attn_cosine/epoch_200.pth
```

## 保存分割可视化
```bash
CUDA_VISIBLE_DEVICES=0 python tools/test.py configs-h/clip/waterseg.py work_dirs/clip_waterseg/epoch_200.pth --show-dir vis_results/clip/
CUDA_VISIBLE_DEVICES=0 python tools/test.py configs-h/vision/pvsa_net_baseline.py work_dirs/pvsa_net_baseline/epoch_200.pth --show-dir vis_results/pvsa_net_baseline/
```

## 参数量与复杂度
```bash
# FLOPs + Params
python tools/analysis_tools/get_flops.py configs-h/clip/waterseg.py --shape 256 256
python tools/analysis_tools/get_flops.py configs-h/vision/pvsa_net_baseline.py --shape 256 256
python tools/analysis_tools/get_flops.py configs-h/clip/attn_waterseg.py --shape 256 256
# 分 stage 详细分析（仅完整 CLIP 版支持）
python tools/analysis_tools/clip_stage_complexity.py configs-h/clip/waterseg.py --shape 256 256
```

## 特征图和注意力可视化（CLIP gqy）
```bash
CUDA_VISIBLE_DEVICES=0 python tools/visualize_pvsa.py configs-h/clip/waterseg.py work_dirs/clip_waterseg/epoch_200.pth --image demo/demo.png --mode clip --device cuda:0 --query-index 32
CUDA_VISIBLE_DEVICES=0 python tools/visualize_pvsa.py configs-h/clip/waterseg.py work_dirs/clip_waterseg/epoch_200.pth 1 --mode clip --device cuda:0 --query-index 32
```

## 部署
```bash
python tools/deploy_clip_pvsa.py --config configs-h/clip/waterseg.py --checkpoint work_dirs/clip_waterseg/epoch_200.pth --output work_dirs/deployed/
```
部署后：backbone 文本注入使用预计算 frozen K/V；head 默认保留图相关原型池化与实时点积。严格 cosine 分类头含逐像素 L2 归一化，不能融合成单个 Conv2d。
> 注：部署脚本针对完整 CLIP 版（含 TTRM/Cross-Attn/TextEncoder）；纯视觉 baseline 不需要部署。

## CUDA 核加速推理
```bash
rm -rf ~/.cache/torch_extensions/py*/pvsa_topp_flash_cuda
export CC=/usr/bin/gcc-11 && export CXX=/usr/bin/g++-11
CUDA_VISIBLE_DEVICES=0 python tools/analysis_tools/benchmark.py configs-h/clip/waterseg.py work_dirs/clip_waterseg/epoch_200.pth --cfg-options model.backbone.topp_flash_backend=cuda model.backbone.topp_flash_debug=True
```
GPU 架构检测失败：`export PVSA_TOPP_FLASH_ARCH="8.6"`

## 注意事项
- 非训练脚本运行前需设置：`export PYTHONPATH=/media/ddc/新加卷/hys/hysnew3/PVSA/PVSA-Clip:$PYTHONPATH`
