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

## 我们的模型（PVSA-Net）
```bash
# 训练 - Baseline
CUDA_VISIBLE_DEVICES=0 python tools/train.py configs-h/biformer/biformer_baseline_waterseg.py --work-dir work_dirs/baseline
# 训练 - CLIP
CUDA_VISIBLE_DEVICES=0 python tools/train.py configs-h/biformer/biformer_clip_waterseg.py --work-dir work_dirs/clip_waterseg
# 推理 - Baseline
CUDA_VISIBLE_DEVICES=0 python tools/analysis_tools/benchmark.py configs-h/biformer/biformer_baseline_waterseg.py work_dirs/baseline/epoch_200.pth
# 推理 - CLIP
CUDA_VISIBLE_DEVICES=0 python tools/analysis_tools/benchmark.py configs-h/biformer/biformer_clip_waterseg.py work_dirs/clip_waterseg/epoch_200.pth
# 保存分割可视化
CUDA_VISIBLE_DEVICES=0 python tools/test.py configs-h/biformer/biformer_baseline_waterseg.py work_dirs/baseline/epoch_200.pth --show-dir vis_results/baseline/
CUDA_VISIBLE_DEVICES=0 python tools/test.py configs-h/biformer/biformer_clip_waterseg.py work_dirs/clip_waterseg/epoch_200.pth --show-dir vis_results/clip/
```

## 对比实验

支持的对比模型（配置在 `configs-h/_base_/compare_models/`）：

| 模型 | Backbone | Decode Head | 配置文件 |
|------|----------|-------------|---------|
| DeepLabV3+ | ResNet-50 | ASPP Head | `deeplabv3plus_r50.py` |
| Swin-T + UPerNet | Swin-Tiny | UPerHead | `swin_t_upernet.py` |
| SegFormer-B2 | MiT-B2 | SegformerHead | `segformer_b2.py` |

切换模型：编辑 `configs-h/biformer/baselines_compare.py`，取消注释对应的 `_base_` 行。
```bash
# 训练（切换模型后改 --work-dir）
CUDA_VISIBLE_DEVICES=0 python tools/train.py configs-h/biformer/baselines_compare.py --work-dir work_dirs/deeplabv3plus_r50
CUDA_VISIBLE_DEVICES=0 python tools/train.py configs-h/biformer/baselines_compare.py --work-dir work_dirs/swin_t_upernet
CUDA_VISIBLE_DEVICES=0 python tools/train.py configs-h/biformer/baselines_compare.py --work-dir work_dirs/segformer_b2
# 测试
CUDA_VISIBLE_DEVICES=0 python tools/test.py configs-h/biformer/baselines_compare.py work_dirs/<model>/best_mIoU.pth
```

预训练权重：ResNet-50 由 mmseg 自动下载；Swin-T 需手动下载 [权重](https://download.openmmlab.com/mmpretrain/v1.0/swin/swin-tiny_3rdparty_16xb64_in1k_300e/swin-tiny_3rdparty_16xb64_in1k_300e_20230912_224855-e6a0c6bf.pth)；MiT-B2 需手动下载 [权重](https://download.openmmlab.com/mmsegmentation/v0.5/segformer/segformer_mit-b2_512x512_160k_ade20k/segformer_mit-b2_512x512_160k_ade20k_20220617_164113-83a7b3e6.pth)。在 backbone config 中加 `init_cfg=dict(type='Pretrained', checkpoint='path/to/weight.pth')`。

训练设置对比：

| 设置 | PVSA-Net | DeepLabV3+ | Swin-T | SegFormer-B2 |
|------|----------|------------|--------|-------------|
| 优化器 | AdamW | SGD | AdamW | AdamW |
| 学习率 | 6e-4 | 0.01 | 6e-4 | 6e-4 |
| LR策略 | PolyLR(p=1.0) | PolyLR(p=0.9) | PolyLR(p=1.0) | PolyLR(p=1.0) |
| 轮次/Batch | 200 / 16 | 200 / 16 | 200 / 16 | 200 / 16 |
| 输入尺寸 | 256x256 | 256x256 | 256x256 | 256x256 |

## 参数量与复杂度
```bash
# FLOPs + Params（所有模型通用）
python tools/analysis_tools/get_flops.py configs-h/biformer/biformer_baseline_waterseg.py --shape 256 256
python tools/analysis_tools/get_flops.py configs-h/biformer/biformer_clip_waterseg.py --shape 256 256
python tools/analysis_tools/get_flops.py configs-h/biformer/baselines_compare.py --shape 256 256
# 分 stage 详细分析
python tools/analysis_tools/pvsa_stage_complexity.py configs-h/biformer/biformer_baseline_waterseg.py --shape 256 256
python tools/analysis_tools/clip_stage_complexity.py configs-h/biformer/biformer_clip_waterseg.py --shape 256 256
```

## 特征图和注意力可视化
```bash
# Baseline
CUDA_VISIBLE_DEVICES=0 python tools/visualize_pvsa.py configs-h/biformer/biformer_baseline_waterseg.py work_dirs/baseline/epoch_200.pth --image demo/demo.png --mode baseline --device cuda:0 --query-index 32
CUDA_VISIBLE_DEVICES=0 python tools/visualize_pvsa.py configs-h/biformer/biformer_baseline_waterseg.py work_dirs/baseline/epoch_200.pth 1 --mode baseline --device cuda:0 --query-index 32
# CLIP
CUDA_VISIBLE_DEVICES=0 python tools/visualize_pvsa.py configs-h/biformer/biformer_clip_waterseg.py work_dirs/clip_waterseg/epoch_200.pth --image demo/demo.png --mode clip --device cuda:0 --query-index 32
CUDA_VISIBLE_DEVICES=0 python tools/visualize_pvsa.py configs-h/biformer/biformer_clip_waterseg.py work_dirs/clip_waterseg/epoch_200.pth 1 --mode clip --device cuda:0 --query-index 32
```
`1` 表示测试集第 1 张图，也可写 `--test-index 1`。保留 `--image` 时优先使用手动路径。加 `--single-route --route-stage 0 --route-block 0` 只看单个路由块。

保存位置：特征图 -> `demo/feathermap/<mode>/<图片名>/`，注意力图 -> `demo/attension_map/<mode>/<图片名>/`。
注意力图子目录：`route_scores/`（全量窗口权重）、`route_scores_topp/`（Top-P 截断后软路由分数）、`top_p_mask/`（实际选中窗口掩码）。CLIP 路径额外保存 `text_injection/`（文本注入前后对比）。

## 部署
```bash
python tools/deploy_clip_pvsa.py --config configs-h/biformer/biformer_clip_waterseg.py --checkpoint work_dirs/clip_waterseg/epoch_200.pth --output work_dirs/deployed/
```
部署后：TextEncoder 移除，Head 融合为 Conv2d，TTRM/Cross-Attn 使用预计算 frozen K/V。

## CUDA 核加速推理
```bash
rm -rf ~/.cache/torch_extensions/py*/pvsa_topp_flash_cuda
export CC=/usr/bin/gcc-11 && export CXX=/usr/bin/g++-11
CUDA_VISIBLE_DEVICES=0 python tools/analysis_tools/benchmark.py configs-h/biformer/biformer_clip_waterseg.py work_dirs/clip_waterseg/epoch_200.pth --cfg-options model.backbone.topp_flash_backend=cuda model.backbone.topp_flash_debug=True
```
GPU 架构检测失败：`export PVSA_TOPP_FLASH_ARCH="8.6"`

## 配置文件
| 文件 | 说明 |
|------|------|
| `configs-h/_base_/models/VTFormer-s-baseline.py` | Baseline 模型 |
| `configs-h/_base_/models/VTFormer-clip.py` | CLIP 模型 |
| `configs-h/_base_/compare_models/` | 对比模型（DeepLabV3+/Swin-T/SegFormer） |
| `configs-h/biformer/biformer_baseline_waterseg.py` | Baseline 训练入口 |
| `configs-h/biformer/biformer_clip_waterseg.py` | CLIP 训练入口 |
| `configs-h/biformer/baselines_compare.py` | 对比实验入口 |

## 注意事项
- 非训练脚本运行前需设置：`export PYTHONPATH=/media/ddc/新加卷/hys/hysnew3/PVSA/PVSA-Clip:$PYTHONPATH`
- 调整 `energy`/`p`/`temperature`/`maxk` 修改配置中的 `topp_route_configs`。
- CLIP Text Encoder 训练时冻结，部署推理时移除。
- 切换对比模型后必须改 `--work-dir`，避免覆盖。num_classes=3 对应 gqy 数据集。
- 显存不够可调小 batch_size 并线性缩放学习率。
- 服务器训练前确保在 `pvsa-v3.0` 分支：`git checkout pvsa-v3.0 && git pull`
