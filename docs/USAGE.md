# PVSA-CLIP 操作指南

## 环境安装

```bash
pip install -r requirements/mminstall.txt
pip install -r requirements/runtime.txt
pip install openai-clip
```

## 生成 Prompt Bank

```bash
python tools/generate_water_prompt_bank.py --output tools/prompt_bank_water.pt --model ViT-B/32
```

## 训练

CLIP 增强训练：
```bash
CUDA_VISIBLE_DEVICES=0 python tools/train.py configs-h/biformer/biformer_clip_waterseg.py --work-dir work_dirs/clip_waterseg
```

原始 PVSA-Net 训练：
```bash
CUDA_VISIBLE_DEVICES=0 python tools/train.py configs-h/biformer/biformer_mm-20k_chase_db1-512x512.py --cfg-options model.backbone.topp_flash_backend=None model.backbone.feature_vis_config.enabled=False model.backbone.attn_vis_config.enabled=False train_dataloader.batch_size=16 --work-dir work_dirs/pvsa_baseline
```

梯度尖峰调试：
```bash
CUDA_VISIBLE_DEVICES=0 python tools/train.py configs-h/biformer/biformer_mm-20k_chase_db1-512x512.py --cfg-options grad_spike_debug=True --work-dir work_dirs/pvsa_debug
```

## 部署

```bash
python tools/deploy_clip_pvsa.py --config configs-h/biformer/biformer_clip_waterseg.py --checkpoint work_dirs/clip_waterseg/best_mIoU_epoch.pth --output work_dirs/deployed/
```

融合步骤：冻结 prototypes → BN+Conv2d 融合 → 移除 CPFM → 跳过 RepRTA → 烘焙 TTRM α → 输出零文本开销模型。

> 注：训练完成后 `work_dirs/clip_waterseg/` 下会有 `best_mIoU_epoch.pth`（最优）和 `epoch_X.pth`（定期保存），部署用最优的那个。

## 推理

原始路径：
```bash
CUDA_VISIBLE_DEVICES=0 python tools/analysis_tools/benchmark.py configs-h/biformer/biformer_mm-20k_chase_db1-512x512.py checkpoint.pth --cfg-options model.backbone.topp_flash_backend=None
```

CLIP 模型：
```bash
CUDA_VISIBLE_DEVICES=0 python tools/analysis_tools/benchmark.py configs-h/biformer/biformer_clip_waterseg.py work_dirs/deployed/deployed_model.pth --cfg-options model.backbone.topp_flash_backend=None
```

保存分割结果：
```bash
CUDA_VISIBLE_DEVICES=0 python tools/test.py configs-h/biformer/biformer_clip_waterseg.py checkpoint.pth --show-dir vis_results/
```

## 复杂度统计

```bash
python tools/analysis_tools/get_flops.py configs-h/biformer/biformer_clip_waterseg.py --shape 224 224
python tools/analysis_tools/pvsa_stage_complexity.py configs-h/biformer/biformer_clip_waterseg.py --shape 224 224
```

## 自定义 CUDA 核推理

```bash
rm -rf ~/.cache/torch_extensions/py*/pvsa_topp_flash_cuda
export PYTHONPATH=$(pwd):$PYTHONPATH
export CC=/usr/bin/gcc-11 && export CXX=/usr/bin/g++-11
CUDA_VISIBLE_DEVICES=0 python tools/analysis_tools/benchmark.py configs-h/biformer/biformer_clip_waterseg.py checkpoint.pth --cfg-options model.backbone.topp_flash_backend=cuda model.backbone.topp_flash_debug=True
```

GPU 架构检测失败时：`export PVSA_TOPP_FLASH_ARCH="8.6"`

## 配置文件

| 文件 | 说明 |
|------|------|
| `configs-h/_base_/models/VTFormer-s.py` | 原始 PVSA-Net |
| `configs-h/_base_/models/VTFormer-clip.py` | CLIP 增强 |
| `configs-h/biformer/biformer_mm-20k_chase_db1-512x512.py` | 原始训练 |
| `configs-h/biformer/biformer_clip_waterseg.py` | CLIP 训练 |

## 关键文件

| 文件 | 说明 |
|------|------|
| `mmseg/models/utils/top_p_bra.py` | TTRM |
| `mmseg/models/utils/text_encoder.py` | TextEncoder |
| `mmseg/models/utils/cpfm.py` | CPFM |
| `mmseg/models/backbones/bi_topp_vote.py` | VTFormer |
| `mmseg/models/backbones/biformer_fusion.py` | BiFormer_fusion |
| `mmseg/models/decode_heads/clip_seg_head.py` | CLIPSegHead |
| `mmseg/models/segmentors/clip_encoder_decoder.py` | CLIPEncoderDecoder |
| `tools/generate_water_prompt_bank.py` | Prompt bank 生成 |
| `tools/deploy_clip_pvsa.py` | 部署融合 |
| `docs/architecture-plan.md` | 架构设计 |

## 注意事项

- CUDA 核路径只面向推理，不用于训练。
- 调整 `energy`/`p`/`temperature`/`maxk` 修改配置中的 `topp_route_configs`。
- CLIP Text Encoder 训练时冻结，CPFM 推理时移除。
- 部署后模型等价于原始 PVSA-Net + 1x1 Conv，零额外推理开销。
