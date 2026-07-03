# PVSA-Net

基于 MMSegmentation 的语义分割项目，支持四条路径：

1. **CLIP 增强路径**：注入 CLIP 文本语义（TTRM 路由级 + Cross-Attn 特征级），部署时零额外开销。
2. **PVSA-Net Baseline 路径**：CLIP 版 backbone 的纯视觉对照，移除全部文本信号，仅保留 ToppAttention / 普通 attention block 切换。
3. **标准 BiFormer Attention 消融**：保留完整 CLIP 文本路径，仅将 backbone 注意力从 ToppAttention 替换为 BiLevelRoutingAttention。
4. **Baseline 路径**（原始）：非 CLIP，含 CNN 分支 + 融合，soft routing，用于跨模型对比实验。

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
  configs-h/clip/waterseg.py \
  --work-dir work_dirs/clip_waterseg
```

Baseline 训练（非 CLIP，soft routing）：
```bash
CUDA_VISIBLE_DEVICES=0 python tools/train.py \
  configs-h/vision/baseline_waterseg.py \
  --work-dir work_dirs/baseline
```

## CLIP 模型推理

```bash
export PYTHONPATH=/media/ddc/新加卷/hys/hysnew3/PVSA/PVSA-Clip:$PYTHONPATH
CUDA_VISIBLE_DEVICES=0 python tools/analysis_tools/benchmark.py \
  configs-h/clip/waterseg.py \
  work_dirs/clip_waterseg/best_mIoU.pth \
  --cfg-options model.backbone.topp_flash_backend=None
```

## 推理并保存分割结果

```bash
export PYTHONPATH=/media/ddc/新加卷/hys/hysnew3/PVSA/PVSA-Clip:$PYTHONPATH
CUDA_VISIBLE_DEVICES=0 python tools/test.py \
  configs-h/clip/waterseg.py \
  work_dirs/clip_waterseg/best_mIoU.pth \
  --show-dir vis_results/
```

## 部署（CLIP 路径）

```bash
python tools/deploy_clip_pvsa.py \
  --config configs-h/clip/waterseg.py \
  --checkpoint work_dirs/clip_waterseg/best_mIoU.pth \
  --output work_dirs/deployed/
```

融合步骤：冻结 prototypes → BN+Conv2d 融合 → TTRM 文本投影预计算 → TextCrossAttention K/V 预计算 → 跳过 RepRTA → 输出零文本开销模型。

## 复杂度统计

```bash
export PYTHONPATH=/media/ddc/新加卷/hys/hysnew3/PVSA/PVSA-Clip:$PYTHONPATH
python tools/analysis_tools/get_flops.py \
  configs-h/clip/waterseg.py --shape 256 256
python tools/analysis_tools/pvsa_stage_complexity.py \
  configs-h/clip/waterseg.py --shape 256 256
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
  configs-h/clip/waterseg.py \
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

### 模型定义（`_base_/models/`）
| 文件 | 说明 |
|------|------|
| `configs-h/_base_/models/clip-topp.py` | CLIP 增强模型（ToppAttention + 全套文本） |
| `configs-h/_base_/models/clip-brg.py` | CLIP + 标准 BiFormer Attention 消融 |
| `configs-h/_base_/models/vision-topp.py` | PVSA-Net Baseline（CLIP 版 backbone 纯视觉） |
| `configs-h/_base_/models/vision-brg.py` | 纯视觉 + 标准 BiFormer Attention 消融 |
| `configs-h/_base_/models/vision-topp-cnn.py` | 原 Baseline 模型（含 CNN 分支） |

### 训练入口
| 文件 | 说明 |
|------|------|
| `configs-h/clip/waterseg.py` | CLIP 增强训练 |
| `configs-h/clip/attn_waterseg.py` | CLIP + 标准 BiFormer Attention 消融 |
| `configs-h/vision/pvsa_net_baseline.py` | PVSA-Net Baseline 训练（CLIP 版 backbone 纯视觉对照） |
| `configs-h/vision/attn_ablation_waterseg.py` | 纯视觉 + 标准 BiFormer Attention 消融 |
| `configs-h/vision/baseline_waterseg.py` | 原 Baseline 训练（gqy） |
| `configs-h/vision/baseline_camvid.py` | 原 Baseline 训练（CamVid） |
| `configs-h/vision/baselines_compare.py` | 对比实验（切换 deeplabv3plus/swin_t/segformer_b2） |

详细命令见 [docs/USAGE.md](docs/USAGE.md)（纯视觉路径）与 [docs/USAGE-clip.md](docs/USAGE-clip.md)（CLIP 路径）。

## 注意事项

- 自定义 CUDA 核路径只面向推理，不用于训练。
- 真实速度测试时不要打开调试日志。
- 如果需要调整 `energy`、`p`、`temperature`、`maxk`，请修改配置文件中的 `topp_route_configs`。
- CLIP Text Encoder 训练时冻结，TextCrossAttention 推理时用预计算 K/V。
- 部署后模型等价于原始 PVSA-Net + 1×1 Conv，零额外推理开销。
- 本地不具备 CUDA 编译环境，CUDA 编译和性能验证以服务器结果为准。
