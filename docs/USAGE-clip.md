# PVSA-Clip（CLIP 路径）操作命令

本文件只保留 **CLIP 文本引导路径** 的常用命令。
纯视觉基线请看 [USAGE.md](USAGE.md)。

当前 `configs-h/clip/*.py` 默认使用 `KAKA` 数据集，并继承：
- `configs-h/_base_/models/clip-topp.py`

标准注意力消融入口 `configs-h/clip/attn_waterseg.py` 也继承同一个模型基配置，只覆盖 backbone 注意力。

## 环境准备

```bash
pip install -r requirements/mminstall.txt
pip install -r requirements/runtime.txt
pip install openai-clip
export PYTHONPATH=$(pwd):$PYTHONPATH
```

## 生成 Prompt Bank

CLIP 路径必须先准备文本原型。

```bash
wget -O tools/ViT-B-32.pt https://openaipublic.azureedge.net/clip/models/40d365715913c9da98579312b702a82c18be219cc2a73407c4526f58eba950af/ViT-B-32.pt
python tools/generate_water_prompt_bank.py --output tools/prompt_bank_water.pt --model ViT-B/32 --model-path tools/ViT-B-32.pt
```

## 训练

### 完整 CLIP 路径

```bash
CUDA_VISIBLE_DEVICES=0 python tools/train.py configs-h/clip/waterseg.py --work-dir work_dirs/clip_waterseg
```

### 标准注意力消融

保留完整文本路径，只把主干注意力从 `ToppAttention` 换成标准 `BiFormer Attention`。

```bash
CUDA_VISIBLE_DEVICES=0 python tools/train.py configs-h/clip/attn_waterseg.py --work-dir work_dirs/clip_biformer_attn
```

### 普通分割头消融

`configs-h/_base_/models/clip-topp.py` 已支持普通分割头开关：

```python
use_clip_decode_head = False
```

切换后，继续使用同一个训练命令即可，建议单独指定输出目录：

```bash
CUDA_VISIBLE_DEVICES=0 python tools/train.py configs-h/clip/waterseg.py --work-dir work_dirs/clip_waterseg_seghead
```

## 推理与测速

### 完整 CLIP 路径

```bash
CUDA_VISIBLE_DEVICES=0 python tools/analysis_tools/benchmark.py configs-h/clip/waterseg.py work_dirs/clip_waterseg/epoch_200.pth
```

### 标准注意力消融

```bash
CUDA_VISIBLE_DEVICES=0 python tools/analysis_tools/benchmark.py configs-h/clip/attn_waterseg.py work_dirs/clip_biformer_attn/epoch_200.pth
```

### 普通分割头消融

在 `use_clip_decode_head = False` 的前提下运行：

```bash
CUDA_VISIBLE_DEVICES=0 python tools/analysis_tools/benchmark.py configs-h/clip/waterseg.py work_dirs/clip_waterseg_seghead/epoch_200.pth
```

## 保存分割可视化

```bash
CUDA_VISIBLE_DEVICES=0 python tools/test.py configs-h/clip/waterseg.py work_dirs/clip_waterseg/epoch_200.pth --show-dir vis_results/clip
```

## 参数量与复杂度

```bash
python tools/analysis_tools/get_flops.py configs-h/clip/waterseg.py --shape 256 256
python tools/analysis_tools/get_flops.py configs-h/clip/attn_waterseg.py --shape 256 256
python tools/analysis_tools/clip_stage_complexity.py configs-h/clip/waterseg.py --shape 256 256
```

## 特征图与注意力可视化

```bash
CUDA_VISIBLE_DEVICES=0 python tools/visualize_pvsa.py configs-h/clip/waterseg.py work_dirs/clip_waterseg/epoch_200.pth --image demo/demo.png --mode clip --device cuda:0 --query-index 32
```

## 部署

```bash
python tools/deploy_clip_pvsa.py --config configs-h/clip/waterseg.py --checkpoint work_dirs/clip_waterseg/epoch_200.pth --output work_dirs/deployed
```

说明：
- 部署脚本面向完整 CLIP 路径，包含 `TTRM`、`Cross-Attn` 和 `TextEncoder`。
- 主干文本注入会冻结为预计算 `K/V`。
- 头部默认保留图相关原型池化和实时点积分类。
- 如果使用严格余弦分类头，逐像素 `L2` 归一化无法融合成单个 `Conv2d`。

## CUDA 核加速推理

```bash
rm -rf ~/.cache/torch_extensions/py*/pvsa_topp_flash_cuda
export CC=/usr/bin/gcc-11
export CXX=/usr/bin/g++-11
CUDA_VISIBLE_DEVICES=0 python tools/analysis_tools/benchmark.py configs-h/clip/waterseg.py work_dirs/clip_waterseg/epoch_200.pth --cfg-options model.backbone.topp_flash_backend=cuda model.backbone.topp_flash_debug=True
```

如果 GPU 架构自动检测失败，可以手动指定：

```bash
export PVSA_TOPP_FLASH_ARCH="8.6"
```
