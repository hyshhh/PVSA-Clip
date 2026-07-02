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
# 训练 - Baseline (gqy 数据集, 200 epoch)
CUDA_VISIBLE_DEVICES=0 python tools/train.py configs-h/biformer/biformer_baseline_waterseg.py --work-dir work_dirs/baseline
# 训练 - Baseline (CamVid 数据集, 400 epoch)
CUDA_VISIBLE_DEVICES=0 python tools/train.py configs-h/biformer/biformer_baseline_camvid.py --work-dir work_dirs/baseline_camvid
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
```bash
# 训练（切换模型后改 --work-dir）
CUDA_VISIBLE_DEVICES=0 python tools/train.py configs-h/biformer/baselines_compare.py --work-dir work_dirs/deeplabv3plus_r50
CUDA_VISIBLE_DEVICES=0 python tools/train.py configs-h/biformer/baselines_compare.py --work-dir work_dirs/swin_t_upernet
CUDA_VISIBLE_DEVICES=0 python tools/train.py configs-h/biformer/baselines_compare.py --work-dir work_dirs/segformer_b2
# 测试
CUDA_VISIBLE_DEVICES=0 python tools/test.py configs-h/biformer/baselines_compare.py work_dirs/<model>/best_mIoU.pth
```

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

## 融合消融实验
```bash
# 遍历 fusion_type × cross_stage_fusion_mode，逐组训练并汇总 best mIoU / FLOPs / Params
CUDA_VISIBLE_DEVICES=0 python tools/analysis_tools/run_fusion_ablation.py configs-h/biformer/biformer_baseline_camvid.py --work-dir-root work_dirs/fusion_ablation_camvid --shape 256 256 --skip-existing

# 如果实验已经跑完，只重刷 summary.csv，不重新训练
CUDA_VISIBLE_DEVICES=0 python tools/analysis_tools/run_fusion_ablation.py configs-h/biformer/biformer_baseline_camvid.py --work-dir-root work_dirs/fusion_ablation_camvid --shape 256 256 --summary-only
```
结果汇总会写到：`work_dirs/fusion_ablation_camvid/summary.csv`
汇总字段包括：`run_name,fusion_type,cross_stage_fusion_mode,flops,params,best_mIoU,status`

## 特征图和注意力可视化
```bash
# Baseline (gqy)
CUDA_VISIBLE_DEVICES=0 python tools/visualize_pvsa.py configs-h/biformer/biformer_baseline_waterseg.py work_dirs/baseline/epoch_200.pth --image demo/demo.png --mode baseline --device cuda:0 --query-index 32
CUDA_VISIBLE_DEVICES=0 python tools/visualize_pvsa.py configs-h/biformer/biformer_baseline_waterseg.py work_dirs/baseline/epoch_200.pth 1 --mode baseline --device cuda:0 --query-index 32
# Baseline (CamVid)
CUDA_VISIBLE_DEVICES=0 python tools/visualize_pvsa.py configs-h/biformer/biformer_baseline_camvid.py work_dirs/baseline_camvid/epoch_400.pth --image demo/demo.png --mode baseline --device cuda:0 --query-index 32
CUDA_VISIBLE_DEVICES=0 python tools/visualize_pvsa.py configs-h/biformer/biformer_baseline_camvid.py work_dirs/baseline_camvid/epoch_400.pth 1 --mode baseline --device cuda:0 --query-index 32
# CLIP (gqy)
CUDA_VISIBLE_DEVICES=0 python tools/visualize_pvsa.py configs-h/biformer/biformer_clip_waterseg.py work_dirs/clip_waterseg/epoch_200.pth --image demo/demo.png --mode clip --device cuda:0 --query-index 32
CUDA_VISIBLE_DEVICES=0 python tools/visualize_pvsa.py configs-h/biformer/biformer_clip_waterseg.py work_dirs/clip_waterseg/epoch_200.pth 1 --mode clip --device cuda:0 --query-index 32

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


## 注意事项
- 非训练脚本运行前需设置：`export PYTHONPATH=/media/ddc/新加卷/hys/hysnew3/PVSA/PVSA-Clip:$PYTHONPATH`
