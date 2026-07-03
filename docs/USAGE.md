# PVSA-Clip 操作指南（Baseline 纯视觉路径）

本文件只保留 **纯视觉 baseline** 相关命令（无 CLIP 文本注入）。
CLIP 文本引导路径全部命令见 [USAGE-clip.md](USAGE-clip.md)。

## 环境安装
```bash
pip install -r requirements/mminstall.txt && pip install -r requirements/runtime.txt
```
> baseline 不依赖 openai-clip 与 Prompt Bank，仅需上述基础环境。

## 训练
```bash
# 训练 - Baseline (gqy 数据集, 200 epoch)
CUDA_VISIBLE_DEVICES=0 python tools/train.py configs-h/vision/baseline_waterseg.py --work-dir work_dirs/baseline
# 训练 - Baseline (CamVid 数据集, 400 epoch)
CUDA_VISIBLE_DEVICES=0 python tools/train.py configs-h/vision/baseline_camvid.py --work-dir work_dirs/baseline_camvid
# 训练 - BiFormer Attention 消融（标准双层路由 attention 替代 ToppAttention, 200 epoch）
CUDA_VISIBLE_DEVICES=0 python tools/train.py configs-h/vision/attn_ablation_waterseg.py --work-dir work_dirs/biformer_attn
```

## 推理
```bash
# 推理 - Baseline
CUDA_VISIBLE_DEVICES=0 python tools/analysis_tools/benchmark.py configs-h/vision/baseline_waterseg.py work_dirs/baseline/epoch_200.pth
# 推理 - BiFormer Attention 消融
CUDA_VISIBLE_DEVICES=0 python tools/analysis_tools/benchmark.py configs-h/vision/attn_ablation_waterseg.py work_dirs/biformer_attn/epoch_200.pth
```

## 保存分割可视化
```bash
CUDA_VISIBLE_DEVICES=0 python tools/test.py configs-h/vision/baseline_waterseg.py work_dirs/baseline/epoch_200.pth --show-dir vis_results/baseline/
CUDA_VISIBLE_DEVICES=0 python tools/test.py configs-h/vision/attn_ablation_waterseg.py work_dirs/biformer_attn/epoch_200.pth --show-dir vis_results/biformer_attn/
```

## 对比实验（共享 baselines_compare 配置切换模型）
```bash
# 训练（切换模型后改 --work-dir）
CUDA_VISIBLE_DEVICES=0 python tools/train.py configs-h/vision/baselines_compare.py --work-dir work_dirs/deeplabv3plus_r50
CUDA_VISIBLE_DEVICES=0 python tools/train.py configs-h/vision/baselines_compare.py --work-dir work_dirs/swin_t_upernet
CUDA_VISIBLE_DEVICES=0 python tools/train.py configs-h/vision/baselines_compare.py --work-dir work_dirs/segformer_b2
# 测试
CUDA_VISIBLE_DEVICES=0 python tools/test.py configs-h/vision/baselines_compare.py work_dirs/<model>/best_mIoU.pth
```

## 参数量与复杂度
```bash
# FLOPs + Params
python tools/analysis_tools/get_flops.py configs-h/vision/baseline_waterseg.py --shape 256 256
python tools/analysis_tools/get_flops.py configs-h/vision/attn_ablation_waterseg.py --shape 256 256
python tools/analysis_tools/get_flops.py configs-h/vision/baselines_compare.py --shape 256 256
# 分 stage 详细分析
python tools/analysis_tools/pvsa_stage_complexity.py configs-h/vision/baseline_waterseg.py --shape 256 256
```

## 融合消融实验
```bash
# 遍历 fusion_type × cross_stage_fusion_mode，逐组训练并汇总 best mIoU / FLOPs / Params
CUDA_VISIBLE_DEVICES=0 python tools/analysis_tools/run_fusion_ablation.py configs-h/vision/baseline_camvid.py --work-dir-root work_dirs/fusion_ablation_camvid --shape 256 256 --skip-existing

# 如果实验已经跑完，只重刷 summary.csv，不重新训练
CUDA_VISIBLE_DEVICES=0 python tools/analysis_tools/run_fusion_ablation.py configs-h/vision/baseline_camvid.py --work-dir-root work_dirs/fusion_ablation_camvid --shape 256 256 --summary-only
```
结果汇总会写到：`work_dirs/fusion_ablation_camvid/summary.csv`
汇总字段包括：`run_name,fusion_type,cross_stage_fusion_mode,flops,params,best_mIoU,status`

## 特征图和注意力可视化（Baseline）
```bash
# Baseline (gqy)
CUDA_VISIBLE_DEVICES=0 python tools/visualize_pvsa.py configs-h/vision/baseline_waterseg.py work_dirs/baseline/epoch_200.pth --image demo/demo.png --mode baseline --device cuda:0 --query-index 32
CUDA_VISIBLE_DEVICES=0 python tools/visualize_pvsa.py configs-h/vision/baseline_waterseg.py work_dirs/baseline/epoch_200.pth 1 --mode baseline --device cuda:0 --query-index 32
# Baseline (CamVid)
CUDA_VISIBLE_DEVICES=0 python tools/visualize_pvsa.py configs-h/vision/baseline_camvid.py work_dirs/baseline_camvid/epoch_400.pth --image demo/demo.png --mode baseline --device cuda:0 --query-index 32
CUDA_VISIBLE_DEVICES=0 python tools/visualize_pvsa.py configs-h/vision/baseline_camvid.py work_dirs/baseline_camvid/epoch_400.pth 1 --mode baseline --device cuda:0 --query-index 32
```

## 注意事项
- 非训练脚本运行前需设置：`export PYTHONPATH=/media/ddc/新加卷/hys/hysnew3/PVSA/PVSA-Clip:$PYTHONPATH`
