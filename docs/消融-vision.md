# Vision 路径消融实验命令

本文件汇总所有基于 `configs-h/vision/` 配置的消融实验，对应纯视觉骨干（无 CLIP 文本注入）路径。

---

## 1. 骨干 CNN Block 消融

支持的 `cnn_block_type`：

```python
'dwconv'
'dwconv_act'
'mbconv'
'mbconv_no_se'
'c2f'
'c3k2'
'convnext'
```

跑全部骨干块消融：

```bash
CUDA_VISIBLE_DEVICES=0 python ablation/run_backbone_block_ablation.py \
  configs-h/vision/baseline_camvid.py \
  --work-dir-root ablation/vision-backbone \
  --shape 256 256 \
  --skip-existing
```

只跑指定几组：

```bash
CUDA_VISIBLE_DEVICES=0 python ablation/run_backbone_block_ablation.py \
  configs-h/vision/baseline_camvid.py \
  --work-dir-root ablation/vision-backbone \
  --cnn-block-types c2f c3k2 convnext \
  --shape 256 256 \
  --skip-existing
```

只重新汇总已有结果：

```bash
python ablation/run_backbone_block_ablation.py \
  configs-h/vision/baseline_camvid.py \
  --work-dir-root ablation/vision-backbone \
  --summary-only
```

汇总文件：

```text
ablation/vision-backbone/summary.csv
```

汇总字段：

```text
run_name,cnn_block_type,flops,params,best_mIoU,status
```
