# CLIP 路径消融实验命令

---

## 1. CLIP 分类头归一化消融

```bash
CUDA_VISIBLE_DEVICES=0 python ablation/run_clip_normalize_ablation.py \
  --work-dir-root ablation/clip-normalize \
  --shape 256 256 \
  --skip-existing
```

```bash
CUDA_VISIBLE_DEVICES=0 python ablation/run_clip_normalize_ablation.py \
  --work-dir-root ablation/clip-normalize \
  --variants norm-N0-topp-dot norm-N1-topp-cos \
  --shape 256 256 \
  --skip-existing
```

```bash
CUDA_VISIBLE_DEVICES=0 python ablation/run_clip_normalize_ablation.py \
  --work-dir-root ablation/clip-normalize \
  --dry-run
```

```bash
python ablation/run_clip_normalize_ablation.py \
  --work-dir-root ablation/clip-normalize \
  --summary-only
```

---

## 2. RepRTA 文本原型精炼消融

```bash
CUDA_VISIBLE_DEVICES=0 python ablation/run_clip_reprta_ablation.py \
  configs-h/clip/waterseg.py \
  --work-dir-root ablation/clip-RepRTA \
  --shape 256 256 \
  --skip-existing
```

```bash
CUDA_VISIBLE_DEVICES=0 python ablation/run_clip_reprta_ablation.py \
  configs-h/clip/waterseg.py \
  --work-dir-root ablation/clip-RepRTA \
  --variants reprta-R0-no reprta-R2-default \
  --shape 256 256 \
  --skip-existing
```

```bash
CUDA_VISIBLE_DEVICES=0 python ablation/run_clip_reprta_ablation.py \
  configs-h/clip/waterseg.py \
  --work-dir-root ablation/clip-RepRTA \
  --dry-run
```

```bash
python ablation/run_clip_reprta_ablation.py \
  configs-h/clip/waterseg.py \
  --work-dir-root ablation/clip-RepRTA \
  --summary-only
```

---

## 3. CLIP 框架消融对比（B0 纯视觉 vs B1 CLIP+BiFormer vs B2 CLIP+Topp）

```bash
CUDA_VISIBLE_DEVICES=0 python ablation/run_framework_ablation.py \
  --work-dir-root ablation/clip-framework \
  --shape 256 256 \
  --skip-existing
```

```bash
CUDA_VISIBLE_DEVICES=0 python ablation/run_framework_ablation.py \
  --work-dir-root ablation/clip-framework \
  --variants framework-B0-pvsa-vision framework-B2-clip-topp \
  --shape 256 256 \
  --skip-existing
```

```bash
CUDA_VISIBLE_DEVICES=0 python ablation/run_framework_ablation.py \
  --work-dir-root ablation/clip-framework \
  --dry-run
```

```bash
python ablation/run_framework_ablation.py \
  --work-dir-root ablation/clip-framework \
  --summary-only
```
