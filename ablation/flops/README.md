# 消融实验 FLOPs / Params 统计

## 快速开始

```bash
# 默认输入尺寸 256×256
python ablation/flops/ablation_flops.py

# 指定输入尺寸
python ablation/flops/ablation_flops.py --shape 512 512
python ablation/flops/ablation_flops.py --shape 256 256
```

## 输出

- 终端打印完整汇总表
- 自动保存 CSV: `ablation/flops/ablation_results.csv`

## 实验组清单

| 组 | 对比维度 | 实验数 |
|----|----------|--------|
| 1 | FAM × PVSA × VFM 三开关 | 5 |
| 2-A | MBConv / PVSA 深度组合 | 5 |
| 2-B | 融合方式（顺序/并行） | 4 |
| 3 | Full 基础上减 CA / SA | 3 |

## 单独跑某个实验

修改 `ablation_flops.py` 中 `EXPERIMENTS` 列表，只保留目标实验即可。
