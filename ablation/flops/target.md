
# 参数量与 FLOPs 统计脚本提示词

请为消融实验编写一个**参数量与 FLOPs 统计脚本**，目标是对本目录下所有消融实验的模型逐组统计参数量（Params）与 FLOPs，并将结果汇总输出。具体要求如下：

## 一、基本要求

1. **统计目标**：以 `../_base_/models/vision-topp-cnn.py` 链路为基准配置，以此作为**基线（Baseline）**统一统计。
2. **脚本位置**：将生成的统计脚本放到本目录 `ablation/flops` 下，用一个脚本完成全部实验的统计。
3. **配置替换方式**：每一项消融实验在基线基础上，由脚本**动态替换**原有配置文件中的对应配置项来完成统计，**不得修改原配置文件本身**。统计完成后汇总结果。

## 二、实验组清单

以下为全部需要统计的实验组，已统一编号。请按编号逐组完成统计。

### 实验组 1：整体消融（Overall Ablation）

> 论文描述：the features extracted by the MBConv block and the BiFormer block are directly concatenated without any dedicated feature fusion or refinement modules.

| 序号 | FFM | PVSA | VFM |
| ---- | --- | ---- | --- |
| 1-1  | ×   | ×    | ×   |
| 1-2  | √   | ×    | ×   |
| 1-3  | ×   | √    | ×   |
| 1-4  | √   | ×    | √   |
| 1-5  | **√** | **√** | **√** |

### 实验组 2：骨干消融（Backbone Ablation）

> 论文描述：when MBConv is set to (0,0,0,0), the network consists solely of the Transformer branch. The Transformer branch employs BiFormer-M blocks, while the CNN branch uses MBConv blocks. C+T and T+C denote sequential designs with different ordering. TC1 represents a parallel dual-branch with direct concatenation, while TC2 builds on TC1 by incorporating the proposed FFM and VFM for feature fusion.

#### 2-A：MBConv-Block 与 PVSA-Block 配置组合

| 序号 | MBConv-Block | PVSA-Block  |
| ---- | ------------ | ----------- |
| 2-A-1 | 0,0,0,0      | 3,4,6,3     |
| 2-A-2 | 1,1,1,1      | 3,4,6,3     |
| 2-A-3 | 1,2,2,1      | 1,3,4,2     |
| 2-A-4 | 1,2,2,1      | 2,6,8,4     |
| 2-A-5 | **1,2,2,1**  | **3,4,6,3** |

#### 2-B：融合方式对比

| 序号 | Method |
| ---- | ------ |
| 2-B-1 | C+T    |
| 2-B-2 | T+C    |
| 2-B-3 | TC1    |
| 2-B-4 | TC2    |

## 三、输出要求

- 按「实验组 / 序号」维度汇总每组的 Params 与 FLOPs。
- 基线（Baseline）单独列出，便于对照。
- 输出最终汇总表，清晰展示各组结果。
