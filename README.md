# PVSA-Net: Top-P Voting Sparse Attention Network

> 分支备注：`pvsa-v3.0` 是当前主线和仓库默认分支；`main`、`pvsa-v2.0`、`backup/before-topp-mask-20260611` 仅作为历史备份保留。后续开发、训练修复和结果复现请优先基于 `pvsa-v3.0`。

基于 MMSegmentation 的语义分割框架，核心创新是 **Top-P 投票稀疏注意力机制**（ToppAttention）。

## 核心特性

### Top-P 注意力机制
传统 Top-K 注意力固定选择 K 个最相关的窗口，而 Top-P 注意力通过**累积概率阈值**动态确定参与计算的窗口数量：
- 对窗口级注意力分数做 Softmax（带温度缩放）
- 按累积概率 `cumsum <= P` 进行截断
- 保留概率质量集中的窗口，自动过滤噪声

### 四种计算后端
| 后端 | 配置 | 显存 | 速度 | 适用场景 |
|------|------|------|------|----------|
| **kv_gather** | `use_topp_flash=False` | 高 | 快 | 显存充足时使用 |
| **pruned_kv_gather** | `use_pruned_kv_gather=True` | 高 | 中 | 按 keep_len 裁剪无效路由 |
| **torch_block** | `backend='torch_block'` | 中 | 中 | 默认推荐，显存受限 |
| **cuda** | `backend='cuda'` | 低 | 慢 | 极致显存优化，需编译环境 |

### Top-P 参数配置
| 原 topk | 实际 topk | P 阈值 | 温度 | 能量补偿 |
|---------|----------|--------|------|----------|
| 16 | 25 | 0.2 | 0.0175 | 4 |
| 12 | 18 | 0.4 | 0.025 | 1.5 |
| 8 | 36 | 0.6 | 0.05 | 0.75 |
| 6 | 49 | 0.8 | 0.15 | 0.4 |

## 项目结构
```
PVSA-Net/
├── mmseg/
│   ├── models/
│   │   ├── backbones/
│   │   │   ├── bi_topp_vote.py      # VTFormer 骨干网络
│   │   │   └── biformer_fusion.py   # 双路融合骨干
│   │   ├── utils/
│   │   │   ├── top_p_bra.py         # ToppAttention 实现
│   │   │   ├── topp_flash_kernel.py # 分块/CUDA 后端
│   │   │   └── common.py            # 基础注意力模块
│   │   └── decode_heads/            # 解码头（SegformerHead 等）
│   └── ops/
│       └── topp_flash/              # CUDA 内核源码
├── configs-h/                       # 当前主线配置
└── tools/                           # 训练/推理工具
```

## 快速开始

### 安装
```bash
git clone -b pvsa-v3.0 https://github.com/hyshhh/PVSA-v1.git
cd PVSA-v1
pip install -r requirements/mminstall.txt
pip install -r requirements/runtime.txt
```

### 训练
```bash
# 单卡训练
python tools/train.py configs-h/biformer/biformer_mm-20k_chase_db1-512x512.py
# 多卡训练
bash tools/dist_train.sh configs-h/biformer/biformer_mm-20k_chase_db1-512x512.py ${GPU_NUM}
```

四种注意力后端对应的训练命令如下。

1. `kv_gather` 模式：原始注意力路径，速度较快，但最占显存。
```bash
CUDA_VISIBLE_DEVICES=0 python tools/train.py configs-h/biformer/biformer_mm-20k_chase_db1-512x512.py \
  --cfg-options model.backbone.use_topp_flash=False model.backbone.feature_vis_config.enabled=False model.backbone.attn_vis_config.enabled=False train_dataloader.batch_size=4
```

2. `pruned_kv_gather` 模式：按 keep_len 裁剪无效路由窗口。
```bash
CUDA_VISIBLE_DEVICES=0 python tools/train.py configs-h/biformer/biformer_mm-20k_chase_db1-512x512.py \
  --cfg-options model.backbone.use_topp_flash=False model.backbone.use_pruned_kv_gather=True model.backbone.feature_vis_config.enabled=False model.backbone.attn_vis_config.enabled=False train_dataloader.batch_size=4
```

3. `torch_block` 模式：分块循环计算，显存减少，推理速度稍微变慢。
```bash
CUDA_VISIBLE_DEVICES=0 python tools/train.py configs-h/biformer/biformer_mm-20k_chase_db1-512x512.py \
  --cfg-options model.backbone.use_topp_flash=True model.backbone.topp_flash_backend=torch_block model.backbone.topp_flash_block_windows=16 model.backbone.feature_vis_config.enabled=False model.backbone.attn_vis_config.enabled=False train_dataloader.batch_size=4
```

4. `cuda` 模式：自定义 CUDA 后端，显存最低，但依赖服务器具备可用的 CUDA 编译环境。
```bash
CUDA_VISIBLE_DEVICES=0 python tools/train.py configs-h/biformer/biformer_mm-20k_chase_db1-512x512.py \
  --cfg-options model.backbone.use_topp_flash=True model.backbone.topp_flash_backend=cuda model.backbone.topp_flash_block_windows=16 model.backbone.feature_vis_config.enabled=False model.backbone.attn_vis_config.enabled=False train_dataloader.batch_size=4
```

### 训练配置
| 配置项 | 值 | 说明 |
|--------|----|------|
| 训练轮数 | 200 epochs | 基于 `EpochBasedTrainLoop` |
| 批量大小 | 4 | 单卡训练默认值 |
| 学习率 | 6e-4 | AdamW 优化器 |
| 验证间隔 | 10 epochs | 每 10 个 epoch 验证一次 |
| 检查点保存间隔 | 10 epochs | 每 10 个 epoch 保存一次 |

训练轮数也可以通过命令行覆盖：
```bash
CUDA_VISIBLE_DEVICES=0 python tools/train.py configs-h/biformer/biformer_mm-20k_chase_db1-512x512.py \
  --cfg-options train_cfg.max_epochs=300
```

### 推荐硬件
- **GPU**: NVIDIA RTX 3090 或更高（24GB 显存）
- **显存需求**: 约 8-10GB（使用 `torch_block` 后端）

### 训练环境
| 项目 | 值 |
|------|-----|
| GPU | *待填写* |
| CUDA | *待填写* |
| PyTorch | *待填写* |
| 显存占用 | *待填写* |

注意：`configs-h/_base_/models/VTFormer-s.py` 只是模型片段配置，缺少数据集、训练循环、优化器、运行时作用域等内容，不能直接用于训练。

### 测试方法
使用 `tools/analysis_tools/benchmark.py` 测试推理速度（FPS），该脚本会自动运行 200 次推理并计算平均 FPS。

1. `kv_gather` 模式（原始注意力路径）：
```bash
CUDA_VISIBLE_DEVICES=0 python tools/analysis_tools/benchmark.py configs-h/biformer/biformer_mm-20k_chase_db1-512x512.py /media/ddc/新加卷/hys/hysnew3/PVSA-v1/work_dirs/1/epoch_8.pth --cfg-options model.backbone.use_topp_flash=False model.backbone.feature_vis_config.enabled=False model.backbone.attn_vis_config.enabled=False
```

2. `pruned_kv_gather` 模式（按 keep_len 裁剪无效路由）：
```bash
CUDA_VISIBLE_DEVICES=0 python tools/analysis_tools/benchmark.py configs-h/biformer/biformer_mm-20k_chase_db1-512x512.py /media/ddc/新加卷/hys/hysnew3/PVSA-v1/work_dirs/1/epoch_8.pth --cfg-options model.backbone.use_topp_flash=False model.backbone.use_pruned_kv_gather=True model.backbone.feature_vis_config.enabled=False model.backbone.attn_vis_config.enabled=False
```

3. `torch_block` 模式（分块循环计算，显存和速度均衡）：
```bash
CUDA_VISIBLE_DEVICES=0 python tools/analysis_tools/benchmark.py configs-h/biformer/biformer_mm-20k_chase_db1-512x512.py /media/ddc/新加卷/hys/hysnew3/PVSA-v1/work_dirs/1/epoch_8.pth --cfg-options model.backbone.use_topp_flash=True model.backbone.topp_flash_backend=torch_block model.backbone.topp_flash_block_windows=16 model.backbone.feature_vis_config.enabled=False model.backbone.attn_vis_config.enabled=False
```

4. `cuda` 模式（自定义 CUDA 后端，显存最低）：
```bash
CUDA_VISIBLE_DEVICES=0 python tools/analysis_tools/benchmark.py configs-h/biformer/biformer_mm-20k_chase_db1-512x512.py /media/ddc/新加卷/hys/hysnew3/PVSA-v1/work_dirs/1/epoch_8.pth --cfg-options model.backbone.use_topp_flash=True model.backbone.topp_flash_backend=cuda model.backbone.feature_vis_config.enabled=False model.backbone.attn_vis_config.enabled=False
```

如果需要强制确认 CUDA 后端没有静默回退到 `torch_block`，测试前设置：
```bash
export PVSA_TOPP_FLASH_STRICT_CUDA=1
```

## 配置说明

### 模型配置
```python
backbone=dict(
    type='BiFormer_fusion',
    embed_dim=[64, 128, 256, 512],
    depth=[3, 4, 6, 3],
    topks=[16, 12, 8, 6],           # 四层 Top-P 路由标志位
    topp_route_configs={             # 标志位到真实路由参数的映射
        16: dict(maxk=25, p=0.2, temperature=0.0175, energy=4.0),
        12: dict(maxk=18, p=0.4, temperature=0.025, energy=1.5),
        8: dict(maxk=36, p=0.6, temperature=0.05, energy=0.75),
        6: dict(maxk=49, p=0.8, temperature=0.15, energy=0.4),
    },
    n_win=7,                         # 窗口数量
    use_topp_flash=True,             # 是否启用分块后端
    topp_flash_backend='torch_block', # 'torch_block' 或 'cuda'
    topp_flash_block_windows=16,      # 分块大小
    use_pruned_kv_gather=False,       # 是否在普通 kv_gather 路径裁剪无效路由
    feature_vis_config=dict(enabled=False, save_dir='cam/features_imgs4', out_size=512, channel_reduce='mean'),
    attn_vis_config=dict(enabled=False, save_topk=True, save_heatmap=False, query_index=32, trigger_maxk=25, image_path='/path/to/source.jpg', topk_save_path='cam/attn/attn_stage_topk.png', once=True)
)
```

### topk 参数说明
- `topk > 0`：使用 ToppAttention（Top-P 稀疏注意力）
- `topk == -1`：使用标准全局注意力
- `topk == -2`：使用带局部位置编码的全局注意力（AttentionLePE）
- `topks`：当前四层 Top-P 注意力实际使用的路由标志位，本配置文件显式设置为 `[16, 12, 8, 6]`，与重构前硬编码一致。
- `topp_route_configs`：每个标志位对应真实 `maxk`、累计概率阈值 `p`、温度 `temperature` 和能量补偿 `energy`，现在必须由配置文件提供。
- `feature_vis_config.enabled`：是否保存特征图，打开后会触发处理器同步和磁盘写入，不建议用于正式测速。
- `attn_vis_config.enabled`：是否保存注意力图，`trigger_maxk` 可限制只在指定真实 `maxk` 的层保存。

## 环境变量
| 变量名 | 说明 | 默认值 |
|--------|------|--------|
| `PVSA_TOPP_FLASH_BACKEND` | 强制指定后端 | `torch_block` |
| `PVSA_TOPP_FLASH_STRICT_CUDA` | CUDA 失败时是否报错 | `0` |
| `PVSA_TOPP_FLASH_VERBOSE` | 打印编译日志 | `0` |
| `PVSA_TOPP_FLASH_ARCH` | 目标 GPU 架构 | 自动检测 |

## 性能对比
三种后端的显存占用对比（相对值）：
| 后端 | 显存峰值 | 推理速度 |
|------|---------|---------|
| kv_gather | 100% | 最快（小 topk） |
| torch_block | ~13% | 中等 |
| cuda | ~0% | 最慢 |

## CUDA 后端编译

### 编译方法
项目通过 `torch.utils.cpp_extension.load()` 在第一次使用 `cuda` 后端时自动 JIT 编译 CUDA 扩展，不需要手动写 `setup.py`。

推荐先在服务器上打开编译日志：
```bash
export PVSA_TOPP_FLASH_VERBOSE=1
export PVSA_TOPP_FLASH_STRICT_CUDA=1
```

如果服务器 GPU 架构自动检测失败，可以手动指定：
```bash
export PVSA_TOPP_FLASH_ARCH="8.6"
```

首次训练或测试时会触发编译：
```bash
CUDA_VISIBLE_DEVICES=0 python tools/train.py configs-h/biformer/biformer_mm-20k_chase_db1-512x512.py \
  --cfg-options model.backbone.use_topp_flash=True model.backbone.topp_flash_backend=cuda train_dataloader.batch_size=4
```

### 编译步骤与风险
1. 拉取最新代码
```bash
git pull origin pvsa-v3.0
```
风险：如果服务器本地改过同名文件，`git pull` 可能产生冲突，需要先处理冲突再训练。

2. 确认 CUDA 编译环境
```bash
nvcc --version
python -c "import torch; print(torch.__version__); print(torch.version.cuda); print(torch.cuda.is_available())"
```
风险：如果没有 `nvcc`，或者 PyTorch 的 CUDA 版本和系统编译环境不匹配，CUDA 扩展会编译失败。

3. 首次触发 JIT 编译
```bash
export PVSA_TOPP_FLASH_VERBOSE=1
export PVSA_TOPP_FLASH_STRICT_CUDA=1
CUDA_VISIBLE_DEVICES=0 python tools/train.py configs-h/biformer/biformer_mm-20k_chase_db1-512x512.py \
  --cfg-options model.backbone.use_topp_flash=True model.backbone.topp_flash_backend=cuda train_dataloader.batch_size=4
```
风险：第一次运行会比平时慢，因为需要编译扩展；如果开启 `--amp`，张量可能变成 `float16`，当前 CUDA forward 只支持 `float32`，会回退或报错。

4. 清理旧编译缓存后重编译
如果修改过 `mmseg/ops/topp_flash/topp_flash_cuda.cu`，建议清理 PyTorch 扩展缓存后重新触发编译：
```bash
rm -rf ~/.cache/torch_extensions/py*/pvsa_topp_flash_cuda
```
风险：删除缓存后下一次启动会重新编译；如果路径写错可能误删其他缓存，请只删除 `pvsa_topp_flash_cuda` 对应目录。

### 正确性检查清单
本地没有服务器训练环境，因此不在本地做编译、训练或数值验证。当前代码层面的检查重点如下：
- 只允许 `cuda` 后端源文件和 README 变化，其他模式代码不变
- `topp_flash_forward_kernel` 中 `blockIdx.x` 必须对应 `coarse = batch * p2 + p`
- launch 的 grid 数量必须为 `n * p2`
- `flat_out` 写入布局必须保持 `{n, p2, q_len, dim}`
- `unflatten_windows_kernel` 的输入输出布局保持不变
- C++ 绑定 `topp_flash_forward(...)` 和 Python 侧 `extension.forward(...)` 调用签名保持不变
- 严格模式下应通过 `PVSA_TOPP_FLASH_STRICT_CUDA=1` 暴露编译或 dtype 问题，避免静默回退

## 引用
如果本项目对您的研究有帮助，请考虑引用：
```bibtex
@misc{pvsa2024,
    title={PVSA-Net: Top-P Voting Sparse Attention for Semantic Segmentation},
    author={PVSA-Net Contributors},
    year={2024}
}
```

## 致谢
本项目基于 [MMSegmentation](https://github.com/open-mmlab/mmsegmentation) 构建，感谢 OpenMMLab 团队的优秀工作。

## 许可证
当前精简分支未保留独立许可证文件；如需正式发布或复用，请从备份分支恢复许可证文件或补充新的许可证说明。
