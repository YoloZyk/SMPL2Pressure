# SMPL2Pressure

基于条件变分自编码器 (CVAE) 的人体接触压力图像生成框架。从 SMPL 人体模型的顶点（点云）条件出发，生成对应的接触压力分布图。

## 项目概述

本项目旨在建立 SMPL 人体模型顶点到接触压力图的映射关系。核心模型采用双分支 CVAE 架构：

- **主分支 (Main Branch)**：负责压力图的编码与重建，将压力图编码到潜在空间，再结合条件信息解码还原。
- **条件分支 (Condition Branch)**：基于 PointNet 对 SMPL 顶点进行编码，提取人体几何特征作为条件信息，同时辅助点云重建任务以提升潜在空间质量。

推理时仅需输入 SMPL 顶点，从标准正态分布采样潜在变量即可生成压力图。

## 模型架构

```
                    ┌────────────────────────────────────────────┐
                    │            SMPL2PressureCVAE               │
                    │                                            │
  vertices ───────► │  PointNet Encoder ──► cond_feat            │
  (B, 6890, 3)      │       │                    │               │
                    │       ▼                    │               │
                    │  PointCloud Decoder        │               │
                    │  (recon_vertices)          │               │
                    │                            ▼               │
  pressure_map ───► │  Main Encoder ──► (μ, σ) ──► z             │
  (B, 1, H, W)      │  (ResNet/UNet)     │         │             │
                    │                    ▼         ▼             │
                    │              Reparameterize  │             │
                    │                              ▼             │
                    │                  UNet Decoder ──► pred_pmap│
                    │                  (z + cond_feat)           │
                    └────────────────────────────────────────────┘
```

**训练阶段**：双分支联合训练，损失函数包含压力图重建损失 (PMR)、点云重建损失 (PCR) 和 KL 散度。

**推理阶段**：仅使用条件编码器 + 主解码器，从先验分布 N(0,1) 采样 z，结合点云条件特征生成压力图。

## 支持的数据集

| 数据集 | 名称 | 压力图尺寸 | 最大压力值 | 说明 |
|--------|------|-----------|-----------|------|
| TIP | `tip` | 56 × 40 | 512.0 | In-Bed Pressure 数据集，支持 unseen_group / unseen_subject 模式 |
| PressurePose | `pressurepose` | 64 × 27 | 100.0 | 压力姿态数据集 |
| MoYo | `moyo` | 110 × 37 | 64.0 | MoYo 数据集 |

## 项目结构

```
SMPL2Pressure/
├── assets/                        # SMPL 模型相关静态资源
│   ├── fid_to_part.pkl            # 面片到身体部位的映射
│   ├── smpl_neutral_hd_sample_from_mesh_out.pkl
│   ├── smpl_neutral_hd_vert_regressor_sparse.npz  # HD SMPL 稀疏回归矩阵
│   ├── smpl_part_vid_fid.pkl      # 部位顶点/面片 ID
│   └── smpl_segments_bounds.pkl   # 身体部位边界
│
├── config/                        # 配置文件
│   └── config_base.yaml           # 基础训练配置
│
├── lib/                           # 核心库
│   ├── baseline/                  # 基线方法实现
│   │   ├── __init__.py
│   │   └── pred.py                # PMR / IPMAN 基线预测
│   │
│   ├── dataset/                   # 数据集加载
│   │   ├── __init__.py            # 统一数据集工厂
│   │   ├── base_dataset.py        # 基类
│   │   ├── moyo.py                # MoYo 数据集
│   │   ├── pressurepose.py        # PressurePose 数据集
│   │   └── tip.py                 # TIP (In-Bed Pressure) 数据集
│   │
│   ├── model/                     # 模型组件
│   │   ├── cvae.py                # SMPL2PressureCVAE 主模型
│   │   ├── loss.py                # 损失函数 (PMR + PCR + KL)
│   │   ├── mlp.py                 # MLP 点云解码器
│   │   ├── pointnet.py            # PointNet 条件编码器
│   │   ├── resnet.py              # ResNet 主编码器 (18/34/50)
│   │   └── unet.py                # UNet 编码器 / 解码器
│   │
│   └── utils/                     # 工具函数
│       ├── mesh_utils.py          # 网格工具
│       ├── metrics.py             # 评估指标 (SSIM, MAE, MSE, MRE, CoP)
│       ├── part_volumes.py        # 身体部位体积计算
│       ├── pyrender_visualization.py  # 3D 渲染可视化
│       ├── sampler_util.py        # 采样工具 (beta, transl)
│       ├── stability.py           # 稳定性分析
│       ├── static.py              # 静态路径与数据集元信息
│       ├── trainer.py             # 训练器
│       └── viz_utils.py           # 可视化工具
│
├── diffusion/                     # Diffusion 模型实验 (扩展)
├── wq/                            # 辅助万权任务
├── examples/                      # 一些使用示例
├── check_data/                    # 数据检查工具
│
├── train.py                       # 训练入口
├── test.py                        # 测试入口
├── pretrain.py                    # 条件分支预训练
├── generate.py                    # 压力图生成 (批量推理)
├── train_smpl2pressure.sh         # 训练启动脚本
└── generate.sh                    # 生成启动脚本
```

## 快速开始

### 1. 环境配置

```bash
# 基本依赖
pip install torch torchvision
pip install pyyaml numpy tqdm scikit-image
pip install smplx              # SMPL 人体模型库
pip install pyrender trimesh   # 3D 可视化 (可选)
pip install tensorboard        # 训练日志可视化
```

### 2. 数据准备

需要自行准备对应数据集，并在 `lib/utils/static.py` 中配置数据路径：

```python
# 修改为你实际的数据路径
PressurePose_PATH = "/path/to/pressurepose"
MoYo_PATH = "/path/to/moyo"
TIP_PATH = "/path/to/tip_dataset"

SMPL_MODEL = "/path/to/smpl_models/"
```

### 3. 训练

```bash
# 使用默认配置训练
python train.py --config config/config_base.yaml

# 指定 GPU
python train.py --config config/config_base.yaml --device cuda:0

# 或使用脚本
bash train_smpl2pressure.sh
```

训练输出保存在 `output/{dataset_name}_{timestamp}/` 目录下，包含：
- `ckpts/`：模型权重 (`best_model.pth`, `latest.pth`)
- `logs/`：训练日志与 TensorBoard 文件
- `config.yaml`：训练时使用的完整配置

### 4. 测试

```bash
# 基本测试
python test.py --exp_dir output/{experiment_dir}

# 指定 checkpoint
python test.py --exp_dir output/{experiment_dir} --ckpt best_model.pth

# 与基线方法对比
python test.py --exp_dir output/{experiment_dir} --baseline

# 可视化预测结果
python test.py --exp_dir output/{experiment_dir} --viz
```

评估指标包括：
- **SSIM**：结构相似性
- **MAE / MSE**：绝对/均方误差
- **MRE**：相对误差
- **CoP Distance**：压力中心距离

### 5. 批量生成

```bash
# 随机姿态生成
python generate.py --mode rand --ckpt_dir output/{experiment_dir} --output_dir /path/to/output

# 指定姿态生成
python generate.py --mode sbjt --pose_dir /path/to/poses --ckpt_dir output/{experiment_dir}

# 可视化
python generate.py --mode rand --ckpt_dir output/{experiment_dir} --viz
```

### 6. 条件分支预训练 (可选)

单独预训练 PointNet 条件编码器/解码器，可用于主训练时的初始化：

```bash
python pretrain.py
```

预训练权重保存在 `output/pointnet_pretrain.pth`，在配置文件中启用：

```yaml
model:
  pretrained_cond_path: "output/pointnet_pretrain.pth"
```

## 配置说明

核心配置文件 `config/config_base.yaml`：
> 当前配置为正常默认配置，正常可以直接用于训练。

```yaml
dataset:
  name: "moyo"          # 数据集: tip / pressurepose / moyo
  normal: true           # 是否归一化压力值

model:
  embed_dim: 256         # 主分支潜在维度
  cond_embed_dim: 256    # 条件分支特征维度
  dropout_rate: 0.3

  main_encoder:
    type: "resnet18"     # 主编码器: resnet18 / resnet34 / resnet50 / unet
  main_decoder:
    type: "unet"         # 主解码器: unet
    bilinear: false
  cond_encoder:
    type: "pointnet"     # 条件编码器: pointnet
    use_spatial_transformer: false
    return_global_feature: true
  cond_decoder:
    type: "mlp"          # 条件解码器: mlp

training:
  batch_size: 64
  num_epochs: 100
  learning_rate: 0.001
  loss:
    pmr_weight: 100.0    # 压力图重建损失权重
    kl_weight: 10.0      # KL 散度权重
    pcr_weight: 60.0     # 点云重建损失权重
  optimizer:
    type: "adam"
    weight_decay: 0.0001
  scheduler:
    type: "cosine"
```

## 基线方法

项目内置了两个经典基线方法用于对比：

- **PMR (Pressure Map Reconstruction)**：基于深度投影的启发式方法，将 SMPL 顶点投影到传感器网格生成压力图。
- **IPMAN**：基于 HD SMPL 稀疏回归 + 高度加权的压力映射方法。

在测试时使用 `--baseline` 参数可同时评估基线方法。
