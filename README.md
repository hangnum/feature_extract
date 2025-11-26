# 多模态医疗图像特征提取与融合项目

基于深度学习的多模态医疗图像分类、特征提取与智能融合系统，支持ResNet、Swin Transformer等预训练模型，集成CMTA跨模态 transformer和ELM极限学习机等先进融合算法。

## 项目概述

本项目用于处理医疗影像数据（224×224灰度图），主要功能包括：

### 核心功能
- **数据预处理**：解析原始数据，按医院和模态划分数据集，支持分层抽样
- **单模态训练**：使用预训练模型（ResNet、Swin Transformer）进行迁移学习和微调
- **特征提取**：提取病人级特征用于后续分析
- **多模态融合**：集成多种先进融合算法
  - CMTA (Cross-Modal Transformer with Alignment)：跨模态Transformer融合
  - ELM (Extreme Learning Machine)：极限学习机特征聚合
  - Sequence Fusion：时序多模态融合
- **实验管理**：完整的日志记录、模型管理和可视化分析

### 技术特色
- **知识分解**：CMTA模型支持跨模态知识分解与重组
- **原型学习**：基于原型库的知识记忆与检索机制
- **特征选择**：ELM集成U-test特征重要性筛选
- **端到端训练**：支持多损失函数联合优化

## 最近更新

- **2024-11** (重大版本更新):
  - **CMTA多模态融合**：集成跨模态Transformer，支持知识分解和原型学习
  - **ELM特征聚合**：实现极限学习机+U-test特征选择流水线
  - **Sequence Fusion**：支持时序多模态数据融合分析
  - **可视化增强**：新增训练曲线绘制和结果分析工具
  - **CLI扩展**：管理脚本支持CMTA、ELM、可视化等新功能

- **2023-11**:
  - 新增 `manage.py` 统一管理脚本，简化操作流程
  - 实现基础特征融合，采用均值池化后拼接策略
  - 优化数据划分逻辑，JM医院数据默认采用分层抽样
  - 增加关闭早停功能的选项

## 环境配置

### 1. 安装依赖

```bash
cd d:\code\feature_extract
pip install -r requirements.txt
```

### 主要依赖

#### 核心框架
- PyTorch >= 2.0.0
- torchvision >= 0.15.0
- timm >= 0.9.0

#### 数据处理
- scikit-learn >= 1.3.0
- pandas >= 2.0.0
- numpy >= 1.24.0
- scipy >= 1.10.0

#### CMTA/ELM专用依赖
- einops >= 0.7.0          # 张量操作库
- numba >= 0.58.0          # 高性能数值计算
- optuna >= 3.0.0          # 超参数优化（ELM）

#### 可视化和工具
- matplotlib >= 3.7.0
- seaborn >= 0.12.0
- tensorboard >= 2.10.0
- PyYAML >= 6.0
- tqdm >= 4.64.0

## 项目结构

```bash
feature_extract/
├── config/                      # 配置文件
│   ├── default_config.yaml      # 默认配置
│   ├── elm_config.example.json  # ELM配置模板
│   └── best_hparams/            # 最佳超参数记录
│       ├── resnet18_A.yaml
│       ├── resnet50_P.yaml
│       └── cmta.yaml           # CMTA最佳配置
├── data/
│   ├── splits/                  # 数据划分CSV文件
│   │   ├── train_{modality}.csv
│   │   ├── val_{modality}.csv
│   │   └── test_{modality}.csv
│   └── features/                # 提取的特征
│       ├── train/
│       ├── val/
│       └── test/
├── src/                         # 源代码
│   ├── data/                    # 数据处理模块
│   │   ├── data_parser.py       # 数据解析
│   │   ├── data_splitter.py     # 数据划分
│   │   ├── dataset.py           # 数据集类
│   │   ├── cmta_dataset.py      # CMTA专用数据集
│   │   └── transforms.py        # 数据增强
│   ├── models/                  # 模型模块
│   │   ├── model_loader.py      # 模型加载
│   │   ├── cmta.py              # CMTA融合模型
│   │   ├── cmta_utils.py        # CMTA工具函数
│   │   ├── knowledge_decomposition.py  # 知识分解
│   │   ├── pib.py               # PIB信息瓶颈
│   │   ├── fusion_utils.py      # 融合工具
│   │   └── losses.py            # 损失函数
│   ├── training/                # 训练模块
│   │   ├── trainer.py           # 训练器
│   │   ├── cmta_trainer.py      # CMTA训练器
│   │   └── metrics.py           # 评估指标
│   ├── feature_extraction/      # 特征提取模块
│   │   └── extractor.py         # 特征提取器
│   ├── elm/                     # ELM极限学习机模块
│   │   ├── pipeline.py          # ELM流水线
│   │   ├── config.py            # ELM配置
│   │   └── cli.py               # ELM命令行接口
│   └── utils/                   # 工具函数
│       ├── config.py            # 配置管理
│       ├── logger.py            # 日志工具
│       ├── metrics.py           # 通用指标计算
│       ├── kmeans.py            # K-means聚类
│       ├── plotting.py          # 绘图工具
│       └── seed.py              # 随机种子
├── scripts/                     # 运行脚本
│   ├── manage.py                # 统一管理脚本（推荐）
│   ├── preprocess_data.py       # 数据预处理
│   ├── train.py                 # 训练脚本
│   ├── extract_features.py      # 特征提取脚本
│   ├── train_cmta.py            # CMTA训练脚本
│   ├── run_elm.py               # ELM运行脚本
│   └── visualize_results.py     # 结果可视化
├── elm/                         # ELM根模块
│   ├── pipeline.py              # ELM特征聚合流水线
│   ├── config.py                # ELM配置管理
│   └── main.py                  # ELM主程序
└── outputs/                     # 输出目录
    └── feature_extract/
        ├── checkpoints/         # 模型检查点
        ├── logs/                # 训练日志
        └── visualizations/      # 可视化结果
```

## 使用说明

### 方式一：使用统一管理脚本（推荐）

项目提供了 `manage.py` 脚本，可以统一管理所有流程：

#### 1. 数据预处理

```bash
python scripts/manage.py preprocess \
    --config config/default_config.yaml \
    --root_dir /path/to/data \
    --modalities A P \
    --output_dir data/splits \
    --train_ratio 0.7 \
    --seed 42
```

**参数说明**：
- `--root_dir`: 数据根目录（覆盖配置文件）
- `--modalities`: 模态列表（如 A P）
- `--output_dir`: 划分文件输出目录
- `--train_ratio`: JM医院训练集比例（默认0.7）
- `--seed`: 随机种子确保可复现
- `--log_dir`: 预处理日志目录

#### 2. 训练模型

```bash
# 训练A模态（ResNet18）
python scripts/manage.py train \
    --modality A \
    --model resnet18 \
    --config config/default_config.yaml \
    --epochs 100 \
    --batch_size 32 \
    --learning_rate 0.001 \
    --loss_type focal \
    --device cuda:0

# 训练P模态（ResNet50）
python scripts/manage.py train \
    --modality P \
    --model resnet50 \
    --config config/default_config.yaml \
    --disable_early_stop \
    --resume
```

**参数说明**：
- `--modality`: 模态名称（A 或 P，必需）
- `--model`: 模型名称（覆盖配置文件）
- `--epochs`: 训练轮数
- `--batch_size`: 批大小
- `--learning_rate`: 学习率
- `--loss_type`: 损失函数（ce, focal, asymmetric）
- `--device`: 训练设备（cuda:0, cpu等）
- `--disable_early_stop`: 关闭早停
- `--resume`: 从检查点恢复训练

#### 3. 特征提取

```bash
python scripts/manage.py extract \
    --modality A \
    --model resnet18 \
    --checkpoint outputs/feature_extract/checkpoints/best_model.pth \
    --output_dir data/features \
    --batch_size 64 \
    --device cuda:0 \
    --align
```

**参数说明**：
- `--modality`: 要提取的模态（A 或 P，必需）
- `--model`: 特征提取器模型名称
- `--checkpoint`: 模型检查点路径（默认best_model.pth）
- `--output_dir`: 特征输出目录
- `--batch_size`: 提取批大小
- `--device`: 提取设备
- `--align`: 提取后对齐多模态特征

#### 4. CMTA多模态融合训练

```bash
# 使用默认CMTA配置训练
python scripts/manage.py cmta \
    --data_dir /path/to/data \
    --modalities A P \
    --model_size small \
    --epochs 100 \
    --batch_size 32 \
    --learning_rate 0.001 \
    --device cuda:0

# 使用自定义配置
python scripts/manage.py cmta \
    --config config/best_hparams/cmta.yaml \
    --data_dir /path/to/data \
    --modalities A P \
    --alpha 0.5 \
    --beta 0.1 \
    --resume outputs/cmta/checkpoints/best_model.pth
```

**CMTA核心参数**：
- `--model_size`: 模型规模 (small, large)
- `--alpha`: 队列损失权重 (默认0.5)
- `--beta`: 辅助损失权重 (默认0.1)
- `--feat_dim`: 特征维度 (默认1024)
- `--num_cluster`: 聚类数量 (默认64)
- `--bank_length`: 原型库长度 (默认16)

#### 5. ELM特征聚合与优化

```bash
# 运行完整的ELM流水线
python scripts/manage.py elm \
    --data_type CT \
    --output outputs/elm \
    --n_trials 100 \
    --auc_floor 0.7 \
    --max_gap 0.2

# 使用自定义ELM配置
python scripts/manage.py elm \
    --data_type BL \
    --elm_config config/elm_config.json \
    --hidden_min 50 \
    --hidden_max 500 \
    --random_state 42
```

**ELM核心参数**：
- `--data_type`: 数据类型标识 (CT, BL等)
- `--n_trials`: Optuna优化试验次数
- `--hidden_min/max`: 隐藏层节点数范围
- `--auc_floor`: 最小AUC阈值
- `--alpha_train/test`: U-test p值阈值

#### 6. 结果可视化

```bash
# 绘制训练曲线
python scripts/manage.py visualize \
    --history_csv outputs/feature_extract/logs/exp_name/training_history.csv \
    --output_dir outputs/feature_extract/visualizations

# TensorBoard实时监控
tensorboard --logdir outputs/feature_extract/logs
```

### 方式二：使用独立脚本

#### 步骤1: 数据预处理

解析原始数据并生成训练/验证/测试集划分：

```bash
python scripts/preprocess_data.py \
    --root_dir "D:\data\raw\Grade" \
    --modalities A P \
    --output_dir "d:\code\feature_extract\data\splits" \
    --train_ratio 0.7
```

**输出**: 在 `data/splits/` 目录下生成6个CSV文件：

- `train_A.csv`, `val_A.csv`, `test_A.csv`
- `train_P.csv`, `val_P.csv`, `test_P.csv`

**数据划分策略**：

- 只处理同时拥有A和P两种模态的病人
- JM医院：按病人ID进行7:3分层抽样（训练集:验证集）
- 其他医院：全部作为测试集（外验）
- 同一病人的所有切片必须在同一集合中，避免数据泄漏

#### 步骤2: 训练模型

为每个模态训练单独的模型：

##### 训练A模态（ResNet18）

```bash
python scripts/train.py \
    --modality A \
    --model resnet18 \
    --epochs 100 \
    --batch_size 64 \
    --learning_rate 1e-4 \
    --loss_type ce \
    --device cuda
```

##### 训练P模态（ResNet50，冻结前3层）

首先修改 `config/default_config.yaml` 中的 `freeze_stages: 3`，然后运行：

```bash
python scripts/train.py \
    --modality P \
    --model resnet50 \
    --epochs 100 \
    --batch_size 64 \
    --learning_rate 1e-4 \
    --loss_type focal
```

**关键参数**:

- `--modality`: 模态名称 (A 或 P)
- `--model`: 模型名称 (resnet18, resnet50, swin_t)
- `--loss_type`: 损失函数 (ce, focal, asymmetric)
- `--optimizer`: 优化器 (adam, sgd, adamw)
- `--scheduler`: 学习率调度器 (cosine, step, plateau)
- `--resume`: 从检查点恢复训练
- `--disable_early_stop`: 关闭早停功能

**输出**:

- 模型检查点: `outputs/feature_extract/checkpoints/best_model.pth`
- 训练日志: `outputs/feature_extract/logs/`
- 最佳超参数: `config/best_hparams/{model}_{modality}.yaml`
- 训练历史: `outputs/feature_extract/logs/{exp_name}/training_history.csv`
- 外验结果: `outputs/feature_extract/logs/{exp_name}/test_metrics.csv`

#### 步骤3: 特征提取

从训练好的模型中提取病人级特征：

```bash
python scripts/extract_features.py \
    --modality A \
    --model resnet18 \
    --checkpoint "D:\outputs\feature_extract\checkpoints\best_model.pth" \
    --output_dir "d:\code\feature_extract\data\features" \
    --batch_size 32
```

对两个模态分别提取特征，然后使用融合脚本对齐：

```bash
# 先提取A模态特征
python scripts/extract_features.py --modality A --model resnet18 ...

# 再提取P模态特征
python scripts/extract_features.py --modality P --model resnet50 ...

# 最后对齐多模态特征
python scripts/fuse_features.py \
    --feature_dir "d:\code\feature_extract\data\features" \
    --modalities A P \
    --output_dir "d:\code\feature_extract\data\features"
```

**输出**:

- 特征文件: `data/features/{split}/{modality}/grade{label}/{patient_id}.npy`
- 特征信息: `data/features/{split}/features_{modality}.csv`
- 对齐后的特征: `data/features/{split}/aligned/`

#### 步骤4: 查看训练结果

使用TensorBoard查看训练曲线：

```bash
tensorboard --logdir outputs/feature_extract/logs
```

在浏览器中打开 `http://localhost:6006`

## CMTA与ELM详解

### CMTA (Cross-Modal Transformer with Alignment)

#### 核心思想
CMTA是一种跨模态Transformer融合模型，通过知识分解和原型学习实现多模态医学图像的智能融合。

#### 技术架构
1. **知识分解模块** (`src/models/knowledge_decomposition.py`)
   - 将单模态特征分解为模态共享知识和模态特有知识
   - 支持跨模态知识的重组与重构

2. **原型学习机制** (`src/models/cmta_utils.py`)
   - 维护可学习的原型库 (Prototype Bank)
   - 支持动态原型更新和检索
   - 实现知识的长期记忆与泛化

3. **多损失函数优化**
   - **队列损失 (Cohort Loss)**: `alpha`权重，增强同类样本聚集
   - **辅助损失 (Auxiliary Loss)**: `beta`权重，促进知识分解
   - **分类损失**: 标准交叉熵损失

#### 关键参数
```yaml
model:
  cmta:
    feat_dim: 1024        # 特征维度
    num_cluster: 64       # 原型聚类数量
    bank_length: 16       # 原型库长度
    update_ratio: 0.1     # 原型更新率
    model_size: small     # 模型规模 (small/large)

training:
  cmta:
    alpha: 0.5            # 队列损失权重
    beta: 0.1             # 辅助损失权重
    seed: 1               # 随机种子
    update_rat: 0.1       # 知识记忆更新率
```

#### 使用场景
- 多模态医学图像融合诊断
- 跨模态知识迁移学习
- 小样本多模态分类任务

### ELM (Extreme Learning Machine)

#### 核心思想
ELM极限学习机结合U-test特征选择，实现高效的多模态特征聚合与优化。

#### 技术流程
1. **特征聚合** (`elm/pipeline.py`)
   - 多模态特征的均值池化和拼接
   - 支持不同模态特征维度的自动对齐

2. **U-test特征选择**
   - 基于Mann-Whitney U检验的特征重要性评估
   - 自动筛选统计显著性高的特征
   - 可配置p值阈值 (`alpha_train`, `alpha_test`)

3. **超参数优化**
   - 使用Optuna进行自动超参数搜索
   - 优化隐藏层节点数、正则化参数等
   - 支持多目标优化 (AUC最大化、过拟合控制)

#### 配置文件
```json
{
  "data_types": ["CT", "BL"],
  "feature_dirs": {
    "train": "data/features/train",
    "val": "data/features/val",
    "test": "data/features/test"
  },
  "elm_params": {
    "hidden_min": 50,
    "hidden_max": 1000,
    "activation": "relu",
    "alpha": 1.0
  },
  "selection": {
    "alpha_train": 0.05,
    "alpha_test": 0.05
  },
  "optimization": {
    "n_trials": 100,
    "auc_floor": 0.7,
    "max_gap": 0.2
  }
}
```

#### 使用场景
- 快速特征聚合与基线模型建立
- 大规模特征集合的高效筛选
- 多模态特征的统计显著性分析

### 时序融合 (Sequence Fusion)

#### 核心功能
- 支持时序多模态数据的融合分析
- GPU加速的K-means聚类算法
- 动态时间规整 (DTW) 距离计算

#### 技术特点
- 高效的GPU并行计算
- 支持长时间序列的批处理
- 集成多种时序相似性度量

## 配置说明

配置文件 `config/default_config.yaml` 包含所有参数设置：

### 数据配置

```yaml
data:
  root_dir: "D:\\data\\raw\\Grade"
  modalities: [A, P]
  train_ratio: 0.7
  image_size: 224
  num_workers: 8
```

### 模型配置

```yaml
model:
  name: resnet18           # resnet18, resnet50, swin_t
  pretrained: true
  num_classes: 2
  freeze_stages: 0         # ResNet50时设为3
```

### 训练配置

```yaml
training:
  batch_size: 64
  epochs: 100
  learning_rate: 0.0001
  weight_decay: 0.0001
  loss_type: ce            # ce, focal, asymmetric
  optimizer: adamw         # adam, sgd, adamw
  scheduler: cosine        # cosine, step, plateau
  early_stop_patience: 10
  early_stop_enabled: true
  device: cuda
```

### 数据增强配置

```yaml
augmentation:
  horizontal_flip: true
  rotation_degrees: 15
  normalize_mean: 0.5
  normalize_std: 0.5
```

### 实验配置

```yaml
experiment:
  name: ""                 # 实验名称（默认自动生成）
  seed: 42                 # 随机种子
  output_dir: "D:\\outputs\\feature_extract"
  save_best_only: true
  log_interval: 10
```

**命令行覆盖配置**：命令行参数优先级高于配置文件，可以灵活调整参数而无需修改配置文件。

## 数据格式

### 原始数据结构

```bash
D:\data\raw\Grade\
├── JM/                          # JM医院数据
│   ├── fold1/                   # 折1（忽略fold，统一处理）
│   │   ├── grade0/              # 标签0
│   │   │   ├── {patient_id}/    # 病人ID
│   │   │   │   ├── A/           # A模态
│   │   │   │   │   └── {patient_id}_a_slice_001.png
│   │   │   │   └── P/           # P模态
│   │   │   │       └── {patient_id}_p_slice_001.png
│   │   └── grade1/              # 标签1
│   ├── fold2/
│   ├── fold3/
│   ├── fold4/
│   └── fold5/
└── OtherHospital/               # 其他医院（外验）
    ├── grade0/
    └── grade1/
```

### 数据说明

- **图片格式**: 224×224×1的灰度图PNG文件
- **标签**: grade0和grade1（映射为0和1）
- **模态**: 每个病人包含多个模态（A, P, T1等），本项目仅使用A和P
- **切片**: 每个病人每个模态包含多张切片图像
- **病人ID唯一性**: patient_id作为唯一标识

### 数据划分CSV格式

生成的CSV文件包含两列：

```csv
image_path,label
D:\data\raw\Grade\JM\fold1\grade0\202009344\A\202009344_a_slice_001.png,0
D:\data\raw\Grade\JM\fold1\grade1\202027938\A\202027938_a_slice_005.png,1
...
```

### 特征格式

每个病人的特征保存为 `.npy` 文件，形状为 `(n, m)`：

- `n`: 该病人的切片数量
- `m`: 特征维度（ResNet18: 512, ResNet50: 2048, Swin-T: 768）

特征信息CSV包含：

```csv
patient_id,feature_path,label,num_slices
202009344,data/features/train/A/grade0/202009344.npy,0,15
...
```

## 常见问题

### Q1: 如何修改模型架构？

编辑 `src/models/model_loader.py` 中的 `load_model` 函数，添加新的模型支持。例如添加EfficientNet：

```python
elif model_name.startswith('efficientnet'):
    model = timm.create_model(model_name, pretrained=pretrained, num_classes=num_classes, in_chans=1)
```

### Q2: 如何处理类别不平衡？

项目支持多种损失函数处理类别不平衡：

1. **Focal Loss**：自动降低易分类样本的权重

   ```bash
   python scripts/train.py --loss_type focal ...
   ```

2. **Asymmetric Loss**：针对正负样本使用不同的损失权重

   ```bash
   python scripts/train.py --loss_type asymmetric ...
   ```

### Q3: 如何调整数据增强？

方法1：修改 `config/default_config.yaml` 中的 `augmentation` 部分

方法2：在训练时通过命令行指定（需要修改train.py支持）

### Q4: 训练中断后如何恢复？

使用 `--resume` 参数从最新的检查点恢复：

```bash
python scripts/train.py --modality A --model resnet18 --resume
```

训练器会自动加载 `best_model.pth` 或 `last_checkpoint.pth`

### Q5: 如何关闭早停？

使用 `--disable_early_stop` 参数：

```bash
python scripts/train.py --disable_early_stop ...
```

### Q6: 最佳超参数如何使用？

训练完成后，最佳超参数会自动保存在 `config/best_hparams/{model}_{modality}.yaml`。下次训练时可以直接使用：

```bash
python scripts/train.py --config config/best_hparams/resnet18_A.yaml --modality A
```

### Q7: 如何处理显存不足？

1. 减小batch_size：`--batch_size 16`
2. 使用梯度累积（需要修改trainer.py）
3. 使用混合精度训练（需要修改trainer.py）
4. 选择更小的模型：`--model resnet18`

### Q8: 如何确保实验可复现？

项目在多个层面保证可复现性：

1. **随机种子**: 在配置文件中设置 `seed: 42`，代码会自动设置Python、NumPy、PyTorch的随机种子
2. **配置保存**: 每次训练会自动保存完整配置到日志目录
3. **最佳超参记录**: 自动记录并保存最佳验证结果的超参数

## 实验记录

所有实验自动记录以下信息：

### 训练过程

- 训练和验证曲线（Loss、AUC、Accuracy、Sensitivity、Specificity）
- 每个epoch的详细指标
- TensorBoard可视化日志

### 模型保存

- 最佳模型检查点（基于验证集AUC）
- 完整的模型状态、优化器状态
- 训练配置和超参数

### 评估结果

- 验证集最佳性能指标
- 外验集（测试集）评估结果
- 自动更新最佳超参数记录

## 典型工作流程

### 基础流程：单模态训练与特征提取

```bash
# 1. 数据预处理
python scripts/manage.py preprocess \
    --config config/default_config.yaml \
    --root_dir /path/to/medical/data \
    --modalities A P \
    --train_ratio 0.7 \
    --seed 42

# 2. 训练A模态（ResNet18）
python scripts/manage.py train \
    --modality A \
    --model resnet18 \
    --config config/default_config.yaml \
    --epochs 100 \
    --batch_size 32 \
    --learning_rate 0.001 \
    --loss_type focal \
    --device cuda:0

# 3. 训练P模态（ResNet50）
python scripts/manage.py train \
    --modality P \
    --model resnet50 \
    --config config/default_config.yaml

# 4. 特征提取与对齐
python scripts/manage.py extract \
    --modality A \
    --model resnet18 \
    --checkpoint outputs/feature_extract/checkpoints/best_model.pth \
    --output_dir data/features \
    --batch_size 64 \
    --device cuda:0 \
    --align

python scripts/manage.py extract \
    --modality P \
    --model resnet50 \
    --checkpoint outputs/feature_extract/checkpoints/best_model.pth \
    --output_dir data/features
```

### 高级流程：CMTA多模态融合

```bash
# 1-2. 基础训练与特征提取（同上）

# 3. CMTA多模态融合训练
python scripts/manage.py cmta \
    --data_dir /path/to/data \
    --modalities A P \
    --model_size small \
    --epochs 100 \
    --batch_size 32 \
    --learning_rate 0.001 \
    --alpha 0.5 \
    --beta 0.1 \
    --device cuda:0

# 4. CMTA模型评估与结果分析
python scripts/manage.py visualize \
    --history_csv outputs/cmta/logs/training_history.csv \
    --output_dir outputs/cmta/visualizations
```

### 完整流程：ELM特征聚合优化

```bash
# 1-2. 基础训练与特征提取（同上）

# 3. ELM特征聚合与超参数优化
python scripts/manage.py elm \
    --data_type CT \
    --output outputs/elm \
    --n_trials 100 \
    --auc_floor 0.7 \
    --alpha_train 0.05 \
    --alpha_test 0.05

# 4. 最优ELM模型评估
# 结果自动保存在 outputs/elm/final_results.mat
```

### 研究流程：全算法对比

```bash
# 1. 数据预处理和基础训练（统一）
python scripts/manage.py preprocess --config config/default_config.yaml ...
python scripts/manage.py train --modality A --model resnet18 ...
python scripts/manage.py train --modality P --model resnet50 ...
python scripts/manage.py extract --modality A --align
python scripts/manage.py extract --modality P

# 2. CMTA融合
python scripts/manage.py cmta --model_size small --epochs 100

# 3. ELM聚合
python scripts/manage.py elm --data_type CT --n_trials 200

# 4. 结果对比与可视化
python scripts/manage.py visualize --history_csv outputs/*/training_history.csv
tensorboard --logdir outputs/
```

## 下一步

完成特征提取后，可以：

1. **CMTA多模态融合**：使用跨模态Transformer进行端到端融合训练
2. **ELM特征聚合**：通过极限学习机实现高效特征聚合与优化
3. **算法对比研究**：综合评估不同融合策略的性能表现
4. **可视化分析**：使用t-SNE、UMAP等工具进行特征降维可视化
5. **临床部署**：将优化后的模型集成到临床诊断系统中

## 技术特性

### 基础能力
- ✅ 支持多种预训练模型（ResNet系列、Swin Transformer）
- ✅ 灵活的损失函数选择（CE、Focal、Asymmetric、Cohort）
- ✅ 完整的实验管理和日志记录
- ✅ 自动保存最佳模型和超参数
- ✅ 支持断点续训和早停机制
- ✅ 病人级特征提取和对齐
- ✅ 数据泄漏防护（病人级划分）
- ✅ 随机种子控制保证可复现
- ✅ TensorBoard可视化支持

### 高级功能
- 🚀 **CMTA融合**：跨模态Transformer与知识分解
- 🚀 **ELM优化**：极限学习机+U-test特征选择
- 🚀 **Sequence Fusion**：时序多模态数据融合
- 🚀 **原型学习**：可学习原型库与知识记忆
- 🚀 **GPU加速**：高性能并行计算支持
- 🚀 **自动调参**：Optuna超参数优化
- 🚀 **统计分析**：严格的统计显著性检验

## 许可证

本项目仅供学术研究使用。

## 联系方式

如有问题，请联系项目维护者。
