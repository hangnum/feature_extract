# 多模态医疗图像特征提取项目

基于深度学习的多模态医疗图像分类和特征提取系统，支持ResNet和Swin Transformer等预训练模型。

## 项目概述

本项目用于处理医疗影像数据（224×224灰度图），主要功能包括：

- **数据预处理**：解析原始数据，按医院和模态划分数据集
- **模型训练**：使用预训练模型进行迁移学习和微调
- **特征提取**：提取病人级特征用于后续分析
- **特征融合**：对齐多模态特征，支持后续融合分析（均值池化+拼接）
- **实验管理**：完整的日志记录和模型管理

## 最近更新

- **2023-11**:
  - 新增 `manage.py` 统一管理脚本，简化操作流程。
  - 实现多模态特征融合（Feature Fusion），采用均值池化后拼接的策略。
  - 优化数据划分逻辑，JM医院数据默认采用分层抽样（Stratified Sampling）。
  - 增加关闭早停（Early Stopping）的选项。

## 环境配置

### 1. 安装依赖

```bash
cd d:\code\feature_extract
pip install -r requirements.txt
```

### 主要依赖

- PyTorch >= 2.0.0
- torchvision >= 0.15.0
- timm >= 0.9.0
- scikit-learn >= 1.3.0
- pandas >= 2.0.0
- PyYAML >= 6.0

## 项目结构

```bash
feature_extract/
├── config/                      # 配置文件
│   ├── default_config.yaml      # 默认配置
│   └── best_hparams/            # 最佳超参数记录
│       ├── resnet18_A.yaml
│       └── resnet50_P.yaml
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
│   │   └── transforms.py        # 数据增强
│   ├── models/                  # 模型模块
│   │   ├── model_loader.py      # 模型加载
│   │   └── losses.py            # 损失函数
│   ├── training/                # 训练模块
│   │   ├── trainer.py           # 训练器
│   │   └── metrics.py           # 评估指标
│   ├── feature_extraction/      # 特征提取模块
│   │   ├── extractor.py         # 特征提取器
│   │   └── feature_aligner.py   # 特征对齐
│   └── utils/                   # 工具函数
│       ├── config.py            # 配置管理
│       ├── logger.py            # 日志工具
│       └── seed.py              # 随机种子
├── scripts/                     # 运行脚本
│   ├── manage.py                # 统一管理脚本（推荐）
│   ├── preprocess_data.py       # 数据预处理
│   ├── train.py                 # 训练脚本
│   ├── extract_features.py      # 特征提取脚本
│   ├── fuse_features.py         # 特征融合脚本
│   └── visualize_results.py     # 结果可视化
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

#### 4. 特征融合

```bash
python scripts/manage.py fuse \
    --modalities A P \
    --feature_dir data/features \
    --output_dir outputs/feature_extract/fusion \
    --use_aligned \
    --C 1.0 \
    --max_iter 1000 \
    --random_state 42
```

**参数说明**：
- `--feature_dir`: 特征文件目录
- `--modalities`: 要融合的模态列表
- `--use_aligned`: 使用对齐后的特征CSV
- `--C`: L2正则化强度的倒数
- `--max_iter`: 逻辑回归最大迭代次数
- `--random_state`: 随机种子
- `--output_dir`: 融合结果输出目录

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

### 完整流程示例

```bash
# 1. 数据预处理（带详细参数）
python scripts/manage.py preprocess \
    --config config/default_config.yaml \
    --root_dir /path/to/medical/data \
    --modalities A P \
    --train_ratio 0.7 \
    --seed 42

# 2. 训练A模态（ResNet18，自定义参数）
python scripts/manage.py train \
    --modality A \
    --model resnet18 \
    --config config/default_config.yaml \
    --epochs 100 \
    --batch_size 32 \
    --learning_rate 0.001 \
    --loss_type focal \
    --device cuda:0

# 3. 训练P模态（ResNet50，使用默认配置）
python scripts/manage.py train \
    --modality P \
    --model resnet50 \
    --config config/default_config.yaml

# 4. 提取A模态特征（带对齐）
python scripts/manage.py extract \
    --modality A \
    --model resnet18 \
    --checkpoint outputs/feature_extract/checkpoints/best_model.pth \
    --output_dir data/features \
    --batch_size 64 \
    --device cuda:0 \
    --align

# 5. 提取P模态特征
python scripts/manage.py extract \
    --modality P \
    --model resnet50 \
    --checkpoint outputs/feature_extract/checkpoints/best_model.pth \
    --output_dir data/features

# 6. 融合多模态特征（L2正则化分类器）
python scripts/manage.py fuse \
    --modalities A P \
    --feature_dir data/features \
    --output_dir outputs/feature_extract/fusion \
    --use_aligned \
    --C 1.0 \
    --max_iter 1000 \
    --random_state 42

# 7. 查看训练结果
tensorboard --logdir outputs/feature_extract/logs
```

## 下一步

完成特征提取后，可以：

1. 使用提取的特征训练多模态融合模型
2. 使用L2正则化的concat策略融合A和P模态
3. 进行进一步的分类或预测任务
4. 使用提取的特征进行可视化分析（t-SNE、UMAP等）

## 技术特性

- ✅ 支持多种预训练模型（ResNet系列、Swin Transformer）
- ✅ 灵活的损失函数选择（CE、Focal、Asymmetric）
- ✅ 完整的实验管理和日志记录
- ✅ 自动保存最佳模型和超参数
- ✅ 支持断点续训
- ✅ 早停机制防止过拟合
- ✅ 病人级特征提取和对齐
- ✅ 数据泄漏防护（病人级划分）
- ✅ 随机种子控制保证可复现
- ✅ TensorBoard可视化支持

## 许可证

本项目仅供学术研究使用。

## 联系方式

如有问题，请联系项目维护者。
