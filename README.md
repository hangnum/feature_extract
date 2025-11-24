# 多模态医疗图像特征提取项目

基于深度学习的多模态医疗图像分类和特征提取系统，支持ResNet和Swin Transformer等预训练模型。

## 项目概述

本项目用于处理医疗影像数据（224×224灰度图），主要功能包括：

- **数据预处理**：解析原始数据，按医院和模态划分数据集
- **模型训练**：使用预训练模型进行迁移学习和微调
- **特征提取**：提取病人级特征用于后续分析
- **实验管理**：完整的日志记录和模型管理

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

## 项目结构

``` bash
feature_extract/
├── config/                      # 配置文件
│   ├── default_config.yaml      # 默认配置
│   └── best_hparams/            # 最佳超参数记录
├── data/
│   ├── splits/                  # 数据划分CSV文件
│   └── features/                # 提取的特征
├── src/                         # 源代码
│   ├── data/                    # 数据处理模块
│   ├── models/                  # 模型模块
│   ├── training/                # 训练模块
│   ├── feature_extraction/      # 特征提取模块
│   └── utils/                   # 工具函数
├── scripts/                     # 运行脚本
│   ├── preprocess_data.py       # 数据预处理
│   ├── train.py                 # 训练脚本
│   └── extract_features.py      # 特征提取脚本
└── outputs/                     # 输出目录
    └── feature_extract/
        ├── checkpoints/         # 模型检查点
        ├── logs/                # 训练日志
        └── visualizations/      # 可视化结果
```

## 使用说明

### 步骤1: 数据预处理

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

### 步骤2: 训练模型

为每个模态训练单独的模型：

#### 训练A模态（ResNet18）

```bash
python scripts/train.py \
    --modality A \
    --model resnet18 \
    --epochs 100 \
    --batch_size 32 \
    --learning_rate 1e-4 \
    --loss_type ce
```

#### 训练P模态（ResNet50，冻结前3层）

```bash
python scripts/train.py \
    --modality P \
    --model resnet50 \
    --epochs 100 \
    --batch_size 32 \
    --learning_rate 1e-4 \
    --loss_type focal
```

**关键参数**:

- `--modality`: 模态名称 (A 或 P)
- `--model`: 模型名称 (resnet18, resnet50, swin_t)
- `--loss_type`: 损失函数 (ce, focal, asymmetric)
- `--resume`: 从检查点恢复训练

**输出**:

- 模型检查点: `outputs/feature_extract/checkpoints/best_model.pth`
- 训练日志: `outputs/feature_extract/logs/`
- 最佳超参数: `config/best_hparams/{model}_{modality}.yaml`

### 步骤3: 特征提取

从训练好的模型中提取病人级特征：

```bash
python scripts/extract_features.py \
    --modality A \
    --model resnet18 \
    --checkpoint "D:\outputs\feature_extract\checkpoints\best_model.pth" \
    --output_dir "d:\code\feature_extract\data\features" \
    --batch_size 32
```

对两个模态都执行特征提取后，使用 `--align` 对齐特征：

```bash
python scripts/extract_features.py \
    --modality P \
    --model resnet50 \
    --checkpoint "path/to/checkpoint.pth" \
    --align
```

**输出**:

- 特征文件: `data/features/{split}/{modality}/grade{label}/{patient_id}.npy`
- 特征信息: `data/features/{split}/features_{modality}.csv`
- 对齐后的特征: `data/features/{split}/aligned/`

### 步骤4: 查看训练结果

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
```

### 模型配置

```yaml
model:
  name: resnet18
  pretrained: true
  num_classes: 2
  freeze_stages: 0  # ResNet50设为3
```

### 训练配置

```yaml
training:
  batch_size: 32
  epochs: 100
  learning_rate: 0.0001
  loss_type: ce
  early_stop_patience: 10
```

## 数据格式

### 原始数据结构

```bash
D:\data\raw\Grade\
├── JM/
│   ├── fold1/
│   │   ├── grade0/
│   │   │   ├── {patient_id}/
│   │   │   │   ├── A/
│   │   │   │   │   └── {patient_id}_a_slice_001.png
│   │   │   │   └── P/
│   │   │   │       └── {patient_id}_p_slice_001.png
│   │   └── grade1/
│   └── fold2/ ...
└── OtherHospital/
    └── grade0/ ...
```

### 数据划分策略

- **JM医院**: 按病人ID进行7:3分层抽样（训练集:验证集）
- **其他医院**: 全部作为测试集（外验）
- **重要**: 同一病人的所有切片必须在同一集合中

### 特征格式

每个病人的特征保存为 `.npy` 文件，形状为 `(n, m)`：

- `n`: 该病人的切片数量
- `m`: 特征维度（ResNet18: 512, ResNet50: 2048）

## 常见问题

### Q1: 如何修改模型架构？

编辑 `src/models/model_loader.py` 中的 `load_model` 函数，添加新的模型支持。

### Q2: 如何处理类别不平衡？

使用Focal Loss或Asymmetric Loss：

```bash
python scripts/train.py --loss_type focal ...
```

### Q3: 如何调整数据增强？

修改 `config/default_config.yaml` 中的 `augmentation` 部分，或在训练时指定：

```bash
python scripts/train.py --rotation_degrees 30 ...
```

### Q4: 训练中断后如何恢复？

使用 `--resume` 参数：

```bash
python scripts/train.py --resume ...
```

## 实验记录

所有实验自动记录以下信息：

- 训练和验证曲线（Loss、AUC、Accuracy等）
- 模型配置和超参数
- 最佳模型检查点

## 下一步

完成特征提取后，可以：

1. 使用提取的特征训练多模态融合模型
2. 使用L2正则化的concat策略融合A和P模态
3. 进行进一步的分类或预测任务

## 许可证

本项目仅供学术研究使用。

## 联系方式

如有问题，请联系项目维护者。
