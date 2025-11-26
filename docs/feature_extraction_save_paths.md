# 特征提取保存路径文档

## 概述

本文档详细记录了项目中特征提取的保存路径逻辑，包括输出目录结构、文件命名规则和相关的配置参数。

## 默认输出目录

### 主要输出目录
```
/home/wwt/code/feature_extract/data/features/
```

### 配置方式
1. **默认值**：`data/features/`
2. **命令行参数**：`--output_dir <path>`
3. **配置文件**：`config/default_config.yaml` 中的 `experiment.output_dir`

## 目录结构

### 完整的目录层次结构
```
output_dir/                          # 根目录（默认：data/features）
├── train/                          # 训练集特征
│   ├── A/                          # 模态A特征
│   │   ├── grade0/                 # 标签0
│   │   │   ├── patient_001.npy     # 患者特征文件
│   │   │   ├── patient_002.npy
│   │   │   └── ...
│   │   └── grade1/                 # 标签1
│   │       ├── patient_003.npy
│   │       └── ...
│   │   └── features_A.csv          # 特征信息记录
│   └── P/                          # 模态P特征
│       ├── grade0/
│       ├── grade1/
│       └── features_P.csv
├── val/                            # 验证集特征（结构同train）
│   ├── A/
│   └── P/
├── test/                           # 测试集特征（结构同train）
│   ├── A/
│   └── P/
└── aligned/                        # 多模态对齐后的特征
    ├── A/
    ├── P/
    ├── aligned_features_A.csv
    └── aligned_features_P.csv
```

## 文件命名规则

### 1. 特征文件 (.npy)
```
{patient_id}.npy
示例：
- patient_001.npy
- JM001.npy
- P001.npy
```

### 2. 特征信息文件 (.csv)
```
features_{modality}.csv
示例：
- features_A.csv
- features_P.csv
```

### 3. 对齐特征信息文件 (.csv)
```
aligned_features_{modality}.csv
示例：
- aligned_features_A.csv
- aligned_features_P.csv
```

## 核心代码实现

### 1. 输出目录设置
```python
# scripts/extract_features.py (第56-57行)
if args.output_dir is None:
    args.output_dir = str(Path(project_root) / 'data' / 'features')
```

### 2. 特征保存逻辑
```python
# src/feature_extraction/extractor.py (第107-113行)
def extract_patient_features(self, dataloader, output_dir, modality):
    for patient_id, features, label in tqdm(dataloader):
        # 创建保存目录
        label_name = f"grade{label}"
        save_dir = Path(output_dir) / modality / label_name
        save_dir.mkdir(parents=True, exist_ok=True)

        # 保存特征文件
        feature_file = save_dir / f"{patient_id}.npy"
        np.save(feature_file, features.numpy())
```

### 3. 特征信息记录
```python
# src/feature_extraction/extractor.py (第125-129行)
info_df = pd.DataFrame(feature_info)
info_csv = Path(output_dir) / f'features_{modality}.csv'
info_df.to_csv(info_csv, index=False)
```

## 特征文件格式

### .npy 文件内容
- **类型**：NumPy 数组
- **形状**：`(num_slices, feature_dim)`
- **数据类型**：`float32`
- **内容**：患者的所有切片特征拼接而成的矩阵

### CSV 信息文件列
```csv
patient_id,modality,label,num_slices,feature_dim,feature_path
patient_001,A,0,15,512,/path/to/patient_001.npy
patient_002,P,1,22,2048,/path/to/patient_002.npy
```

## 多模态对齐

### 对齐功能
```python
# src/feature_extraction/extractor.py (第134-184行)
def align_patient_features(modalities, feature_dir, output_dir):
    """对齐不同模态的患者特征，确保顺序一致"""
    # 获取患者交集
    # 重新排序特征
    # 保存对齐后的特征
```

### 对齐后的保存位置
```
output_dir/aligned/
├── A/
├── P/
├── aligned_features_A.csv
└── aligned_features_P.csv
```

## 使用方式

### 1. 通过管理脚本
```bash
# 使用默认输出目录
python scripts/manage.py extract --modality A --model resnet18

# 指定输出目录
python scripts/manage.py extract \
  --modality A \
  --model resnet18 \
  --output_dir /path/to/output
```

### 2. 直接使用提取脚本
```bash
python scripts/extract_features.py \
  --checkpoint outputs/feature_extract/checkpoints/best_model.pth \
  --data_dir data/splits \
  --modality A \
  --output_dir data/features
```

## 批量处理

### 自动处理三个数据集
```python
# scripts/extract_features.py (第84-114行)
for split in ['train', 'val', 'test']:
    csv_path = data_dir / f'{split}_{args.modality}.csv'
    dataset = create_dataset(csv_path, transform=transforms)

    # 提取并保存特征
    split_output_dir = Path(args.output_dir) / split
    extractor.extract_patient_features(...)
```

## 配置参数

### 1. 命令行参数
```bash
--output_dir:    特征输出目录
--modality:      模态名称 (A/P)
--batch_size:    批处理大小
--device:        计算设备
--align:         是否对齐多模态特征
```

### 2. 配置文件参数
```yaml
# config/default_config.yaml
experiment:
  output_dir: "./outputs/feature_extract"

data:
  modalities: ['A', 'P']

feature_extraction:
  batch_size: 64
  device: cuda
```

## 示例命令

### 基本特征提取
```bash
# 提取模态A的特征到默认目录
python scripts/manage.py extract --modality A --model resnet18

# 输出目录：data/features/train/A/, data/features/val/A/, data/features/test/A/
```

### 指定输出目录
```bash
# 指定自定义输出目录
python scripts/manage.py extract \
  --modality P \
  --model resnet50 \
  --output_dir /path/to/custom/features
```

### 多模态特征对齐
```bash
# 提取A模态特征
python scripts/manage.py extract --modality A --model resnet18

# 提取P模态特征
python scripts/manage.py extract --modality P --model resnet50

# 对齐特征（确保患者顺序一致）
python scripts/manage.py extract --modality A --align
```

## 注意事项

1. **权限要求**：确保有写入输出目录的权限
2. **磁盘空间**：特征文件可能较大，确保有足够空间
3. **路径一致性**：使用绝对路径避免相对路径问题
4. **文件覆盖**：相同患者ID会覆盖现有特征文件
5. **对齐依赖**：多模态对齐需要先提取所有模态的特征

## 历史记录

- **2025-11-26**: 初始文档创建，记录特征提取保存路径逻辑

---

*本文档由 Claude Code 自动生成*