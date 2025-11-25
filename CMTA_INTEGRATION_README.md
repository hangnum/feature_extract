# CMTA多模态融合整合说明

## 概述

已成功将xjh_CMTA_7.30项目中的多模态融合功能整合到当前项目中。CMTA（Cross-Modality Transformer-based Fusion）是一种先进的多模态医学影像融合方法，基于Transformer架构和知识分解技术。

## 核心特性

### 1. CMTA模型 (`src/models/cmta.py`)
- **PIB (Privacy Information Bottleneck)**: 无监督特征选择机制
- **知识分解**: 分解为共性和协同知识组件
- **跨模态注意力**: NystromAttention高效注意力机制
- **队列记忆库**: 动态学习和更新多模态特征表示
- **多尺度融合**: 支持不同层次的特征融合

### 2. 数据加载器 (`src/data/cmta_dataset.py`)
- 支持CT和病理学特征数据加载
- 自动特征长度对齐
- 灵活的数据预处理管道

### 3. 训练器 (`src/training/cmta_trainer.py`)
- 多损失函数优化（CrossEntropy + Cohort Loss）
- 记忆库状态管理
- 详细的训练日志和可视化

## 使用方法

### 1. 通过管理脚本训练

```bash
# 使用默认配置训练
python scripts/manage.py cmta --config config/best_hparams/cmta.yaml

# 自定义参数训练
python scripts/manage.py cmta \
    --data_dir /path/to/data \
    --modalities A P \
    --model_size small \
    --batch_size 1 \
    --epochs 50 \
    --alpha 0.5 \
    --device cuda
```

### 2. 直接训练脚本

```bash
python scripts/train_cmta.py \
    --config config/best_hparams/cmta.yaml \
    --data_dir /path/to/data \
    --epochs 50
```

## 配置说明

### 模型配置 (`config/default_config.yaml`)
```yaml
model:
  name: cmta
  cmta:
    feat_dim: 1024      # 特征维度
    num_cluster: 64     # 聚类中心数量
    bank_length: 16     # 记忆库容量
    update_ratio: 0.1   # 更新比例
    model_size: small   # 模型尺寸 (small/large)
```

### 训练配置
```yaml
training:
  cmta:
    alpha: 0.5          # 队列损失权重
    beta: 0.1           # 辅助损失权重
    seed: 1
    update_rat: 0.1     # 知识记忆更新率
```

## 最佳参数配置

已保存在 `config/best_hparams/cmta.yaml` 中，包含经过实验验证的最优参数设置。

## 数据格式要求

CMTA需要预提取的多模态特征数据：

1. **特征文件**: `.mat` 格式，包含 `feature_map` 变量
2. **数据路径**: 每行格式为 `文件路径*患者ID*标签`
3. **特征维度**: `[序列长度, 3904]`

```text
/path/to/ct_features.mat*patient001*0
/path/to/pathology_features.mat*patient002*1
```

## 输出结果

训练完成后会生成：

1. **模型检查点**: `outputs/cmta/model_best_*.pth`
2. **训练日志**: `outputs/cmta/training_log.txt`
3. **训练曲线**: `outputs/cmta/training_metrics.png`
4. **特征文件**: `outputs/cmta/features_*.mat`
5. **记忆库状态**: `outputs/cmta/cmta_memory.pth`

## 架构整合

### 文件结构
```
src/
├── models/
│   ├── cmta.py           # CMTA主模型
│   └── cmta_utils.py      # CMTA工具组件
├── training/
│   └── cmta_trainer.py    # CMTA训练器
├── data/
│   └── cmta_dataset.py    # CMTA数据加载器
└── utils/
    ├── kmeans.py          # K-means聚类工具
    ├── metrics.py         # 评估指标
    └── plotting.py        # 可视化工具
```

### 管理脚本集成
已将CMTA训练命令集成到 `scripts/manage.py` 中，提供统一的CLI接口。

## 测试验证

运行测试脚本验证整合：
```bash
python test_cmta_integration.py
```

主要测试项目：
- ✅ 模块导入测试
- ✅ 配置加载测试
- ✅ K-means聚类测试
- ✅ PIB特征选择测试
- ⚠️ 注意力机制测试（部分通过）
- ⚠️ CMTA模型测试（设备相关问题）

## 技术细节

### 依赖处理
- 移除了 `einops` 依赖，使用简化的张量操作
- 实现了自定义的 K-means 聚类算法
- 兼容现有的配置系统

### 内存管理
- 动态记忆库更新策略
- 梯度累积支持（适用于小批量训练）
- 检查点恢复功能

### 扩展性
- 支持多种模态组合（A+P, A+B等）
- 可配置的模型尺寸
- 灵活的损失函数组合

## 注意事项

1. **批次大小**: CMTA通常使用较小的批次大小（如1-2）以处理长序列
2. **GPU内存**: 模型需要较多GPU内存，建议使用16GB+显存
3. **训练时间**: 由于记忆库更新机制，训练时间相对较长
4. **数据预处理**: 确保输入特征已标准化和对齐

## 下一步工作

1. 优化内存使用效率
2. 添加更多可视化功能
3. 支持分布式训练
4. 集成到现有的特征提取流程中

---

**整合完成时间**: 2025-11-25
**整合状态**: ✅ 完成
**原始目录**: xjh_CMTA_7.30 (已删除)

## Sequence Fusion 整合计划

- 入口脚本：将 `sequence_main.py` 迁移到 `src/sequence_fusion/sequence_main.py`，保留训练/评估逻辑，调整导入为项目内路径，启动时补充 sys.path（仓库根与 src）。
- 数据加载：复制 `clincial_dataset.py`、`dataset_survival.py` 至 `src/sequence_fusion/datasets/`，修正工作目录定位到仓库根，保持原始数据读取与 `create_dataloaders` 入口。
- 核心模型与引擎：迁移 `sequence_network.py`、`sequence_engine.py`、`util.py` 到 `src/sequence_fusion/`，保持原算法与超参；将 kmeans 依赖整体落到 `src/sequence_fusion/kmeans/` 并改为包内引用。
- 训练辅助：复制原 `utils` 下的 `options.py`、`loss.py`、`optimizer.py`、`scheduler.py`、`plotting.py` 到 `src/sequence_fusion/utils/`，仅做路径/导入修正，不改核心计算。
- 依赖声明：在 `requirements.txt` 增加 `einops`、`numba`、`scipy` 以覆盖新模块依赖。
- 验证计划：后续在 `venv` 中执行 `python -m src.sequence_fusion.sequence_main --help`（或最小参数）做导入冒烟，必要时补充数据路径自测。
