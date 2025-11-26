# ResNet 微调过程中的参数和配置保存逻辑

## 概述

本文档详细记录了ResNet微调过程中最佳参数、配置文件和模型检查点的保存逻辑。系统实现了智能的参数保存机制，只在验证性能提升时更新最佳参数。

## 核心文件结构

```
/home/wwt/code/feature_extract/
├── scripts/train.py                 # 主训练脚本
├── src/training/trainer.py          # 训练器实现
└── config/
    ├── default_config.yaml          # 默认配置
    └── best_hparams/                # 最佳超参数存储
        ├── resnet18_A.yaml          # ResNet18+模态A的最佳参数
        ├── resnet50_P.yaml          # ResNet50+模态P的最佳参数
        └── ...
```

## 1. 最佳参数保存逻辑

### 触发条件
- **验证AUC提升时**：当前验证集AUC > 历史最佳验证集AUC
- **训练完成后**：保存最终的训练结果

### 实现位置
- **文件**：`scripts/train.py` (第335-370行)
- **关键函数**：`_update_best_hyperparameters()`

### 保存内容
```yaml
# 完整的训练配置
data:
  root_dir: "数据路径"
  modalities: ["A", "P"]
  ...
model:
  name: "resnet18"
  pretrained: true
  ...
training:
  batch_size: 128
  learning_rate: 0.0001
  ...

# 最佳验证指标
best_metrics:
  epoch: 10
  val_auc: 0.85
  val_accuracy: 0.82
  val_sensitivity: 0.78
  val_specificity: 0.85
  timestamp: "2025-11-24T14:05:04"

# 测试集性能（如果有）
test_metrics:
  accuracy: 0.80
  auc: 0.83
  ...

# 更新时间戳
updated_at: "2025-11-24T14:05:45.969008"
```

### 命名规则
```
config/best_hparams/{model_name}_{modality}.yaml
示例：
- config/best_hparams/resnet18_A.yaml
- config/best_hparams/resnet50_P.yaml
- config/best_hparams/swin_t_A.yaml
```

## 2. 配置文件保存

### 保存时机
- **训练开始前**：自动保存当前实验的完整配置

### 保存位置
```
outputs/logs/{experiment_name}/config.yaml
示例：
- outputs/logs/exp_20251124_140504/config.yaml
```

### 实现代码
```python
# scripts/train.py (第167-169行)
config_save_path = Path(config.experiment.output_dir) / 'logs' / config.experiment.name / 'config.yaml'
config_save_path.parent.mkdir(parents=True, exist_ok=True)
config.to_yaml(str(config_save_path))
```

## 3. 模型检查点保存

### 实现位置
- **文件**：`src/training/trainer.py` (第148-175行)

### 保存类型

#### 3.1 最新检查点（每次epoch）
```python
# 总是保存最新的训练状态
latest_path = self.checkpoint_dir / 'latest.pth'
torch.save(checkpoint, latest_path)
```

#### 3.2 最佳模型（仅验证提升时）
```python
# 仅在验证AUC提升时保存
if is_best:
    best_path = self.checkpoint_dir / 'best_model.pth'
    torch.save(checkpoint, best_path)
```

### 检查点内容
```python
checkpoint = {
    'epoch': self.current_epoch,
    'model_state_dict': self.model.state_dict(),
    'optimizer_state_dict': self.optimizer.state_dict(),
    'scheduler_state_dict': self.scheduler.state_dict(),
    'best_val_auc': self.best_val_auc,
    'history': self.history
}
```

### 触发逻辑
```python
# src/training/trainer.py (第172行)
is_best = val_metrics['auc'] > self.best_val_auc
if is_best:
    self.best_val_auc = val_metrics['auc']
    # 保存最佳模型
```

## 4. 训练历史记录

### 记录格式
- **CSV文件**：`outputs/logs/{experiment_name}/training_history.csv`
- **列名**：epoch, train_loss, val_loss, val_acc, val_auc, val_sensitivity, val_specificity

### 实现代码
```python
# src/training/trainer.py (第281-286行)
history_df = pd.DataFrame(self.history)
history_path = self.log_dir / f'{self.experiment_name}_training_history.csv'
history_df.to_csv(history_path, index=False)
```

## 5. 关键代码片段

### 5.1 最佳参数更新逻辑
```python
# scripts/train.py (第335-370行)
def _update_best_hyperparameters(config, best_metrics, test_metrics_result=None):
    """更新最佳超参数"""
    model_name = config.model.name
    modality = config.data.modalities[0]  # 假设第一个模态是主要的

    # 构建文件路径
    best_hparams_path = Path('config/best_hparams') / f'{model_name}_{modality}.yaml'
    best_hparams_path.parent.mkdir(parents=True, exist_ok=True)

    # 读取历史最佳AUC
    previous_best_auc = -1
    if best_hparams_path.exists():
        prev_data = yaml.safe_load(best_hparams_path.read_text(encoding='utf-8'))
        previous_best_auc = prev_data.get('best_metrics', {}).get('val_auc', -1)

    # 仅在当前更好时更新
    current_best_auc = best_metrics['val_auc']
    if current_best_auc > previous_best_auc:
        # 准备保存的数据
        to_save = config.to_dict()
        to_save['best_metrics'] = best_metrics
        to_save['test_metrics'] = test_metrics_result
        to_save['updated_at'] = datetime.now().isoformat()

        # 写入文件
        with open(best_hparams_path, 'w', encoding='utf-8') as f:
            yaml.dump(to_save, f, default_flow_style=False, allow_unicode=True)
```

### 5.2 验证提升检查
```python
# src/training/trainer.py (第169-175行)
def _validate(self, val_loader):
    """验证阶段"""
    self.model.eval()
    val_metrics = self._compute_metrics(val_loader)

    # 检查是否是最佳模型
    is_best = val_metrics['auc'] > self.best_val_auc

    # 更新历史
    self._update_history(val_metrics)

    return val_metrics, is_best
```

## 6. 使用方式

### 6.1 加载最佳参数
```python
from src.utils.config import Config

# 加载最佳参数配置
best_config = Config.from_file('config/best_hparams/resnet18_A.yaml')
```

### 6.2 继续训练
```python
# 从最佳检查点继续训练
checkpoint = torch.load('outputs/feature_extract/checkpoints/best_model.pth')
model.load_state_dict(checkpoint['model_state_dict'])
optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
```

## 7. 特点和优势

1. **智能更新**：只在性能提升时更新，避免覆盖更好的结果
2. **完整记录**：保存完整的训练配置和性能指标
3. **版本控制**：通过时间戳跟踪更新历史
4. **自动管理**：无需手动干预，自动创建目录和文件
5. **标准化命名**：统一的文件命名规则，便于管理
6. **可复现性**：保存的配置可用于复现实验

## 8. 注意事项

1. **存储空间**：每次训练都会保存检查点，需定期清理
2. **并发训练**：多进程训练时注意文件锁的问题
3. **权限要求**：确保有写入`config/best_hparams/`目录的权限
4. **备份重要**：重要实验结果建议手动备份

## 历史记录

- **2025-11-24**: 初始文档创建，记录ResNet训练保存逻辑

---

*本文档由 Claude Code 自动生成*