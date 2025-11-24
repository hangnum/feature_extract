"""
训练主脚本

用于训练单个模态的模型
"""

import sys
from pathlib import Path

# 添加项目根目录到Python路径
project_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(project_root))

import argparse
import torch
import torch.optim as optim
from torch.utils.data import DataLoader

from src.data.dataset import MedicalImageDataset
from src.data.transforms import get_train_transform, get_val_transform
from src.models.model_loader import load_model
from src.models.losses import get_loss_function
from src.training.trainer import Trainer
from src.utils.config import Config, save_default_config
from src.utils.logger import setup_logger
from src.utils.seed import set_seed


def get_optimizer(model, config):
    """获取优化器"""
    if config.training.optimizer == 'adam':
        optimizer = optim.Adam(
            model.parameters(),
            lr=config.training.learning_rate,
            weight_decay=config.training.weight_decay
        )
    elif config.training.optimizer == 'adamw':
        optimizer = optim.AdamW(
            model.parameters(),
            lr=config.training.learning_rate,
            weight_decay=config.training.weight_decay
        )
    elif config.training.optimizer == 'sgd':
        optimizer = optim.SGD(
            model.parameters(),
            lr=config.training.learning_rate,
            momentum=0.9,
            weight_decay=config.training.weight_decay
        )
    else:
        raise ValueError(f"不支持的优化器: {config.training.optimizer}")
    
    return optimizer


def get_scheduler(optimizer, config, steps_per_epoch):
    """获取学习率调度器"""
    if config.training.scheduler == 'cosine':
        scheduler = optim.lr_scheduler.CosineAnnealingLR(
            optimizer,
            T_max=config.training.epochs * steps_per_epoch
        )
    elif config.training.scheduler == 'step':
        scheduler = optim.lr_scheduler.StepLR(
            optimizer,
            step_size=30,
            gamma=0.1
        )
    elif config.training.scheduler == 'plateau':
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            optimizer,
            mode='max',
            factor=0.5,
            patience=5,
            verbose=True
        )
    else:
        scheduler = None
    
    return scheduler


def main():
    parser = argparse.ArgumentParser(description='训练脚本')
    parser.add_argument('--config', type=str, default=None, help='配置文件路径')
    parser.add_argument('--modality', type=str, required=True, choices=['A', 'P'], help='训练的模态')
    parser.add_argument('--model', type=str, default='resnet18', help='模型名称')
    parser.add_argument('--epochs', type=int, default=None, help='训练轮数')
    parser.add_argument('--batch_size', type=int, default=None, help='batch size')
    parser.add_argument('--learning_rate', type=float, default=None, help='学习率')
    parser.add_argument('--loss_type', type=str, default=None, help='损失函数类型')
    parser.add_argument('--resume', action='store_true', help='从检查点恢复训练')
    parser.add_argument('--device', type=str, default='cuda', help='设备')
    
    args = parser.parse_args()

    # Load config: CLI path takes precedence, otherwise default file
    if args.config:
        config = Config.from_yaml(args.config)
        config_source = args.config
    else:
        default_config_path = Path(project_root) / 'config' / 'default_config.yaml'
        if default_config_path.exists():
            config = Config.from_yaml(str(default_config_path))
            config_source = str(default_config_path)
        else:
            config = Config()
            config_source = 'Config() default'

    print(f'Using config: {config_source}')

    # Apply CLI overrides
    config.update_from_args(args)

    
    # 设置随机种子
    set_seed(config.experiment.seed)
    
    # 设置日志
    logger = setup_logger(
        name="feature_extract",
        log_dir=Path(config.experiment.output_dir) / 'logs',
        console=True
    )
    
    logger.info("=" * 60)
    logger.info(f"训练模型: {config.model.name}")
    logger.info(f"模态: {args.modality}")
    logger.info("=" * 60)
    
    # 打印配置
    logger.info(f"配置信息:")
    logger.info(f"  Batch Size: {config.training.batch_size}")
    logger.info(f"  Epochs: {config.training.epochs}")
    logger.info(f"  Learning Rate: {config.training.learning_rate}")
    logger.info(f"  Loss Type: {config.training.loss_type}")
    logger.info(f"  Optimizer: {config.training.optimizer}")
    logger.info(f"  Scheduler: {config.training.scheduler}")
    logger.info(f"  Early Stop Patience: {config.training.early_stop_patience}")
    
    # 保存配置
    config_save_path = Path(config.experiment.output_dir) / 'logs' / config.experiment.name / 'config.yaml'
    config.to_yaml(str(config_save_path))
    logger.info(f"配置已保存至: {config_save_path}")
    
    # 数据路径
    data_dir = Path(project_root) / 'data' / 'splits'
    train_csv = data_dir / f'train_{args.modality}.csv'
    val_csv = data_dir / f'val_{args.modality}.csv'
    
    # 检查文件是否存在
    if not train_csv.exists() or not val_csv.exists():
        logger.error("数据划分文件不存在！请先运行 preprocess_data.py")
        return
    
    # 数据变换
    train_transform = get_train_transform(
        image_size=config.data.image_size,
        normalize_mean=config.augmentation.normalize_mean,
        normalize_std=config.augmentation.normalize_std,
        horizontal_flip=config.augmentation.horizontal_flip,
        rotation_degrees=config.augmentation.rotation_degrees
    )
    
    val_transform = get_val_transform(
        image_size=config.data.image_size,
        normalize_mean=config.augmentation.normalize_mean,
        normalize_std=config.augmentation.normalize_std
    )
    
    # 数据集
    train_dataset = MedicalImageDataset(str(train_csv), transform=train_transform)
    val_dataset = MedicalImageDataset(str(val_csv), transform=val_transform)
    
    # 数据加载器
    train_loader = DataLoader(
        train_dataset,
        batch_size=config.training.batch_size,
        shuffle=True,
        num_workers=config.data.num_workers,
        pin_memory=True
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=config.training.batch_size,
        shuffle=False,
        num_workers=config.data.num_workers,
        pin_memory=True
    )
    
    # 模型
    model = load_model(
        model_name=config.model.name,
        num_classes=config.model.num_classes,
        pretrained=config.model.pretrained,
        freeze_stages=config.model.freeze_stages
    )
    
    # 损失函数
    criterion = get_loss_function(
        loss_type=config.training.loss_type,
        num_classes=config.model.num_classes,
        device=config.training.device
    )
    
    # 优化器
    optimizer = get_optimizer(model, config)
    
    # 学习率调度器
    steps_per_epoch = len(train_loader)
    scheduler = get_scheduler(optimizer, config, steps_per_epoch)
    
    # 训练器
    trainer = Trainer(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        criterion=criterion,
        optimizer=optimizer,
        scheduler=scheduler,
        device=config.training.device,
        output_dir=config.experiment.output_dir,
        experiment_name=config.experiment.name,
        early_stop_patience=config.training.early_stop_patience,
        log_interval=config.experiment.log_interval
    )
    
    # 训练
    trainer.train(
        num_epochs=config.training.epochs,
        resume=args.resume
    )
    
    # 保存最佳超参数
    best_hparams_dir = Path(project_root) / 'config' / 'best_hparams'
    best_hparams_dir.mkdir(parents=True, exist_ok=True)
    best_hparams_path = best_hparams_dir / f'{config.model.name}_{args.modality}.yaml'
    
    config.to_yaml(str(best_hparams_path))
    logger.info(f"最佳超参数已保存至: {best_hparams_path}")


if __name__ == '__main__':
    main()
