"""
CMTA模型训练脚本

使用方法:
python scripts/train_cmta.py --config config/best_hparams/cmta.yaml --data_dir /path/to/data
"""

import os
import sys
import argparse
import time
from pathlib import Path
import torch
import torch.nn as nn
import torch.optim as optim

# 添加项目根目录到路径
project_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(project_root))

from src.utils.config import Config
from src.utils.logger import setup_logger
from src.utils.seed import set_seed
from src.models.cmta import CMTA
from src.training.cmta_trainer import CMTATrainer
from src.data.cmta_dataset import create_cmta_dataloaders
from src.utils.optimizer import define_optimizer
from src.utils.scheduler import define_scheduler


def parse_args():
    """解析命令行参数"""
    parser = argparse.ArgumentParser(description='CMTA多模态融合训练')
    parser.add_argument('--config', type=str, required=True, help='配置文件路径')
    parser.add_argument('--data_dir', type=str, help='数据目录路径')
    parser.add_argument('--modalities', nargs='+', default=['A', 'P'], help='使用的模态')
    parser.add_argument('--model_size', type=str, default='small', choices=['small', 'large'], help='模型尺寸')
    parser.add_argument('--batch_size', type=int, default=1, help='批次大小')
    parser.add_argument('--epochs', type=int, default=50, help='训练轮数')
    parser.add_argument('--learning_rate', type=float, default=1e-4, help='学习率')
    parser.add_argument('--alpha', type=float, default=0.5, help='队列损失权重')
    parser.add_argument('--beta', type=float, default=0.1, help='辅助损失权重')
    parser.add_argument('--seed', type=int, default=42, help='随机种子')
    parser.add_argument('--device', type=str, default='cuda', help='训练设备')
    parser.add_argument('--resume', type=str, help='恢复训练的检查点路径')
    parser.add_argument('--output_dir', type=str, default='./outputs/cmta', help='输出目录')

    return parser.parse_args()


def main():
    # 解析参数
    args = parse_args()

    # 设置随机种子
    set_seed(args.seed)

    # 加载配置
    config = Config.from_yaml(args.config)

    # 设置设备
    device = torch.device(args.device if torch.cuda.is_available() else 'cpu')
    print(f"使用设备: {device}")

    # 创建输出目录
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # 设置日志
    logger = setup_logger(
        name="cmta_training",
        log_dir=output_dir / 'logs',
        console=True
    )

    logger.info("=== CMTA多模态融合训练开始 ===")
    logger.info(f"配置文件: {args.config}")
    logger.info(f"输出目录: {output_dir}")

    # 更新配置参数
    if args.data_dir:
        config.data.root_dir = args.data_dir
    if args.modalities:
        config.data.modalities = args.modalities
    if args.batch_size:
        config.training.batch_size = args.batch_size
    if args.epochs:
        config.training.epochs = args.epochs
    if args.learning_rate:
        config.training.learning_rate = args.learning_rate
    if args.alpha:
        config.training.cmta.alpha = args.alpha
    if args.beta:
        config.training.cmta.beta = args.beta

    # 创建数据加载器
    logger.info("创建数据加载器...")
    try:
        train_loader, val_loader, test_loader = create_cmta_dataloaders(config)
        logger.info(f"数据加载成功 - 训练: {len(train_loader)}, 验证: {len(val_loader)}, 测试: {len(test_loader)}")
    except Exception as e:
        logger.error(f"数据加载失败: {e}")
        return

    # 创建模型
    logger.info("创建CMTA模型...")
    model_params = {
        'n_classes': config.model.num_classes,
        'fusion': 'concat',
        'model_size': getattr(config.model.cmta, 'model_size', args.model_size),
        'feat_dim': getattr(config.model.cmta, 'feat_dim', 1024),
        'num_cluster': getattr(config.model.cmta, 'num_cluster', 64),
        'bank_length': getattr(config.model.cmta, 'bank_length', 16),
        'update_ratio': getattr(config.model.cmta, 'update_ratio', 0.1)
    }

    model = CMTA(**model_params)
    model = model.to(device)

    # 打印模型信息
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    logger.info(f"模型参数总数: {total_params:,}")
    logger.info(f"可训练参数: {trainable_params:,}")

    # 创建优化器
    optimizer = define_optimizer(config.training, model)

    # 创建学习率调度器
    scheduler = define_scheduler(config.training, optimizer)

    # 创建训练器
    trainer = CMTATrainer(
        model=model,
        args=config.training,
        results_dir=str(output_dir),
        fold=0
    )

    # 开始训练
    logger.info("开始训练...")
    start_time = time.time()

    try:
        best_score, best_epoch = trainer.train(
            train_loader=train_loader,
            val_loader=val_loader,
            test_loader=test_loader,
            optimizer=optimizer,
            scheduler=scheduler,
            resume_checkpoint=args.resume
        )

        training_time = time.time() - start_time
        logger.info(f"训练完成 - 最佳分数: {best_score:.4f} (Epoch {best_epoch})")
        logger.info(f"训练用时: {training_time/60:.2f} 分钟")

    except KeyboardInterrupt:
        logger.info("训练被用户中断")
    except Exception as e:
        logger.error(f"训练过程中发生错误: {e}")
        raise

    logger.info("=== CMTA训练结束 ===")


if __name__ == '__main__':
    main()