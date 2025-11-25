"""
训练器

管理完整的训练流程，包括训练、验证、早停、断点续训等
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from pathlib import Path
from typing import Optional, Dict, Tuple
import pandas as pd
import logging
from tqdm import tqdm

from src.training.metrics import evaluate_model

logger = logging.getLogger("feature_extract")


class Trainer:
    """模型训练器"""
    
    def __init__(
        self,
        model: nn.Module,
        train_loader: DataLoader,
        val_loader: DataLoader,
        criterion: nn.Module,
        optimizer: optim.Optimizer,
        scheduler: Optional[optim.lr_scheduler._LRScheduler] = None,
        device: str = 'cuda',
        output_dir: str = './outputs',
        experiment_name: str = 'experiment',
        early_stop_patience: int = 10,
        early_stop_enabled: bool = True,
        log_interval: int = 10
    ):
        """
        初始化训练器  # 编码修复：将乱码恢复为中文注释

        Args:
            model: 模型  # 编码修复：将乱码恢复为中文注释
            train_loader: 训练数据加载器  # 编码修复：将乱码恢复为中文注释
            val_loader: 验证数据加载器  # 编码修复：将乱码恢复为中文注释
            criterion: 损失函数  # 编码修复：将乱码恢复为中文注释
            optimizer: 优化器  # 编码修复：将乱码恢复为中文注释
            scheduler: 学习率调度器  # 编码修复：将乱码恢复为中文注释
            device: 设备  # 编码修复：将乱码恢复为中文注释
            output_dir: 输出目录  # 编码修复：将乱码恢复为中文注释
            experiment_name: 实验名称  # 编码修复：将乱码恢复为中文注释
            early_stop_patience: 早停耐心值  # 编码修复：将乱码恢复为中文注释
            early_stop_enabled: 是否启用早停  # 编码修复：将乱码恢复为中文注释
            log_interval: 日志间隔  # 编码修复：将乱码恢复为中文注释
        """
        self.model = model.to(device)
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.criterion = criterion
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.device = device
        self.early_stop_patience = early_stop_patience
        self.early_stop_enabled = early_stop_enabled
        self.log_interval = log_interval
        
        # 创建输出目录
        self.output_dir = Path(output_dir)
        self.checkpoint_dir = self.output_dir / 'checkpoints'
        self.log_dir = self.output_dir / 'logs' / experiment_name
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        self.log_dir.mkdir(parents=True, exist_ok=True)
        
        # TensorBoard
        self.writer = SummaryWriter(log_dir=str(self.log_dir))
        
        # 训练状态
        self.current_epoch = 0
        self.best_val_auc = 0.0
        self.epochs_without_improvement = 0
        self.history = {
            'epoch': [],
            'train_loss': [],
            'val_loss': [],
            'val_acc': [],
            'val_auc': [],
            'val_sensitivity': [],
            'val_specificity': []
        }
    
    def train_epoch(self) -> float:
        """
        训练一个epoch
        
        Returns:
            平均训练损失
        """
        self.model.train()
        total_loss = 0.0
        
        progress_bar = tqdm(
            self.train_loader,
            desc=f'Epoch {self.current_epoch + 1} [Train]'
        )
        
        for batch_idx, (images, labels) in enumerate(progress_bar):
            images = images.to(self.device)
            labels = labels.to(self.device)
            
            # 前向传播
            self.optimizer.zero_grad()
            outputs = self.model(images)
            loss = self.criterion(outputs, labels)
            
            # 反向传播
            loss.backward()
            self.optimizer.step()
            
            total_loss += loss.item()
            
            # 更新进度条
            if (batch_idx + 1) % self.log_interval == 0:
                avg_loss = total_loss / (batch_idx + 1)
                progress_bar.set_postfix({'loss': f'{avg_loss:.4f}'})
        
        avg_loss = total_loss / len(self.train_loader)
        return avg_loss
    
    def validate(self) -> Tuple[float, Dict[str, float]]:
        """
        验证模型
        
        Returns:
            (验证损失, 验证指标)
        """
        val_loss, val_metrics = evaluate_model(
            model=self.model,
            dataloader=self.val_loader,
            criterion=self.criterion,
            device=self.device
        )
        
        return val_loss, val_metrics
    
    def save_checkpoint(self, is_best: bool = False) -> None:
        """
        保存检查点
        
        Args:
            is_best: 是否为最佳模型
        """
        checkpoint = {
            'epoch': self.current_epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'best_val_auc': self.best_val_auc,
            'history': self.history
        }
        
        if self.scheduler is not None:
            checkpoint['scheduler_state_dict'] = self.scheduler.state_dict()
        
        # 保存最新检查点
        latest_path = self.checkpoint_dir / 'latest.pth'
        torch.save(checkpoint, latest_path)
        
        # 保存最佳模型
        if is_best:
            best_path = self.checkpoint_dir / 'best_model.pth'
            torch.save(checkpoint, best_path)
            logger.info(f"保存最佳模型 (AUC: {self.best_val_auc:.4f})")
    
    def load_checkpoint(self, checkpoint_path: str) -> None:
        """
        加载检查点
        
        Args:
            checkpoint_path: 检查点路径
        """
        checkpoint = torch.load(checkpoint_path, map_location=self.device, weights_only=False)
        
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.current_epoch = checkpoint['epoch']
        self.best_val_auc = checkpoint['best_val_auc']
        self.history = checkpoint['history']
        
        if self.scheduler is not None and 'scheduler_state_dict' in checkpoint:
            self.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        
        logger.info(f"从 {checkpoint_path} 加载检查点 (Epoch {self.current_epoch})")
    
    def train(self, num_epochs: int, resume: bool = False) -> None:
        """
        Train the model.

        Args:
            num_epochs: total epochs to run
            resume: whether to resume from latest checkpoint
        """
        if resume:
            latest_checkpoint = self.checkpoint_dir / 'latest.pth'
            if latest_checkpoint.exists():
                self.load_checkpoint(str(latest_checkpoint))
                logger.info("Resumed from latest checkpoint")

        logger.info("=" * 60)
        logger.info("Start training")
        logger.info("=" * 60)
        logger.info(f"Total epochs: {num_epochs}")
        logger.info(f"Train size: {len(self.train_loader.dataset)}")
        logger.info(f"Val size: {len(self.val_loader.dataset)}")
        if self.early_stop_enabled:
            logger.info(f"Early stop enabled, patience: {self.early_stop_patience}")
        else:
            logger.info("Early stop disabled")

        for epoch in range(self.current_epoch, num_epochs):
            self.current_epoch = epoch

            train_loss = self.train_epoch()
            val_loss, val_metrics = self.validate()

            if self.scheduler is not None:
                if isinstance(self.scheduler, optim.lr_scheduler.ReduceLROnPlateau):
                    self.scheduler.step(val_metrics['auc'])
                else:
                    self.scheduler.step()

            self.history['epoch'].append(epoch + 1)
            self.history['train_loss'].append(train_loss)
            self.history['val_loss'].append(val_loss)
            self.history['val_acc'].append(val_metrics['accuracy'])
            self.history['val_auc'].append(val_metrics['auc'])
            self.history['val_sensitivity'].append(val_metrics['sensitivity'])
            self.history['val_specificity'].append(val_metrics['specificity'])

            self.writer.add_scalar('Loss/train', train_loss, epoch)
            self.writer.add_scalar('Loss/val', val_loss, epoch)
            self.writer.add_scalar('Metrics/accuracy', val_metrics['accuracy'], epoch)
            self.writer.add_scalar('Metrics/auc', val_metrics['auc'], epoch)
            self.writer.add_scalar('Metrics/sensitivity', val_metrics['sensitivity'], epoch)
            self.writer.add_scalar('Metrics/specificity', val_metrics['specificity'], epoch)

            logger.info(
                f"Epoch {epoch + 1}/{num_epochs} | "
                f"Train Loss: {train_loss:.4f} | "
                f"Val Loss: {val_loss:.4f} | "
                f"Val AUC: {val_metrics['auc']:.4f} | "
                f"Val Acc: {val_metrics['accuracy']:.4f}"
            )

            is_best = val_metrics['auc'] > self.best_val_auc
            if is_best:
                self.best_val_auc = val_metrics['auc']
                self.epochs_without_improvement = 0
            else:
                if self.early_stop_enabled:
                    self.epochs_without_improvement += 1
                else:
                    self.epochs_without_improvement = 0

            self.save_checkpoint(is_best=is_best)

            if self.early_stop_enabled and self.epochs_without_improvement >= self.early_stop_patience:
                logger.info(f"\nEarly stopping triggered! No improvement for {self.early_stop_patience} epochs")
                break

        self.save_history()

        logger.info("=" * 60)
        logger.info("Training finished")
        logger.info(f"Best Val AUC: {self.best_val_auc:.4f}")
        logger.info("=" * 60)

        self.writer.close()

    def save_history(self) -> None:
        """保存训练历史到CSV"""
        history_df = pd.DataFrame(self.history)
        history_path = self.log_dir / 'training_history.csv'
        history_df.to_csv(history_path, index=False)
        logger.info(f"训练历史已保存至: {history_path}")
