"""
CMTA模型的训练器实现

支持多模态特征融合训练，包含：
1. CMTA模型训练
2. 多损失函数优化
3. 队列记忆库管理
4. 验证和测试
"""

import os
import time
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Tuple, Optional, List
from sklearn.metrics import roc_auc_score, accuracy_score
import scipy.io as scio
import matplotlib.pyplot as plt

from ..models.cmta import CMTA
from ..utils.metrics import calculate_metrics, AverageMeter
from ..utils.plotting import plot_training_metrics


class CMTATrainer:
    """CMTA模型训练器"""

    def __init__(self,
                 model: CMTA,
                 args,
                 results_dir: str,
                 fold: int = 0):
        self.model = model
        self.args = args
        self.results_dir = results_dir
        self.fold = fold
        self.best_score = 0
        self.best_epoch = 0
        self.filename_best = None

        # 确保结果目录存在
        os.makedirs(results_dir, exist_ok=True)

        # 初始化损失函数
        self.criterion = self._build_criterion()

        # 训练指标记录
        self.metrics_history = {
            'train_auc': [],
            'train_acc': [],
            'train_loss': [],
            'val_auc': [],
            'val_acc': [],
            'val_loss': [],
            'test_auc': [],
            'test_acc': [],
            'test_loss': []
        }

    def _build_criterion(self):
        """构建损失函数"""
        if self.args.loss == "CohortLoss":
            return [nn.CrossEntropyLoss(), CohortLoss()]
        elif self.args.loss == "focal_loss":
            return [FocalLoss(alpha=0.24, gamma=2), nn.L1Loss()]
        else:
            return [nn.CrossEntropyLoss(), nn.L1Loss()]

    def train_epoch(self, train_loader, optimizer, epoch: int) -> Dict:
        """训练一个epoch"""
        losses = AverageMeter()
        acc_meter = AverageMeter()

        # 记录特征和标签
        fusion_features = []
        train_labels = []
        train_scores = []

        self.model.train()

        for batch_idx, (ct_features, pathology_features, labels) in enumerate(train_loader):
            if torch.cuda.is_available():
                ct_features = ct_features.cuda()
                pathology_features = pathology_features.cuda()
                labels = labels.type(torch.LongTensor).cuda()

            # 前向传播
            outputs = self.model(
                x_path=pathology_features,
                x_radio=ct_features,
                phase='train',
                label=labels
            )

            logits = outputs['logits']
            fusion = outputs['fusion']

            # 计算损失
            loss_ce = self.criterion[0](logits, labels)
            loss_cohort = self.criterion[1](outputs, labels)
            loss = loss_ce + self.args.alpha * loss_cohort

            # 计算准确率
            _, pred = torch.max(logits, dim=1)
            acc = (pred == labels).float().mean()
            batch_size = labels.size(0)

            # 更新统计量
            losses.update(loss.item(), batch_size)
            acc_meter.update(acc.item(), batch_size)

            # 保存特征用于分析
            fusion_features.append(fusion.detach().cpu())
            train_labels.append(labels.cpu())
            train_scores.append(F.softmax(logits, dim=1)[:, 1].cpu())

            # 反向传播
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            if batch_idx % 50 == 0:
                print(f'Epoch: [{epoch+1}/{self.args.num_epoch}] '
                      f'Iter: [{batch_idx+1}/{len(train_loader)}] '
                      f'Loss: {losses.val:.4f} ({losses.avg:.4f}) '
                      f'Acc: {acc_meter.val*100:.2f} ({acc_meter.avg*100:.2f})')

        # 保存记忆库
        self._save_memory_bank()

        # 计算epoch指标
        all_fusion_features = torch.cat(fusion_features, dim=0)
        all_labels = torch.cat(train_labels, dim=0)
        all_scores = torch.cat(train_scores, dim=0)

        try:
            train_auc = roc_auc_score(all_labels.numpy(), all_scores.numpy())
        except:
            train_auc = 0.5

        return {
            'loss': losses.avg,
            'acc': acc_meter.avg,
            'auc': train_auc,
            'features': all_fusion_features.numpy(),
            'labels': all_labels.numpy()
        }

    def validate(self, val_loader, phase: str = 'val') -> Dict:
        """验证模型"""
        losses = AverageMeter()
        acc_meter = AverageMeter()

        fusion_features = []
        val_labels = []
        val_scores = []

        self.model.eval()

        with torch.no_grad():
            for ct_features, pathology_features, labels in val_loader:
                if torch.cuda.is_available():
                    ct_features = ct_features.cuda()
                    pathology_features = pathology_features.cuda()
                    labels = labels.type(torch.LongTensor).cuda()

                # 前向传播
                outputs = self.model(
                    x_path=pathology_features,
                    x_radio=ct_features,
                    phase=phase,
                    label=labels
                )

                logits = outputs['logits']
                fusion = outputs['fusion']

                # 计算损失
                loss_ce = self.criterion[0](logits, labels)
                loss_cohort = self.criterion[1](outputs, labels)
                loss = loss_ce + self.args.alpha * loss_cohort

                # 计算准确率
                _, pred = torch.max(logits, dim=1)
                acc = (pred == labels).float().mean()
                batch_size = labels.size(0)

                # 更新统计量
                losses.update(loss.item(), batch_size)
                acc_meter.update(acc.item(), batch_size)

                # 保存特征
                fusion_features.append(fusion.cpu())
                val_labels.append(labels.cpu())
                val_scores.append(F.softmax(logits, dim=1)[:, 1].cpu())

        # 计算指标
        all_fusion_features = torch.cat(fusion_features, dim=0)
        all_labels = torch.cat(val_labels, dim=0)
        all_scores = torch.cat(val_scores, dim=0)

        try:
            val_auc = roc_auc_score(all_labels.numpy(), all_scores.numpy())
        except:
            val_auc = 0.5

        return {
            'loss': losses.avg,
            'acc': acc_meter.avg,
            'auc': val_auc,
            'features': all_fusion_features.numpy(),
            'labels': all_labels.numpy()
        }

    def train(self,
              train_loader,
              val_loader,
              test_loader,
              optimizer,
              scheduler=None,
              resume_checkpoint: Optional[str] = None) -> Tuple[float, int]:
        """完整训练流程"""

        # 加载检查点（如果存在）
        if resume_checkpoint and os.path.isfile(resume_checkpoint):
            print(f"=> 加载检查点 '{resume_checkpoint}'")
            checkpoint = torch.load(resume_checkpoint)
            self.best_score = checkpoint['best_score']
            self.model.load_state_dict(checkpoint['state_dict'])
            print(f"=> 加载完成 (score: {checkpoint['best_score']:.4f})")

        for epoch in range(self.args.num_epoch):
            print(f'\n=== Epoch {epoch+1}/{self.args.num_epoch} ===')

            # 训练
            train_metrics = self.train_epoch(train_loader, optimizer, epoch)

            # 验证
            val_metrics = self.validate(val_loader, phase='val')

            # 测试
            test_metrics = self.validate(test_loader, phase='test')

            # 记录指标
            self.metrics_history['train_auc'].append(train_metrics['auc'])
            self.metrics_history['train_acc'].append(train_metrics['acc'])
            self.metrics_history['train_loss'].append(train_metrics['loss'])
            self.metrics_history['val_auc'].append(val_metrics['auc'])
            self.metrics_history['val_acc'].append(val_metrics['acc'])
            self.metrics_history['val_loss'].append(val_metrics['loss'])
            self.metrics_history['test_auc'].append(test_metrics['auc'])
            self.metrics_history['test_acc'].append(test_metrics['acc'])
            self.metrics_history['test_loss'].append(test_metrics['loss'])

            # 打印epoch结果
            print(f'Train - AUC: {train_metrics["auc"]:.4f}, '
                  f'Acc: {train_metrics["acc"]:.4f}, Loss: {train_metrics["loss"]:.4f}')
            print(f'Val   - AUC: {val_metrics["auc"]:.4f}, '
                  f'Acc: {val_metrics["acc"]:.4f}, Loss: {val_metrics["loss"]:.4f}')
            print(f'Test  - AUC: {test_metrics["auc"]:.4f}, '
                  f'Acc: {test_metrics["acc"]:.4f}, Loss: {test_metrics["loss"]:.4f}')

            # 保存最佳模型
            if val_metrics['auc'] > 0.70 and val_metrics['auc'] > self.best_score * 0.95:
                self._save_checkpoint(
                    epoch=epoch,
                    state_dict=self.model.state_dict(),
                    score=test_metrics['auc'],
                    train_metrics=train_metrics,
                    val_metrics=val_metrics,
                    test_metrics=test_metrics
                )

            # 学习率调度
            if scheduler is not None:
                scheduler.step()

        # 绘制训练曲线
        self._plot_training_curves()

        return self.best_score, self.best_epoch

    def _save_checkpoint(self,
                        epoch: int,
                        state_dict: dict,
                        score: float,
                        train_metrics: dict,
                        val_metrics: dict,
                        test_metrics: dict):
        """保存模型检查点"""
        checkpoint = {
            'epoch': epoch,
            'state_dict': state_dict,
            'best_score': score,
            'train_metrics': train_metrics,
            'val_metrics': val_metrics,
            'test_metrics': test_metrics,
            'args': self.args
        }

        self.filename_best = os.path.join(
            self.results_dir,
            f'model_best_{score:.4f}_{epoch}.pth'
        )
        torch.save(checkpoint, self.filename_best)
        print(f'保存最佳模型: {self.filename_best}')

        # 保存特征为MAT文件
        self._save_features_mat(train_metrics, val_metrics, test_metrics)

    def _save_features_mat(self, train_metrics, val_metrics, test_metrics):
        """保存特征为MATLAB文件"""
        mat_data = {
            'train_features': train_metrics['features'],
            'train_labels': train_metrics['labels'],
            'val_features': val_metrics['features'],
            'val_labels': val_metrics['labels'],
            'test_features': test_metrics['features'],
            'test_labels': test_metrics['labels']
        }

        mat_path = os.path.join(
            self.results_dir,
            f'features_{train_metrics["auc"]:.3f}_{val_metrics["auc"]:.3f}_{test_metrics["auc"]:.3f}.mat'
        )
        scio.savemat(mat_path, mat_data)

    def _save_memory_bank(self):
        """保存记忆库状态"""
        memory_path = os.path.join(self.results_dir, 'cmta_memory.pth')
        if hasattr(self.model, 'path_know_memory') and hasattr(self.model, 'radio_know_memory'):
            torch.save({
                'path_know_memory': self.model.path_know_memory.cpu(),
                'radio_know_memory': self.model.radio_know_memory.cpu(),
                'patient_bank': [pb.cpu() if hasattr(pb, 'cpu') else pb
                               for pb in self.model.patient_bank]
            }, memory_path)

    def _plot_training_curves(self):
        """绘制训练曲线"""
        plot_training_metrics(
            metrics_dict=self.metrics_history,
            title='CMTA Training Metrics',
            output_path=os.path.join(self.results_dir, 'training_metrics.png')
        )


class CohortLoss(nn.Module):
    """队列损失函数"""

    def __init__(self, temperature: float = 2.0):
        super().__init__()
        self.temperature = temperature

    def forward(self, outputs, labels):
        loss = 0
        if 'cohort' in outputs:
            indiv, origs = outputs['decompose']
            cohort = outputs['cohort']

            # 内损失
            mask = torch.tensor([[1, 1], [0, 0], [1, 0], [0, 1]])
            if torch.cuda.is_available():
                mask = mask.cuda()
            indiv_know = indiv.view(4, 1, -1)
            orig = torch.cat(origs, dim=1).detach()
            sim = F.cosine_similarity(indiv_know, orig, dim=-1)
            intra_loss = torch.mean(torch.abs(sim) * (1 - mask) - mask * sim) + 1

            # 间损失
            pos_feat = []
            neg_feat = []

            for feat in cohort:
                if labels.item() == 1:
                    pos_feat.append(feat.detach())
                else:
                    neg_feat.append(feat.detach())

            if len(pos_feat) < 1 or len(neg_feat) < 1:
                inter_loss = 0
            else:
                pos_feat = torch.cat(pos_feat, dim=0)
                neg_feat = torch.cat(neg_feat, dim=0)

                neg_dis = torch.matmul(indiv_know.squeeze(1), neg_feat.T) / self.temperature
                pos_dis = torch.matmul(indiv_know.squeeze(1), pos_feat.T) / self.temperature

                inter_loss = -torch.log(
                    torch.exp(pos_dis).mean() / (torch.exp(pos_dis).mean() + torch.exp(neg_dis).mean() + 1e-10)
                )

            loss = intra_loss.mean() + inter_loss

        return loss


class FocalLoss(nn.Module):
    """Focal Loss用于处理类别不平衡"""

    def __init__(self, alpha: float = 0.24, gamma: float = 2, num_classes: int = 2):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.num_classes = num_classes

        if isinstance(alpha, list):
            self.alpha_t = torch.Tensor(alpha)
        else:
            self.alpha_t = torch.zeros(num_classes)
            self.alpha_t[0] += alpha
            self.alpha_t[1:] += (1 - alpha)

    def forward(self, inputs, targets):
        preds_softmax = F.softmax(inputs, dim=-1)
        self.alpha_t = self.alpha_t.to(preds_softmax.device)
        preds_softmax = preds_softmax.view(-1, preds_softmax.size(-1))
        preds_logsoft = torch.log(preds_softmax)

        preds_softmax = preds_softmax.gather(1, targets.view(-1, 1))
        preds_logsoft = preds_logsoft.gather(1, targets.view(-1, 1))
        alpha = self.alpha_t.gather(0, targets.view(-1))
        loss = -torch.mul(torch.pow((1 - preds_softmax), self.gamma), preds_logsoft)
        loss = torch.mul(alpha, loss.t())

        return loss.mean()
