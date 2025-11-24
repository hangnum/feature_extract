"""
损失函数

实现多种损失函数用于处理类别不平衡
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional


class FocalLoss(nn.Module):
    """
    Focal Loss: 用于处理类别不平衡
    
    论文: Focal Loss for Dense Object Detection
    """
    
    def __init__(
        self,
        alpha: Optional[torch.Tensor] = None,
        gamma: float = 2.0,
        reduction: str = 'mean'
    ):
        """
        Args:
            alpha: 类别权重，形状为(num_classes,)
            gamma: 聚焦参数，默认2.0
            reduction: 'none' | 'mean' | 'sum'
        """
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction
    
    def forward(self, inputs: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """
        Args:
            inputs: 模型输出 logits，形状 (N, C)
            targets: 真实标签，形状 (N,)
        
        Returns:
            损失值
        """
        ce_loss = F.cross_entropy(inputs, targets, reduction='none')
        pt = torch.exp(-ce_loss)
        focal_loss = (1 - pt) ** self.gamma * ce_loss
        
        if self.alpha is not None:
            alpha_t = self.alpha[targets]
            focal_loss = alpha_t * focal_loss
        
        if self.reduction == 'mean':
            return focal_loss.mean()
        elif self.reduction == 'sum':
            return focal_loss.sum()
        else:
            return focal_loss


class AsymmetricLoss(nn.Module):
    """
    Asymmetric Loss: 对正负样本使用不同的权重
    
    适用于不平衡分类任务
    """
    
    def __init__(
        self,
        gamma_pos: float = 0.0,
        gamma_neg: float = 4.0,
        clip: float = 0.05,
        reduction: str = 'mean'
    ):
        """
        Args:
            gamma_pos: 正样本的gamma
            gamma_neg: 负样本的gamma
            clip: 概率裁剪值
            reduction: 'none' | 'mean' | 'sum'
        """
        super(AsymmetricLoss, self).__init__()
        self.gamma_pos = gamma_pos
        self.gamma_neg = gamma_neg
        self.clip = clip
        self.reduction = reduction
    
    def forward(self, inputs: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """
        Args:
            inputs: 模型输出 logits，形状 (N, C)
            targets: 真实标签，形状 (N,)
        
        Returns:
            损失值
        """
        # 转换为概率
        probs = torch.softmax(inputs, dim=1)
        
        # One-hot编码
        targets_one_hot = F.one_hot(targets, num_classes=inputs.size(1)).float()
        
        # 正样本和负样本
        pos_probs = probs * targets_one_hot
        neg_probs = probs * (1 - targets_one_hot)
        
        # 裁剪
        if self.clip is not None and self.clip > 0:
            neg_probs = (neg_probs + self.clip).clamp(max=1)
        
        # 计算损失
        pos_loss = -(pos_probs + 1e-7).log() * (1 - pos_probs) ** self.gamma_pos
        neg_loss = -(1 - neg_probs + 1e-7).log() * neg_probs ** self.gamma_neg
        
        loss = pos_loss + neg_loss
        loss = loss.sum(dim=1)
        
        if self.reduction == 'mean':
            return loss.mean()
        elif self.reduction == 'sum':
            return loss.sum()
        else:
            return loss


def get_loss_function(
    loss_type: str = 'ce',
    num_classes: int = 2,
    class_weights: Optional[torch.Tensor] = None,
    device: str = 'cuda'
) -> nn.Module:
    """
    获取损失函数
    
    Args:
        loss_type: 损失类型 ('ce', 'focal', 'asymmetric')
        num_classes: 类别数
        class_weights: 类别权重
        device: 设备
    
    Returns:
        损失函数
    """
    if loss_type == 'ce':
        # 交叉熵损失
        if class_weights is not None:
            class_weights = class_weights.to(device)
        criterion = nn.CrossEntropyLoss(weight=class_weights)
    
    elif loss_type == 'focal':
        # Focal Loss
        alpha = class_weights.to(device) if class_weights is not None else None
        criterion = FocalLoss(alpha=alpha, gamma=2.0)
    
    elif loss_type == 'asymmetric':
        # Asymmetric Loss
        criterion = AsymmetricLoss(gamma_pos=0.0, gamma_neg=4.0)
    
    else:
        raise ValueError(f"不支持的损失类型: {loss_type}")
    
    return criterion
