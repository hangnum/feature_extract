"""
评估指标

计算分类性能指标
"""

import numpy as np
import torch
from sklearn.metrics import (
    accuracy_score,
    roc_auc_score,
    confusion_matrix,
    classification_report
)
from typing import Tuple, Dict


def calculate_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    y_prob: np.ndarray
) -> Dict[str, float]:
    """
    计算分类指标
    
    Args:
        y_true: 真实标签
        y_pred: 预测标签
        y_prob: 预测概率（针对正类）
    
    Returns:
        指标字典
    """
    # 准确率
    acc = accuracy_score(y_true, y_pred)
    
    # AUC
    try:
        auc = roc_auc_score(y_true, y_prob)
    except:
        auc = 0.0
    
    # 混淆矩阵
    cm = confusion_matrix(y_true, y_pred)
    
    # 敏感性（召回率）和特异性
    if cm.shape == (2, 2):
        tn, fp, fn, tp = cm.ravel()
        sensitivity = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        specificity = tn / (tn + fp) if (tn + fp) > 0 else 0.0
    else:
        sensitivity = 0.0
        specificity = 0.0
    
    metrics = {
        'accuracy': acc,
        'auc': auc,
        'sensitivity': sensitivity,
        'specificity': specificity
    }
    
    return metrics


def evaluate_model(
    model: torch.nn.Module,
    dataloader: torch.utils.data.DataLoader,
    criterion: torch.nn.Module,
    device: str = 'cuda'
) -> Tuple[float, Dict[str, float]]:
    """
    评估模型
    
    Args:
        model: 模型
        dataloader: 数据加载器
        criterion: 损失函数
        device: 设备
    
    Returns:
        (平均损失, 指标字典)
    """
    model.eval()
    
    all_labels = []
    all_preds = []
    all_probs = []
    total_loss = 0.0
    
    with torch.no_grad():
        for batch_idx, (images, labels) in enumerate(dataloader):
            images = images.to(device)
            labels = labels.to(device)
            
            # 前向传播
            outputs = model(images)
            loss = criterion(outputs, labels)
            
            total_loss += loss.item()
            
            # 预测
            probs = torch.softmax(outputs, dim=1)
            preds = torch.argmax(outputs, dim=1)
            
            # 收集结果
            all_labels.extend(labels.cpu().numpy())
            all_preds.extend(preds.cpu().numpy())
            all_probs.extend(probs[:, 1].cpu().numpy())  # 正类概率
    
    # 计算指标
    avg_loss = total_loss / len(dataloader)
    
    metrics = calculate_metrics(
        y_true=np.array(all_labels),
        y_pred=np.array(all_preds),
        y_prob=np.array(all_probs)
    )
    
    return avg_loss, metrics
