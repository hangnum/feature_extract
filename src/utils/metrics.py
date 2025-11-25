"""
评估指标计算模块

支持计算各种医学影像分类任务的评估指标
"""

import numpy as np
from sklearn.metrics import (
    roc_auc_score,
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    classification_report,
)
import torch


class AverageMeter:
    """滑动统计工具，记录当前值、累计和与均值（训练/验证均可复用）。"""

    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0
        self.acc = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


def calculate_metrics(y_true, y_pred, y_prob=None):
    """
    计算分类评估指标

    Args:
        y_true: 真实标签 [N]
        y_pred: 预测标签 [N]
        y_prob: 预测概率 [N] (可选，用于计算AUC)

    Returns:
        Dict: 包含各种指标的字典
    """
    metrics = {}

    # 基本指标
    metrics['accuracy'] = accuracy_score(y_true, y_pred)
    metrics['precision'] = precision_score(y_true, y_pred, average='weighted', zero_division=0)
    metrics['recall'] = recall_score(y_true, y_pred, average='weighted', zero_division=0)
    metrics['f1'] = f1_score(y_true, y_pred, average='weighted', zero_division=0)

    # AUC（如果提供了概率）
    if y_prob is not None:
        try:
            if len(np.unique(y_true)) == 2:
                metrics['auc'] = roc_auc_score(y_true, y_prob)
            else:
                metrics['auc'] = 0.5  # 多类情况下的默认值
        except Exception:
            metrics['auc'] = 0.5
    else:
        metrics['auc'] = 0.5

    # 混淆矩阵
    metrics['confusion_matrix'] = confusion_matrix(y_true, y_pred).tolist()

    return metrics


def calculate_multiclass_metrics(y_true, y_pred, y_prob=None, class_names=None):
    """
    计算多分类任务的详细指标

    Args:
        y_true: 真实标签 [N]
        y_pred: 预测标签 [N]
        y_prob: 预测概率 [N, num_classes] (可选)
        class_names: 类别名称列表 (可选)

    Returns:
        Dict: 包含各种指标的字典
    """
    metrics = calculate_metrics(y_true, y_pred, y_prob)

    # 添加分类报告
    if class_names:
        metrics['classification_report'] = classification_report(
            y_true, y_pred, target_names=class_names, output_dict=True
        )
    else:
        metrics['classification_report'] = classification_report(
            y_true, y_pred, output_dict=True
        )

    return metrics


def calculate_confidence_intervals(y_true, y_scores, n_bootstraps=1000, confidence_level=0.95):
    """
    通过bootstrap计算置信区间

    Args:
        y_true: 真实标签 [N]
        y_scores: 预测分数 [N]
        n_bootstraps: bootstrap采样次数
        confidence_level: 置信水平

    Returns:
        Dict: 包含置信区间的字典
    """
    np.random.seed(42)
    n = len(y_true)
    scores = []

    for _ in range(n_bootstraps):
        # 有放回采样
        indices = np.random.choice(n, n, replace=True)
        if len(np.unique(y_true[indices])) < 2:
            continue

        score = roc_auc_score(y_true[indices], y_scores[indices])
        scores.append(score)

    scores = np.array(scores)
    alpha = 1 - confidence_level
    lower_percentile = (alpha / 2) * 100
    upper_percentile = (1 - alpha / 2) * 100

    ci_lower = np.percentile(scores, lower_percentile)
    ci_upper = np.percentile(scores, upper_percentile)
    ci_mean = scores.mean()

    return {
        'mean': ci_mean,
        'lower': ci_lower,
        'upper': ci_upper,
        'std': scores.std()
    }


def calculate_sensitivity_specificity(y_true, y_pred):
    """
    计算敏感性和特异性

    Args:
        y_true: 真实标签 [N]
        y_pred: 预测标签 [N]

    Returns:
        Dict: 包含敏感性和特异性的字典
    """
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()

    sensitivity = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    specificity = tn / (tn + fp) if (tn + fp) > 0 else 0.0

    return {
        'sensitivity': sensitivity,
        'specificity': specificity
    }


def calculate_threshold_metrics(y_true, y_prob, thresholds=None):
    """
    在不同阈值下计算指标

    Args:
        y_true: 真实标签 [N]
        y_prob: 预测概率 [N]
        thresholds: 阈值列表，如果为None则使用默认值

    Returns:
        Dict: 包含不同阈值下指标的字典
    """
    if thresholds is None:
        thresholds = np.arange(0.1, 1.0, 0.1)

    results = {}

    for threshold in thresholds:
        y_pred = (y_prob >= threshold).astype(int)

        # 计算指标
        tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()

        sensitivity = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        specificity = tn / (tn + fp) if (tn + fp) > 0 else 0.0
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        f1 = 2 * (precision * sensitivity) / (precision + sensitivity) if (precision + sensitivity) > 0 else 0.0

        results[threshold] = {
            'threshold': threshold,
            'sensitivity': sensitivity,
            'specificity': specificity,
            'precision': precision,
            'f1': f1,
            'tp': tp,
            'tn': tn,
            'fp': fp,
            'fn': fn
        }

    return results
