"""
可视化模块

支持绘制训练曲线、混淆矩阵、ROC曲线等
"""

import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from pathlib import Path
from sklearn.metrics import roc_curve, auc, precision_recall_curve
import torch


def plot_training_metrics(metrics_dict, title='Training Metrics', x_label='Epoch',
                        y_label='Value', output_path=None, figsize=(12, 8)):
    """
    绘制训练指标曲线

    Args:
        metrics_dict: 包含指标历史的字典
        title: 图表标题
        x_label: x轴标签
        y_label: y轴标签
        output_path: 保存路径
        figsize: 图表大小
    """
    plt.figure(figsize=figsize)

    epochs = range(1, len(metrics_dict.get('train_auc', [])) + 1)

    # 创建子图
    fig, axes = plt.subplots(2, 2, figsize=figsize)
    fig.suptitle(title, fontsize=16)

    # AUC曲线
    if 'train_auc' in metrics_dict and 'val_auc' in metrics_dict:
        axes[0, 0].plot(epochs, metrics_dict['train_auc'], 'b-', label='Train AUC')
        axes[0, 0].plot(epochs, metrics_dict['val_auc'], 'r-', label='Val AUC')
        if 'test_auc' in metrics_dict:
            axes[0, 0].plot(epochs, metrics_dict['test_auc'], 'g-', label='Test AUC')
        axes[0, 0].set_title('AUC over Epochs')
        axes[0, 0].set_xlabel('Epoch')
        axes[0, 0].set_ylabel('AUC')
        axes[0, 0].legend()
        axes[0, 0].grid(True)

    # 准确率曲线
    if 'train_acc' in metrics_dict and 'val_acc' in metrics_dict:
        axes[0, 1].plot(epochs, metrics_dict['train_acc'], 'b-', label='Train Acc')
        axes[0, 1].plot(epochs, metrics_dict['val_acc'], 'r-', label='Val Acc')
        if 'test_acc' in metrics_dict:
            axes[0, 1].plot(epochs, metrics_dict['test_acc'], 'g-', label='Test Acc')
        axes[0, 1].set_title('Accuracy over Epochs')
        axes[0, 1].set_xlabel('Epoch')
        axes[0, 1].set_ylabel('Accuracy')
        axes[0, 1].legend()
        axes[0, 1].grid(True)

    # 损失曲线
    if 'train_loss' in metrics_dict and 'val_loss' in metrics_dict:
        axes[1, 0].plot(epochs, metrics_dict['train_loss'], 'b-', label='Train Loss')
        axes[1, 0].plot(epochs, metrics_dict['val_loss'], 'r-', label='Val Loss')
        if 'test_loss' in metrics_dict:
            axes[1, 0].plot(epochs, metrics_dict['test_loss'], 'g-', label='Test Loss')
        axes[1, 0].set_title('Loss over Epochs')
        axes[1, 0].set_xlabel('Epoch')
        axes[1, 0].set_ylabel('Loss')
        axes[1, 0].legend()
        axes[1, 0].grid(True)

    # 清空最后一个子图或添加其他信息
    axes[1, 1].axis('off')
    if epochs:
        best_epoch_auc = np.argmax(metrics_dict.get('val_auc', [0])) + 1
        best_auc = max(metrics_dict.get('val_auc', [0]))
        axes[1, 1].text(0.1, 0.8, f'Best Epoch: {best_epoch_auc}', fontsize=12)
        axes[1, 1].text(0.1, 0.6, f'Best Val AUC: {best_auc:.4f}', fontsize=12)

    plt.tight_layout()

    if output_path:
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        print(f"训练曲线已保存到: {output_path}")

    plt.close()


def plot_confusion_matrix(cm, class_names=None, normalize=False, title='Confusion Matrix',
                         output_path=None, figsize=(8, 6)):
    """
    绘制混淆矩阵

    Args:
        cm: 混淆矩阵
        class_names: 类别名称
        normalize: 是否归一化
        title: 图表标题
        output_path: 保存路径
        figsize: 图表大小
    """
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        fmt = '.2f'
    else:
        fmt = 'd'

    plt.figure(figsize=figsize)
    sns.heatmap(cm, annot=True, fmt=fmt, cmap='Blues',
                xticklabels=class_names, yticklabels=class_names)
    plt.title(title)
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')

    if output_path:
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        print(f"混淆矩阵已保存到: {output_path}")

    plt.close()


def plot_roc_curve(y_true, y_scores, title='ROC Curve', output_path=None, figsize=(8, 6)):
    """
    绘制ROC曲线

    Args:
        y_true: 真实标签
        y_scores: 预测分数
        title: 图表标题
        output_path: 保存路径
        figsize: 图表大小
    """
    fpr, tpr, _ = roc_curve(y_true, y_scores)
    roc_auc = auc(fpr, tpr)

    plt.figure(figsize=figsize)
    plt.plot(fpr, tpr, color='darkorange', lw=2,
             label=f'ROC curve (AUC = {roc_auc:.2f})')
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title(title)
    plt.legend(loc="lower right")
    plt.grid(True)

    if output_path:
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        print(f"ROC曲线已保存到: {output_path}")

    plt.close()


def plot_precision_recall_curve(y_true, y_scores, title='Precision-Recall Curve',
                                output_path=None, figsize=(8, 6)):
    """
    绘制精确率-召回率曲线

    Args:
        y_true: 真实标签
        y_scores: 预测分数
        title: 图表标题
        output_path: 保存路径
        figsize: 图表大小
    """
    precision, recall, _ = precision_recall_curve(y_true, y_scores)
    pr_auc = auc(recall, precision)

    plt.figure(figsize=figsize)
    plt.plot(recall, precision, color='blue', lw=2,
             label=f'PR curve (AUC = {pr_auc:.2f})')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.title(title)
    plt.legend(loc="lower left")
    plt.grid(True)

    if output_path:
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        print(f"PR曲线已保存到: {output_path}")

    plt.close()


def plot_feature_distribution(features, labels, title='Feature Distribution',
                           output_path=None, figsize=(12, 8)):
    """
    绘制特征分布

    Args:
        features: 特征矩阵 [N, D]
        labels: 标签 [N]
        title: 图表标题
        output_path: 保存路径
        figsize: 图表大小
    """
    features = np.array(features)
    labels = np.array(labels)

    # 选择前几个维度进行可视化
    n_features_to_plot = min(16, features.shape[1])

    plt.figure(figsize=figsize)

    for i in range(n_features_to_plot):
        plt.subplot(4, 4, i + 1)

        for class_label in np.unique(labels):
            class_features = features[labels == class_label, i]
            plt.hist(class_features, alpha=0.6, label=f'Class {class_label}', bins=20)

        plt.title(f'Feature {i+1}')
        plt.xlabel('Value')
        plt.ylabel('Frequency')
        plt.legend()

    plt.suptitle(title)
    plt.tight_layout()

    if output_path:
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        print(f"特征分布图已保存到: {output_path}")

    plt.close()


def plot_attention_weights(attention_weights, title='Attention Weights',
                          output_path=None, figsize=(10, 6)):
    """
    绘制注意力权重热图

    Args:
        attention_weights: 注意力权重矩阵
        title: 图表标题
        output_path: 保存路径
        figsize: 图表大小
    """
    plt.figure(figsize=figsize)
    sns.heatmap(attention_weights, cmap='viridis', cbar=True)
    plt.title(title)
    plt.xlabel('Key Position')
    plt.ylabel('Query Position')

    if output_path:
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        print(f"注意力权重图已保存到: {output_path}")

    plt.close()


def save_training_log(metrics_history, output_path):
    """
    保存训练日志到文件

    Args:
        metrics_history: 指标历史字典
        output_path: 保存路径
    """
    with open(output_path, 'w') as f:
        f.write("Epoch,Train_AUC,Train_Acc,Train_Loss,Val_AUC,Val_Acc,Val_Loss\n")

        for epoch in range(len(metrics_history.get('train_auc', []))):
            train_auc = metrics_history.get('train_auc', [0])[epoch]
            train_acc = metrics_history.get('train_acc', [0])[epoch]
            train_loss = metrics_history.get('train_loss', [0])[epoch]
            val_auc = metrics_history.get('val_auc', [0])[epoch]
            val_acc = metrics_history.get('val_acc', [0])[epoch]
            val_loss = metrics_history.get('val_loss', [0])[epoch]

            f.write(f"{epoch+1},{train_auc:.4f},{train_acc:.4f},{train_loss:.4f},"
                   f"{val_auc:.4f},{val_acc:.4f},{val_loss:.4f}\n")

    print(f"训练日志已保存到: {output_path}")


def create_summary_report(metrics, config, output_path):
    """
    创建实验摘要报告

    Args:
        metrics: 评估指标字典
        config: 配置对象
        output_path: 保存路径
    """
    with open(output_path, 'w') as f:
        f.write("=== 实验摘要报告 ===\n\n")

        # 配置信息
        f.write("配置信息:\n")
        f.write(f"  模型: {config.model.name}\n")
        f.write(f"  批次大小: {config.training.batch_size}\n")
        f.write(f"  学习率: {config.training.learning_rate}\n")
        f.write(f"  训练轮数: {config.training.epochs}\n")
        f.write(f"  损失函数: {config.training.loss_type}\n")
        f.write(f"  优化器: {config.training.optimizer}\n")
        f.write(f"  设备: {config.training.device}\n\n")

        # 评估结果
        f.write("评估结果:\n")
        for key, value in metrics.items():
            if isinstance(value, (int, float)):
                f.write(f"  {key}: {value:.4f}\n")
            else:
                f.write(f"  {key}: {value}\n")

    print(f"摘要报告已保存到: {output_path}")