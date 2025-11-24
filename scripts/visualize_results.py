"""
可视化训练结果

生成训练历史的折线图
"""

import sys
from pathlib import Path

# 添加项目根目录到Python路径
project_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(project_root))

import argparse
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

# 设置中文字体
plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei']
plt.rcParams['axes.unicode_minus'] = False

def plot_training_history(history_csv: str, output_dir: str):
    """
    绘制训练历史曲线
    
    Args:
        history_csv: 训练历史CSV文件路径
        output_dir: 输出目录
    """
    # 读取数据
    df = pd.read_csv(history_csv)
    
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    # 设置样式
    sns.set_style("whitegrid")
    
    # 1. Loss曲线
    plt.figure(figsize=(10, 6))
    plt.plot(df['epoch'], df['train_loss'], label='训练集Loss', marker='o')
    plt.plot(df['epoch'], df['val_loss'], label='验证集Loss', marker='s')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('训练和验证Loss曲线')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(output_path / 'loss_curve.png', dpi=300)
    plt.close()
    
    # 2. AUC曲线
    plt.figure(figsize=(10, 6))
    plt.plot(df['epoch'], df['val_auc'], label='验证集AUC', marker='o', color='green')
    plt.xlabel('Epoch')
    plt.ylabel('AUC')
    plt.title('验证集AUC曲线')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(output_path / 'auc_curve.png', dpi=300)
    plt.close()
    
    # 3. 准确率曲线
    plt.figure(figsize=(10, 6))
    plt.plot(df['epoch'], df['val_acc'], label='验证集准确率', marker='o', color='blue')
    plt.xlabel('Epoch')
    plt.ylabel('准确率')
    plt.title('验证集准确率曲线')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(output_path / 'accuracy_curve.png', dpi=300)
    plt.close()
    
    # 4. 敏感性和特异性
    plt.figure(figsize=(10, 6))
    plt.plot(df['epoch'], df['val_sensitivity'], label='敏感性', marker='o')
    plt.plot(df['epoch'], df['val_specificity'], label='特异性', marker='s')
    plt.xlabel('Epoch')
    plt.ylabel('值')
    plt.title('验证集敏感性和特异性曲线')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(output_path / 'sensitivity_specificity_curve.png', dpi=300)
    plt.close()
    
    # 5. 综合指标对比
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    
    # Loss
    axes[0, 0].plot(df['epoch'], df['train_loss'], label='训练集', marker='o')
    axes[0, 0].plot(df['epoch'], df['val_loss'], label='验证集', marker='s')
    axes[0, 0].set_xlabel('Epoch')
    axes[0, 0].set_ylabel('Loss')
    axes[0, 0].set_title('Loss曲线')
    axes[0, 0].legend()
    axes[0, 0].grid(True, alpha=0.3)
    
    # AUC
    axes[0, 1].plot(df['epoch'], df['val_auc'], marker='o', color='green')
    axes[0, 1].set_xlabel('Epoch')
    axes[0, 1].set_ylabel('AUC')
    axes[0, 1].set_title('AUC曲线')
    axes[0, 1].grid(True, alpha=0.3)
    
    # 准确率
    axes[1, 0].plot(df['epoch'], df['val_acc'], marker='o', color='blue')
    axes[1, 0].set_xlabel('Epoch')
    axes[1, 0].set_ylabel('准确率')
    axes[1, 0].set_title('准确率曲线')
    axes[1, 0].grid(True, alpha=0.3)
    
    # 敏感性和特异性
    axes[1, 1].plot(df['epoch'], df['val_sensitivity'], label='敏感性', marker='o')
    axes[1, 1].plot(df['epoch'], df['val_specificity'], label='特异性', marker='s')
    axes[1, 1].set_xlabel('Epoch')
    axes[1, 1].set_ylabel('值')
    axes[1, 1].set_title('敏感性和特异性曲线')
    axes[1, 1].legend()
    axes[1, 1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(output_path / 'all_metrics.png', dpi=300)
    plt.close()
    
    print(f"可视化结果已保存至: {output_path}")
    print("\n生成的图片:")
    print(f"  - loss_curve.png")
    print(f"  - auc_curve.png")
    print(f"  - accuracy_curve.png")
    print(f"  - sensitivity_specificity_curve.png")
    print(f"  - all_metrics.png")
    
    # 打印最佳结果
    best_epoch = df.loc[df['val_auc'].idxmax()]
    print(f"\n最佳结果 (Epoch {int(best_epoch['epoch'])}):")
    print(f"  AUC: {best_epoch['val_auc']:.4f}")
    print(f"  准确率: {best_epoch['val_acc']:.4f}")
    print(f"  敏感性: {best_epoch['val_sensitivity']:.4f}")
    print(f"  特异性: {best_epoch['val_specificity']:.4f}")


def main():
    parser = argparse.ArgumentParser(description='可视化训练结果')
    parser.add_argument('--history_csv', type=str, required=True, help='训练历史CSV文件路径')
    parser.add_argument('--output_dir', type=str, default=None, help='输出目录')
    
    args = parser.parse_args()
    
    if args.output_dir is None:
        # 默认保存在同一目录的visualizations文件夹
        history_path = Path(args.history_csv)
        args.output_dir = str(history_path.parent.parent.parent / 'visualizations' / history_path.parent.name)
    
    plot_training_history(args.history_csv, args.output_dir)


if __name__ == '__main__':
    main()
