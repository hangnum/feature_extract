from typing import Dict, List, Optional
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np


def plot_metrics(
        metrics_dict: Dict[str, List[float]],
        title: str = 'Metrics History',
        x_label: str = 'Epoch',
        y_label: str = 'Value',
        output_path: Optional[Path] = None,
        font_name: Optional[str] = 'DejaVu Sans'  # 新增: 字体名称参数
) -> None:
    """根据一个包含多个指标的字典绘制可视化图表。

    这种设计允许动态传入任意数量的指标进行绘图，更具扩展性。

    Args:
      metrics_dict: 一个字典，键是指标名称(str)，值是数据列表(List[float])。
                    例如: {'Train Loss': [0.5, 0.4], 'Val Acc': [0.8, 0.9]}
      title: 图表的标题。
      x_label: X轴的标签。
      y_label: Y轴的标签。
      output_path: 可选，图表保存路径。如果为 None，则显示图表。
      font_name: 可选，用于图表的字体名称。默认为 'DejaVu Sans'，
                 在Debian系统中有很好的兼容性。若设为 None，则使用matplotlib默认字体。
    """
    # 新增: 使用 rc_context 临时设置绘图字体，避免影响全局配置
    font_config = {'font.family': 'sans-serif', 'font.sans-serif': [font_name]}

    # 如果 font_name 为 None，则使用空的 context，即 matplotlib 默认配置
    context = plt.rc_context(font_config if font_name else {})

    with context:
        fig, ax = plt.subplots(figsize=(12, 8))

        num_points = 0
        # 动态绘制字典中的每一项指标
        for label, data in metrics_dict.items():
            if not num_points:
                num_points = len(data)

            # 使用 numpy 数组以获得更好的性能
            y_data = np.array(data)
            x_data = np.arange(1, len(y_data) + 1)

            ax.plot(x_data, y_data, marker='o', linestyle='-', label=label)

        # 设置图表属性
        ax.set_title(title, fontsize=16)
        ax.set_xlabel(x_label, fontsize=12)
        ax.set_ylabel(y_label, fontsize=12)
        ax.legend(fontsize=10)
        ax.grid(True, linestyle='--', alpha=0.6)

    # 保存或显示 (移出 with 块，但 fig 对象已完成配置)
    if output_path:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(output_path, dpi=300, bbox_inches='tight')
        print(f"指标图表已保存至: {output_path}")
    else:
        plt.show()

    # 释放内存
    plt.close(fig)