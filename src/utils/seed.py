"""
随机种子设置工具

用于确保实验可重复性
"""

import random
import numpy as np
import torch


def set_seed(seed: int = 42) -> None:
    """
    设置所有相关库的随机种子
    
    Args:
        seed: 随机种子值，默认为42
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        # 确保CUDA操作的确定性
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    
    print(f"随机种子已设置为: {seed}")


def get_seed() -> int:
    """
    获取当前使用的随机种子
    
    Returns:
        当前的随机种子值
    """
    return 42  # 项目标准种子
