"""
数据增强和预处理

定义训练集和验证集的数据变换
"""

from torchvision import transforms
from typing import Tuple


def get_transforms(
    image_size: int = 224,
    normalize_mean: float = 0.5,
    normalize_std: float = 0.5,
    horizontal_flip: bool = True,
    rotation_degrees: int = 15,
    is_training: bool = True
) -> transforms.Compose:
    """
    获取数据增强变换
    
    Args:
        image_size: 图像大小
        normalize_mean: 归一化均值
        normalize_std: 归一化标准差
        horizontal_flip: 是否水平翻转
        rotation_degrees: 旋转角度
        is_training: 是否为训练模式
    
    Returns:
        数据变换组合
    """
    transform_list = []
    
    # 调整大小（如果需要）
    transform_list.append(transforms.Resize((image_size, image_size)))
    
    if is_training:
        # 训练集增强
        if horizontal_flip:
            transform_list.append(transforms.RandomHorizontalFlip(p=0.5))
        
        if rotation_degrees > 0:
            transform_list.append(
                transforms.RandomRotation(degrees=rotation_degrees)
            )
        
        # 可选：添加其他增强
        # transform_list.append(transforms.RandomVerticalFlip(p=0.3))
    
    # 转为Tensor
    transform_list.append(transforms.ToTensor())
    
    # 归一化
    transform_list.append(
        transforms.Normalize(mean=[normalize_mean], std=[normalize_std])
    )
    
    return transforms.Compose(transform_list)


def get_train_transform(
    image_size: int = 224,
    normalize_mean: float = 0.5,
    normalize_std: float = 0.5,
    horizontal_flip: bool = True,
    rotation_degrees: int = 15
) -> transforms.Compose:
    """
    获取训练集数据变换
    
    Args:
        image_size: 图像大小
        normalize_mean: 归一化均值
        normalize_std: 归一化标准差
        horizontal_flip: 是否水平翻转
        rotation_degrees: 旋转角度
    
    Returns:
        训练集数据变换
    """
    return get_transforms(
        image_size=image_size,
        normalize_mean=normalize_mean,
        normalize_std=normalize_std,
        horizontal_flip=horizontal_flip,
        rotation_degrees=rotation_degrees,
        is_training=True
    )


def get_val_transform(
    image_size: int = 224,
    normalize_mean: float = 0.5,
    normalize_std: float = 0.5
) -> transforms.Compose:
    """
    获取验证/测试集数据变换
    
    Args:
        image_size: 图像大小
        normalize_mean: 归一化均值
        normalize_std: 归一化标准差
    
    Returns:
        验证/测试集数据变换
    """
    return get_transforms(
        image_size=image_size,
        normalize_mean=normalize_mean,
        normalize_std=normalize_std,
        is_training=False
    )
