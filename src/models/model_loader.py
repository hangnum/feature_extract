"""
模型加载器

支持加载预训练模型并修改为单通道灰度图输入
"""

import torch
import torch.nn as nn
import torchvision.models as models
import timm
import logging
from typing import Optional

logger = logging.getLogger("feature_extract")


def modify_first_conv_for_grayscale(model: nn.Module, model_name: str) -> nn.Module:
    """
    修改模型第一层卷积以适配单通道灰度图
    
    Args:
        model: 原始模型
        model_name: 模型名称
    
    Returns:
        修改后的模型
    """
    if 'resnet' in model_name.lower():
        # ResNet系列
        old_conv = model.conv1
        
        # 创建新的单通道卷积层
        new_conv = nn.Conv2d(
            in_channels=1,
            out_channels=old_conv.out_channels,
            kernel_size=old_conv.kernel_size,
            stride=old_conv.stride,
            padding=old_conv.padding,
            bias=old_conv.bias is not None
        )
        
        # 复制权重：取RGB三通道权重的平均值
        with torch.no_grad():
            new_conv.weight[:, 0, :, :] = old_conv.weight.mean(dim=1)
            if old_conv.bias is not None:
                new_conv.bias = old_conv.bias
        
        model.conv1 = new_conv
        logger.info(f"已将 {model_name} 的第一层卷积修改为单通道")
        
    elif 'swin' in model_name.lower():
        # Swin Transformer
        # 通常在patch_embed层
        if hasattr(model, 'patch_embed'):
            old_proj = model.patch_embed.proj
            
            new_proj = nn.Conv2d(
                in_channels=1,
                out_channels=old_proj.out_channels,
                kernel_size=old_proj.kernel_size,
                stride=old_proj.stride,
                padding=old_proj.padding,
                bias=old_proj.bias is not None
            )
            
            with torch.no_grad():
                new_proj.weight[:, 0, :, :] = old_proj.weight.mean(dim=1)
                if old_proj.bias is not None:
                    new_proj.bias = old_proj.bias
            
            model.patch_embed.proj = new_proj
            logger.info(f"已将 {model_name} 的patch_embed修改为单通道")
    
    return model


def freeze_model_stages(model: nn.Module, model_name: str, freeze_stages: int = 0) -> nn.Module:
    """
    冻结模型的前几个stage
    
    Args:
        model: 模型
        model_name: 模型名称
        freeze_stages: 冻结的stage数量
    
    Returns:
        冻结后的模型
    """
    if freeze_stages == 0:
        logger.info("不冻结任何层")
        return model
    
    if 'resnet' in model_name.lower():
        # ResNet有4个layer（stage）
        stages_to_freeze = []
        if freeze_stages >= 1:
            stages_to_freeze.append('conv1')
            stages_to_freeze.append('bn1')
        if freeze_stages >= 2:
            stages_to_freeze.append('layer1')
        if freeze_stages >= 3:
            stages_to_freeze.append('layer2')
        if freeze_stages >= 4:
            stages_to_freeze.append('layer3')
        
        for name, param in model.named_parameters():
            for stage in stages_to_freeze:
                if name.startswith(stage):
                    param.requires_grad = False
                    break
        
        logger.info(f"已冻结 {model_name} 的前 {freeze_stages} 个stage: {stages_to_freeze}")
    
    elif 'swin' in model_name.lower():
        # Swin Transformer的层级结构不同
        # 简单策略：冻结patch_embed和前几个layers
        if freeze_stages >= 1 and hasattr(model, 'patch_embed'):
            for param in model.patch_embed.parameters():
                param.requires_grad = False
        
        if hasattr(model, 'layers'):
            for i in range(min(freeze_stages - 1, len(model.layers))):
                for param in model.layers[i].parameters():
                    param.requires_grad = False
        
        logger.info(f"已冻结 {model_name} 的前 {freeze_stages} 个stage")
    
    return model


def load_model(
    model_name: str,
    num_classes: int = 2,
    pretrained: bool = True,
    freeze_stages: int = 0
) -> nn.Module:
    """
    加载预训练模型并修改为适配灰度图和目标类别数
    
    Args:
        model_name: 模型名称 (resnet18, resnet50, swin_t等)
        num_classes: 分类类别数
        pretrained: 是否使用预训练权重
        freeze_stages: 冻结的stage数量（ResNet50建议3）
    
    Returns:
        修改后的模型
    """
    logger.info(f"加载模型: {model_name}")
    logger.info(f"  预训练: {pretrained}")
    logger.info(f"  类别数: {num_classes}")
    logger.info(f"  冻结stage数: {freeze_stages}")
    
    model = None
    
    # 加载模型
    if model_name == 'resnet18':
        if pretrained:
            model = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1)
        else:
            model = models.resnet18(weights=None)
        in_features = model.fc.in_features
        model.fc = nn.Linear(in_features, num_classes)
        
    elif model_name == 'resnet50':
        if pretrained:
            model = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V1)
        else:
            model = models.resnet50(weights=None)
        in_features = model.fc.in_features
        model.fc = nn.Linear(in_features, num_classes)
        
    elif model_name == 'swin_t':
        model = timm.create_model(
            'swin_tiny_patch4_window7_224',
            pretrained=pretrained,
            num_classes=num_classes
        )
    
    else:
        raise ValueError(f"不支持的模型: {model_name}")
    
    # 修改第一层卷积为单通道
    model = modify_first_conv_for_grayscale(model, model_name)
    
    # 冻结前几个stage
    if freeze_stages > 0:
        model = freeze_model_stages(model, model_name, freeze_stages)
    
    return model


def get_feature_extractor(
    model: nn.Module,
    model_name: str
) -> nn.Module:
    """
    移除分类层，保留特征提取部分
    
    Args:
        model: 完整模型
        model_name: 模型名称
    
    Returns:
        特征提取器
    """
    if 'resnet' in model_name.lower():
        # ResNet: 移除fc层
        feature_extractor = nn.Sequential(*list(model.children())[:-1])
        
    elif 'swin' in model_name.lower():
        # Swin Transformer: 移除head
        if hasattr(model, 'head'):
            model.head = nn.Identity()
        feature_extractor = model
    
    else:
        raise ValueError(f"不支持的模型: {model_name}")
    
    logger.info(f"创建特征提取器: {model_name}")
    
    return feature_extractor
