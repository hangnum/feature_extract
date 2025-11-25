"""
知识分解模块

实现多模态特征的知识分解和交互估计
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple
from .fusion_utils import MLP_Block


class InteractionEstimator(nn.Module):
    """
    交互估计器：学习两个模态之间的交互知识
    """
    def __init__(self, feat_len: int = 64, dim: int = 256):
        super().__init__()
        # 特征对齐网络
        self.feat1_fc = MLP_Block(dim, dim)
        self.feat2_fc = MLP_Block(dim, dim)

        # 注意力权重生成器
        self.feat1_atten = nn.Linear(dim, 1)
        self.feat2_atten = nn.Linear(dim, 1)

    def forward(self, feat1: torch.Tensor, feat2: torch.Tensor) -> torch.Tensor:
        """
        计算两个模态特征之间的交互

        Args:
            feat1: [B, N, dim] 第一个模态特征
            feat2: [B, M, dim] 第二个模态特征

        Returns:
            interaction: [B, N, M, dim] 交互特征
        """
        # 特征对齐
        feat1_aligned = self.feat1_fc(feat1)  # [B, N, dim]
        feat2_aligned = self.feat2_fc(feat2)  # [B, M, dim]

        # 计算交互注意力
        interaction_matrix = feat1_aligned.unsqueeze(3) * feat2_aligned.unsqueeze(2)  # [B, N, M, dim]

        # 生成注意力权重
        feat1_att = torch.sigmoid(self.feat1_atten(interaction_matrix)).squeeze(-1)  # [B, N, M]
        feat2_att = torch.sigmoid(self.feat2_atten(interaction_matrix.permute(0, 1, 3, 2))).squeeze(-1)  # [B, M, N]

        # 加权交互
        interaction = feat2_aligned.unsqueeze(1) * feat2_att.unsqueeze(-1) + \
                     feat1_aligned.unsqueeze(2) * feat1_att.unsqueeze(-1)

        return interaction


class SpecificityEstimator(nn.Module):
    """
    特异性估计器：学习模态特异性知识
    """
    def __init__(self, feat_len: int = 64, dim: int = 256):
        super().__init__()
        self.conv = MLP_Block(dim, dim)

    def forward(self, feat: torch.Tensor) -> torch.Tensor:
        """
        提取模态特异性特征

        Args:
            feat: [B, N, dim] 输入特征

        Returns:
            spec_feat: [B, N, dim] 特异性特征
        """
        return self.conv(feat)


class KnowledgeDecomposition(nn.Module):
    """
    知识分解模块

    将多模态特征分解为：
    1. 共性知识（Common Knowledge）：模态间共享的知识
    2. 协同知识（Synergy Knowledge）：模态间互补增强的知识
    3. 特异性知识（Specific Knowledge）：每个模态独有的知识
    """
    def __init__(self, feat_len: int = 64, dim: int = 256, use_specificity: bool = False):
        super().__init__()
        self.use_specificity = use_specificity

        # 交互编码器
        self.common_encoder = InteractionEstimator(feat_len, dim)
        self.synergy_encoder = InteractionEstimator(feat_len, dim)

        # 特异性编码器（可选）
        if use_specificity:
            self.feat1_spec = SpecificityEstimator(feat_len, dim)
            self.feat2_spec = SpecificityEstimator(feat_len, dim)

    def forward(self, feat1: torch.Tensor, feat2: torch.Tensor) -> Tuple[torch.Tensor, ...]:
        """
        知识分解前向传播

        Args:
            feat1: [B, N, dim] 第一个模态特征
            feat2: [B, M, dim] 第二个模态特征

        Returns:
            common: [B, N, M, dim] 共性知识
            synergy: [B, N, M, dim] 协同知识
            feat1_spec: [B, N, dim] 第一个模态特异性（可选）
            feat2_spec: [B, M, dim] 第二个模态特异性（可选）
        """
        # 计算共性知识（模态间共同模式）
        common = self.common_encoder(feat1, feat2)

        # 计算协同知识（模态间互补增强）
        synergy = self.synergy_encoder(feat1, feat2)

        if self.use_specificity:
            # 计算模态特异性知识
            feat1_spec = self.feat1_spec(feat1)
            feat2_spec = self.feat2_spec(feat2)
            return common, synergy, feat1_spec, feat2_spec

        return common, synergy


class HungarianMatching(nn.Module):
    """
    匈牙利匹配算法：用于聚类中心对齐
    """
    def forward(self, centers: torch.Tensor, priors: torch.Tensor) -> torch.Tensor:
        """
        使用匈牙利匹配算法对齐聚类中心

        Args:
            centers: [num_centers, dim] 待对齐的中心
            priors: [num_centers, dim] 参考中心

        Returns:
            aligned_centers: [num_centers, dim] 对齐后的中心
        """
        from scipy.optimize import linear_sum_assignment

        cost = torch.cdist(centers, priors, p=1).detach().cpu().numpy()
        row_indices, col_indices = linear_sum_assignment(cost)

        # 创建one-hot映射矩阵
        one_hot_targets = F.one_hot(torch.tensor(col_indices), centers.shape[0]).float().to(centers.device)
        aligned_centers = torch.mm(one_hot_targets.T, centers)

        return aligned_centers


def hungarian_matching(centers: torch.Tensor, priors: torch.Tensor) -> torch.Tensor:
    """
    函数形式的匈牙利匹配，便于调用
    """
    from scipy.optimize import linear_sum_assignment

    cost = torch.cdist(centers, priors, p=1).detach().cpu()
    row_indices, col_indices = linear_sum_assignment(cost)

    one_hot_targets = F.one_hot(torch.tensor(col_indices), centers.shape[0]).float().cuda()
    aligned_centers = torch.mm(one_hot_targets.T, centers)

    return aligned_centers