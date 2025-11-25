"""
Progressive Information Bottleneck (PIB) 模块

用于无监督特征选择和信息压缩
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple, Optional


class PIB(nn.Module):
    """
    Progressive Information Bottleneck for unsupervised feature selection

    该模块通过以下步骤实现特征选择：
    1. 特征编码：将输入特征映射到编码空间
    2. 代理学习：学习可学习的代理中心（proxy centers）
    3. 特征选择：基于与代理中心的相似度选择top-k特征
    4. 噪声注入：训练时注入高斯噪声增强鲁棒性
    """
    def __init__(
        self,
        x_dim: int,              # 输入特征维度
        z_dim: int = 256,        # 编码空间维度
        num_proxies: int = 8,    # 代理数量（自动学习）
        topk: int = 128,         # 选择的top-k特征
        noise_std: float = 0.1,  # 高斯噪声标准差
        sample_num: int = 5      # 噪声采样次数
    ):
        super(PIB, self).__init__()

        self.z_dim = z_dim
        self.num_proxies = num_proxies
        self.topk = topk
        self.noise_std = noise_std
        self.sample_num = sample_num

        # 编码器网络
        self.encoder = nn.Sequential(
            nn.Linear(x_dim, z_dim * 2),
            nn.GELU(),
            nn.LayerNorm(z_dim * 2),
            nn.Linear(z_dim * 2, z_dim),
            nn.Dropout(0.2)
        )

        # 可学习的代理中心（自动聚类）
        self.proxies = nn.Parameter(torch.randn(num_proxies, z_dim))
        nn.init.kaiming_normal_(self.proxies, mode='fan_out')

        # 特征重要性预测器
        self.importance_predictor = nn.Sequential(
            nn.Linear(z_dim, z_dim // 2),
            nn.ReLU(),
            nn.Linear(z_dim // 2, 1),
            nn.Sigmoid()
        )

    def add_noise(self, z: torch.Tensor) -> torch.Tensor:
        """注入高斯噪声的增强方法"""
        if self.training and self.noise_std > 0:
            noise = torch.randn_like(z) * self.noise_std
            return z + noise
        return z

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        前向传播

        Args:
            x: [B, N, x_dim] 特征组

        Returns:
            selected_features: [B, topk, z_dim] 选择的特征
            proxy_weights: [B, num_proxies] 代理权重分布
            topk_indices: [B, topk] 选择的特征索引
            encoded_features: [B, N, z_dim] 编码后的所有特征
        """
        B, N, _ = x.shape

        # 1. 特征编码
        z = self.encoder(x)  # [B, N, z_dim]
        z = self.add_noise(z)

        # 2. 计算特征-代理相似度
        z_norm = F.normalize(z, dim=-1)
        proxies_norm = F.normalize(self.proxies, dim=-1)  # [num_proxies, z_dim]
        sim_matrix = torch.matmul(z_norm, proxies_norm.T)  # [B, N, num_proxies]

        # 3. 动态重要性加权
        importance = self.importance_predictor(z)  # [B, N, 1]
        weighted_sim = sim_matrix * importance  # [B, N, num_proxies]

        # 4. Top-K特征选择
        _, topk_indices = torch.topk(
            weighted_sim.mean(dim=-1),  # 跨代理平均重要性
            k=min(self.topk, N),
            dim=1
        )

        # 5. 收集选择的特征
        selected_features = torch.gather(
            z,
            1,
            topk_indices.unsqueeze(-1).expand(-1, -1, self.z_dim)
        )  # [B, topk, z_dim]

        # 6. 代理权重分布（可解释性分析）
        proxy_weights = F.softmax(weighted_sim.mean(dim=1), dim=-1)  # [B, num_proxies]

        return selected_features, proxy_weights, topk_indices, z

    def get_proxy_centers(self) -> torch.Tensor:
        """获取当前代理中心（用于可视化）"""
        return self.proxies.detach().cpu().numpy()

    def get_feature_importance(self, x: torch.Tensor) -> torch.Tensor:
        """
        获取特征重要性分数（用于解释性分析）

        Args:
            x: [B, N, x_dim] 输入特征

        Returns:
            importance: [B, N, 1] 特征重要性分数
        """
        with torch.no_grad():
            z = self.encoder(x)
            importance = self.importance_predictor(z)
        return importance