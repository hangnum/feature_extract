"""
简化的K-means实现，用于CMTA模型中的聚类操作
"""

import torch
import numpy as np
from typing import Tuple, Optional


def kmeans(X: torch.Tensor,
           num_clusters: int,
           cluster_centers: Optional[torch.Tensor] = None,
           distance: str = 'euclidean',
           device: torch.device = torch.device('cpu'),
           tqdm_flag: bool = False,
           seed: int = 1,
           tol: float = 1e-4,
           max_iter: int = 100) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    K-means聚类算法实现

    Args:
        X: 输入数据 [N, D]
        num_clusters: 聚类数量
        cluster_centers: 初始聚类中心 [num_clusters, D]
        distance: 距离度量方式
        device: 计算设备
        tqdm_flag: 是否显示进度条
        seed: 随机种子
        tol: 收敛容差
        max_iter: 最大迭代次数

    Returns:
        cluster_ids: 聚类标签 [N]
        cluster_centers: 聚类中心 [num_clusters, D]
    """
    if distance != 'euclidean':
        raise ValueError("目前只支持欧几里得距离")

    # 设置随机种子
    torch.manual_seed(seed)
    np.random.seed(seed)

    X = X.to(device)
    N, D = X.shape

    # 初始化聚类中心
    if cluster_centers is None:
        # 随机选择数据点作为初始中心
        indices = torch.randperm(N)[:num_clusters]
        cluster_centers = X[indices].clone()
    else:
        cluster_centers = cluster_centers.to(device)

    # 初始化变量
    prev_cluster_centers = cluster_centers.clone()
    iteration = 0

    while iteration < max_iter:
        # 计算距离矩阵 [N, num_clusters]
        distances = torch.cdist(X, cluster_centers, p=2)

        # 分配聚类标签
        cluster_ids = torch.argmin(distances, dim=1)

        # 更新聚类中心
        new_cluster_centers = torch.zeros_like(cluster_centers)
        for k in range(num_clusters):
            mask = (cluster_ids == k)
            if mask.sum() > 0:
                new_cluster_centers[k] = X[mask].mean(dim=0)
            else:
                # 如果某个聚类为空，重新初始化
                new_cluster_centers[k] = X[torch.randint(0, N, (1,))].squeeze()

        # 检查收敛
        center_shift = torch.norm(new_cluster_centers - prev_cluster_centers)
        if center_shift < tol:
            break

        cluster_centers = new_cluster_centers.clone()
        prev_cluster_centers = new_cluster_centers.clone()
        iteration += 1

    return cluster_ids, cluster_centers