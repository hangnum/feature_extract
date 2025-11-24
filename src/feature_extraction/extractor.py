"""
特征提取模块

提取病人级特征并保存
"""

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from pathlib import Path
from typing import Dict, List
import numpy as np
import pandas as pd
import logging
from tqdm import tqdm
from collections import defaultdict

logger = logging.getLogger("feature_extract")


class FeatureExtractor:
    """特征提取器"""
    
    def __init__(
        self,
        model: nn.Module,
        device: str = 'cuda'
    ):
        """
        初始化特征提取器
        
        Args:
            model: 特征提取模型（已移除分类层）
            device: 设备
        """
        self.model = model.to(device)
        self.model.eval()
        self.device = device
    
    def extract_batch(self, images: torch.Tensor) -> np.ndarray:
        """
        提取一个batch的特征
        
        Args:
            images: 图像张量
        
        Returns:
            特征数组 (batch_size, feature_dim)
        """
        with torch.no_grad():
            images = images.to(self.device)
            features = self.model(images)
            
            # 展平特征（如果是多维的）
            if len(features.shape) > 2:
                features = features.view(features.size(0), -1)
            
            return features.cpu().numpy()
    
    def extract_patient_features(
        self,
        dataloader: DataLoader,
        output_dir: str,
        modality: str
    ) -> None:
        """
        提取病人级特征
        
        将每个病人的所有slice特征拼接为(n, m)矩阵
        
        Args:
            dataloader: 数据加载器（需要返回patient_id）
            output_dir: 输出目录
            modality: 模态名称
        """
        logger.info(f"开始提取特征 - 模态: {modality}")
        
        # 用于存储每个病人的特征
        patient_features = defaultdict(list)
        patient_labels = {}
        
        # 提取特征
        progress_bar = tqdm(dataloader, desc='提取特征')
        for images, labels, patient_ids in progress_bar:
            # 提取当前batch的特征
            features = self.extract_batch(images)
            
            # 按病人ID组织特征
            for i, patient_id in enumerate(patient_ids):
                patient_features[patient_id].append(features[i])
                patient_labels[patient_id] = labels[i].item()
        
        logger.info(f"共提取 {len(patient_features)} 个病人的特征")
        
        # 保存特征
        output_path = Path(output_dir)
        
        feature_info = []
        
        for patient_id, features_list in patient_features.items():
            # 拼接特征 (n, m)
            patient_feature_matrix = np.stack(features_list, axis=0)
            
            label = patient_labels[patient_id]
            label_name = f"grade{label}"
            
            # 创建保存目录
            save_dir = output_path / modality / label_name
            save_dir.mkdir(parents=True, exist_ok=True)
            
            # 保存为.npy文件
            feature_file = save_dir / f"{patient_id}.npy"
            np.save(feature_file, patient_feature_matrix)
            
            # 记录信息
            feature_info.append({
                'patient_id': patient_id,
                'modality': modality,
                'label': label,
                'num_slices': len(features_list),
                'feature_dim': patient_feature_matrix.shape[1],
                'feature_path': str(feature_file)
            })
        
        # 保存特征信息到CSV
        info_df = pd.DataFrame(feature_info)
        info_csv = output_path / f'features_{modality}.csv'
        info_df.to_csv(info_csv, index=False)
        logger.info(f"特征信息已保存至: {info_csv}")
        
        logger.info(f"所有特征已保存至: {output_path / modality}")


def align_patient_features(
    modalities: List[str],
    feature_dir: str,
    output_dir: str
) -> None:
    """
    对齐不同模态的病人特征
    
    确保同一病人在不同模态中的顺序一致
    
    Args:
        modalities: 模态列表
        feature_dir: 特征目录
        output_dir: 输出目录
    """
    logger.info("开始对齐多模态特征...")
    
    feature_dir = Path(feature_dir)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # 读取每个模态的特征信息
    modality_dfs = {}
    for modality in modalities:
        csv_path = feature_dir / f'features_{modality}.csv'
        if csv_path.exists():
            modality_dfs[modality] = pd.read_csv(csv_path)
        else:
            logger.warning(f"未找到 {modality} 的特征信息文件")
    
    if len(modality_dfs) < 2:
        logger.error("至少需要两个模态的特征")
        return
    
    # 找到所有模态共有的病人ID
    common_patients = set(modality_dfs[modalities[0]]['patient_id'])
    for modality in modalities[1:]:
        common_patients &= set(modality_dfs[modality]['patient_id'])
    
    common_patients = sorted(list(common_patients))
    logger.info(f"共有病人数: {len(common_patients)}")
    
    # 为每个模态生成对齐后的CSV
    for modality in modalities:
        df = modality_dfs[modality]
        aligned_df = df[df['patient_id'].isin(common_patients)]
        aligned_df = aligned_df.set_index('patient_id').loc[common_patients].reset_index()
        
        output_csv = output_dir / f'aligned_features_{modality}.csv'
        aligned_df.to_csv(output_csv, index=False)
        logger.info(f"保存对齐后的 {modality} 特征信息: {output_csv}")
    
    logger.info("多模态特征对齐完成！")
