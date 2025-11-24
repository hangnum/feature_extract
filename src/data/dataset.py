"""
PyTorch数据集类

支持从CSV文件加载医疗图像数据
"""

import pandas as pd
import torch
from torch.utils.data import Dataset
from PIL import Image
from pathlib import Path
from typing import Optional, Callable, Tuple
import logging

logger = logging.getLogger("feature_extract")


class MedicalImageDataset(Dataset):
    """
    医疗图像数据集
    
    支持加载224x224x1的灰度图像
    """
    
    def __init__(
        self,
        csv_path: str,
        transform: Optional[Callable] = None,
        return_patient_id: bool = False
    ):
        """
        初始化数据集
        
        Args:
            csv_path: CSV文件路径，包含image_path和label列
            transform: 数据增强/预处理函数
            return_patient_id: 是否返回病人ID（用于特征提取）
        """
        self.csv_path = csv_path
        self.transform = transform
        self.return_patient_id = return_patient_id
        
        # 加载CSV
        self.data = pd.read_csv(csv_path)
        
        logger.info(f"加载数据集: {csv_path}")
        logger.info(f"  样本数: {len(self.data)}")
        logger.info(f"  标签分布: {self.data['label'].value_counts().to_dict()}")
        
        # 检查文件是否存在
        missing_files = []
        for idx, row in self.data.iterrows():
            if not Path(row['image_path']).exists():
                missing_files.append(row['image_path'])
        
        if missing_files:
            logger.warning(f"发现 {len(missing_files)} 个缺失文件")
            if len(missing_files) <= 5:
                for f in missing_files:
                    logger.warning(f"  缺失: {f}")
    
    def __len__(self) -> int:
        """返回数据集大小"""
        return len(self.data)
    
    def __getitem__(self, idx: int) -> Tuple:
        """
        获取单个样本
        
        Args:
            idx: 索引
        
        Returns:
            如果return_patient_id=False: (image, label)
            如果return_patient_id=True: (image, label, patient_id)
        """
        row = self.data.iloc[idx]
        
        # 加载图像
        img_path = row['image_path']
        image = Image.open(img_path).convert('L')  # 转为灰度图
        
        # 应用变换
        if self.transform:
            image = self.transform(image)
        
        # 获取标签
        label = torch.tensor(row['label'], dtype=torch.long)
        
        # 返回结果
        if self.return_patient_id:
            patient_id = row['patient_id']
            return image, label, patient_id
        else:
            return image, label
    
    def get_labels(self) -> list:
        """
        获取所有标签
        
        Returns:
            标签列表
        """
        return self.data['label'].tolist()
    
    def get_patient_ids(self) -> list:
        """
        获取所有病人ID
        
        Returns:
            病人ID列表
        """
        if 'patient_id' in self.data.columns:
            return self.data['patient_id'].tolist()
        else:
            return []
