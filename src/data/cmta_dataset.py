"""
CMTA模型的数据加载器

支持多模态特征数据的加载和预处理：
1. CT特征和病理学特征
2. 标签和患者信息
3. 数据增强和预处理
"""

import os
import torch
from torch.utils.data import Dataset, DataLoader
import scipy.io as scio
from typing import Tuple, Optional, Dict


class CMTADataset(Dataset):
    """
    CMTA数据集类

    支持加载预处理的多模态特征数据
    """

    def __init__(self,
                 ct_data_file: str,
                 pathology_data_file: str,
                 dataset_type: str = 'train',
                 transform: bool = True):
        """
        初始化数据集

        Args:
            ct_data_file: CT特征文件路径
            pathology_data_file: 病理学特征文件路径
            dataset_type: 数据集类型 ('train', 'val', 'test')
            transform: 是否进行数据变换
        """
        self.ct_data_file = ct_data_file
        self.pathology_data_file = pathology_data_file
        self.dataset_type = dataset_type
        self.transform = transform
        self.ct_size = 0
        self.pathology_size = 0
        self.ct_list = []
        self.pathology_list = []

        # 加载数据
        self._load_data()

    def _load_data(self):
        """加载数据文件列表"""
        # 加载CT特征路径
        if os.path.isfile(self.ct_data_file):
            with open(self.ct_data_file, 'r') as f:
                for line in f:
                    line = line.strip()
                    if line:
                        self.ct_list.append(line)
                        self.ct_size += 1
        else:
            print(f"警告: CT数据文件 {self.ct_data_file} 不存在")

        # 加载病理学特征路径
        if os.path.isfile(self.pathology_data_file):
            with open(self.pathology_data_file, 'r') as f:
                for line in f:
                    line = line.strip()
                    if line:
                        self.pathology_list.append(line)
                        self.pathology_size += 1
        else:
            print(f"警告: 病理学数据文件 {self.pathology_data_file} 不存在")

        # 确保两个列表长度一致
        min_length = min(len(self.ct_list), len(self.pathology_list))
        self.ct_list = self.ct_list[:min_length]
        self.pathology_list = self.pathology_list[:min_length]
        self.ct_size = min_length
        self.pathology_size = min_length

        print(f"加载 {self.dataset_type} 数据: {self.ct_size} 个样本")

    def __len__(self) -> int:
        return self.ct_size

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, str]:
        """
        获取单个数据样本

        Returns:
            ct_features: CT特征 [N, D]
            pathology_features: 病理学特征 [N, D]
            label: 标签 [1]
            data_path: 数据路径（用于标识）
        """
        # 解析CT特征路径
        ct_path = self.ct_list[idx].split('*')[0]
        if not os.path.isfile(ct_path):
            print(f"CT特征文件不存在: {ct_path}")
            # 返回零张量作为后备
            ct_features = torch.zeros(100, 3904)
            label = 0
            data_path = ct_path
        else:
            try:
                ct_data = scio.loadmat(ct_path)
                ct_features = torch.from_numpy(ct_data['feature_map']).float()
                label = int(self.ct_list[idx].split('*')[2])
                data_path = ct_path
            except Exception as e:
                print(f"加载CT特征失败: {ct_path}, 错误: {e}")
                ct_features = torch.zeros(100, 3904)
                label = 0
                data_path = ct_path

        # 解析病理学特征路径
        pathology_path = self.pathology_list[idx].split('*')[0]
        if not os.path.isfile(pathology_path):
            print(f"病理学特征文件不存在: {pathology_path}")
            pathology_features = torch.zeros(100, 3904)
        else:
            try:
                pathology_data = scio.loadmat(pathology_path)
                pathology_features = torch.from_numpy(pathology_data['feature_map']).float()
            except Exception as e:
                print(f"加载病理学特征失败: {pathology_path}, 错误: {e}")
                pathology_features = torch.zeros(100, 3904)

        # 确保特征长度一致
        max_length = max(ct_features.shape[0], pathology_features.shape[0])

        if ct_features.shape[0] < max_length:
            padding = torch.zeros(max_length - ct_features.shape[0], ct_features.shape[1])
            ct_features = torch.cat([ct_features, padding], dim=0)

        if pathology_features.shape[0] < max_length:
            padding = torch.zeros(max_length - pathology_features.shape[0], pathology_features.shape[1])
            pathology_features = torch.cat([pathology_features, padding], dim=0)

        # 数据变换（如果需要）
        if self.transform:
            ct_features = self._apply_transform(ct_features)
            pathology_features = self._apply_transform(pathology_features)

        label_tensor = torch.tensor(label, dtype=torch.long)

        return ct_features, pathology_features, label_tensor, data_path

    def _apply_transform(self, features: torch.Tensor) -> torch.Tensor:
        """应用数据变换"""
        # 这里可以添加数据增强技术，例如：
        # - 特征归一化
        # - 添加噪声
        # - 随机采样

        # 简单的特征标准化
        if features.std() > 0:
            features = (features - features.mean()) / features.std()

        return features


class CMTADataLoader:
    """CMTA数据加载器管理类"""

    def __init__(self, args):
        self.args = args
        self.batch_size = args.batch_size

    def create_dataloaders(self,
                          data_dir: str,
                          modalities: Optional[list[str]] = None) -> Tuple[DataLoader, DataLoader, DataLoader]:
        """
        创建训练、验证和测试数据加载器

        Args:
            data_dir: 数据根目录
            modalities: 模态列表，如 ['A', 'P'] 表示AP和PB

        Returns:
            train_loader, val_loader, test_loader
        """
        # 构建文件路径
        if modalities is None:
            modalities = ['A', 'P']
        train_files = self._build_file_paths(data_dir, 'train', modalities)
        val_files = self._build_file_paths(data_dir, 'val', modalities)
        test_files = self._build_file_paths(data_dir, 'test', modalities)

        # 创建数据集
        train_dataset = CMTADataset(
            ct_data_file=train_files['ct'],
            pathology_data_file=train_files['pathology'],
            dataset_type='train',
            transform=True
        )

        val_dataset = CMTADataset(
            ct_data_file=val_files['ct'],
            pathology_data_file=val_files['pathology'],
            dataset_type='val',
            transform=False
        )

        test_dataset = CMTADataset(
            ct_data_file=test_files['ct'],
            pathology_data_file=test_files['pathology'],
            dataset_type='test',
            transform=False
        )

        # 创建数据加载器
        train_loader = DataLoader(
            train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            drop_last=False,
            num_workers=self.args.get('num_workers', 0),
            pin_memory=torch.cuda.is_available()
        )

        val_loader = DataLoader(
            val_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            drop_last=False,
            num_workers=self.args.get('num_workers', 0),
            pin_memory=torch.cuda.is_available()
        )

        test_loader = DataLoader(
            test_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            drop_last=False,
            num_workers=self.args.get('num_workers', 0),
            pin_memory=torch.cuda.is_available()
        )

        print(f"数据加载完成:")
        print(f"  训练集: {len(train_dataset)} 个样本")
        print(f"  验证集: {len(val_dataset)} 个样本")
        print(f"  测试集: {len(test_dataset)} 个样本")

        return train_loader, val_loader, test_loader

    def _build_file_paths(self, data_dir: str, split: str, modalities: list) -> Dict[str, str]:
        """构建数据文件路径"""
        files = {}

        # 根据模态构建文件名
        if 'A' in modalities and 'P' in modalities:
            # AP和PB模态组合
            files['ct'] = os.path.join(data_dir, f'{split}_A.txt')
            files['pathology'] = os.path.join(data_dir, f'{split}_P.txt')
        elif 'A' in modalities and 'B' in modalities:
            # AB组合
            files['ct'] = os.path.join(data_dir, f'{split}_A.txt')
            files['pathology'] = os.path.join(data_dir, f'{split}_B.txt')
        else:
            # 默认组合
            files['ct'] = os.path.join(data_dir, f'{split}_A.txt')
            files['pathology'] = os.path.join(data_dir, f'{split}_P.txt')

        return files


def create_cmta_dataloaders(args) -> Tuple[DataLoader, DataLoader, DataLoader]:
    """
    创建CMTA数据加载器的便捷函数

    Args:
        args: 包含数据路径和参数的配置对象

    Returns:
        train_loader, val_loader, test_loader
    """
    data_loader = CMTADataLoader(args)
    return data_loader.create_dataloaders(
        data_dir=args.data_dir,
        modalities=getattr(args, 'modalities', ['A', 'P'])
    )


# 为了向后兼容，保留原始函数名
create_dataloaders = create_cmta_dataloaders