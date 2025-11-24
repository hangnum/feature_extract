"""
特征提取脚本

从训练好的模型中提取病人级特征
"""

import sys
from pathlib import Path

# 添加项目根目录到Python路径
project_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(project_root))

import argparse
import torch
from torch.utils.data import DataLoader

from src.data.dataset import MedicalImageDataset
from src.data.transforms import get_val_transform
from src.models.model_loader import load_model, get_feature_extractor
from src.feature_extraction.extractor import FeatureExtractor, align_patient_features
from src.utils.logger import setup_logger
from src.utils.seed import set_seed


def main():
    parser = argparse.ArgumentParser(description='特征提取脚本')
    parser.add_argument('--modality', type=str, required=True, choices=['A', 'P'], help='提取的模态')
    parser.add_argument('--model', type=str, default='resnet18', help='模型名称')
    parser.add_argument('--checkpoint', type=str, required=True, help='模型检查点路径')
    parser.add_argument('--output_dir', type=str, default=None, help='输出目录')
    parser.add_argument('--batch_size', type=int, default=32, help='batch size')
    parser.add_argument('--device', type=str, default='cuda', help='设备')
    parser.add_argument('--align', action='store_true', help='对齐多模态特征')
    
    args = parser.parse_args()
    
    # 设置随机种子
    set_seed(42)
    
    # 设置日志
    logger = setup_logger(
        name="feature_extract",
        log_dir=Path(project_root) / 'outputs' / 'feature_extract' / 'logs',
        console=True
    )
    
    logger.info("=" * 60)
    logger.info("特征提取")
    logger.info("=" * 60)
    logger.info(f"模型: {args.model}")
    logger.info(f"模态: {args.modality}")
    logger.info(f"检查点: {args.checkpoint}")
    
    # 输出目录
    if args.output_dir is None:
        args.output_dir = str(Path(project_root) / 'data' / 'features')
    
    # 加载模型
    logger.info("\n加载模型...")
    model = load_model(
        model_name=args.model,
        num_classes=2,
        pretrained=False,
        freeze_stages=0
    )
    
    # 加载检查点
    checkpoint = torch.load(args.checkpoint, map_location=args.device, weights_only=False)
    model.load_state_dict(checkpoint['model_state_dict'])
    logger.info(f"已加载检查点 (Epoch {checkpoint['epoch']})")
    
    # 获取特征提取器
    feature_model = get_feature_extractor(model, args.model)
    
    # 创建特征提取器
    extractor = FeatureExtractor(feature_model, device=args.device)
    
    # 准备数据
    logger.info("\n准备数据...")
    data_dir = Path(project_root) / 'data' / 'splits'
    
    # 处理训练集、验证集和测试集
    for split in ['train', 'val', 'test']:
        csv_path = data_dir / f'{split}_{args.modality}.csv'
        
        if not csv_path.exists():
            logger.warning(f"未找到 {split} 数据: {csv_path}")
            continue
        
        logger.info(f"\n处理 {split} 集...")
        
        # 数据变换（使用验证模式）
        transform = get_val_transform(image_size=224)
        
        # 数据集（返回patient_id）
        dataset = MedicalImageDataset(str(csv_path), transform=transform, return_patient_id=True)
        
        # 数据加载器
        dataloader = DataLoader(
            dataset,
            batch_size=args.batch_size,
            shuffle=False,
            num_workers=4,
            pin_memory=True
        )
        
        # 提取特征
        split_output_dir = Path(args.output_dir) / split
        extractor.extract_patient_features(
            dataloader=dataloader,
            output_dir=str(split_output_dir),
            modality=args.modality
        )
    
    logger.info("\n" + "=" * 60)
    logger.info("特征提取完成！")
    logger.info("=" * 60)
    
    # 对齐特征（如果需要）
    if args.align:
        logger.info("\n对齐多模态特征...")
        for split in ['train', 'val', 'test']:
            split_feature_dir = Path(args.output_dir) / split
            if split_feature_dir.exists():
                align_patient_features(
                    modalities=['A', 'P'],
                    feature_dir=str(split_feature_dir),
                    output_dir=str(split_feature_dir / 'aligned')
                )


if __name__ == '__main__':
    main()
