"""
数据预处理主脚本

解析原始数据，验证完整性，并生成训练/验证/测试集划分
"""

import sys
from pathlib import Path

# 添加项目根目录到Python路径
project_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(project_root))

import argparse
from src.data.data_parser import (
    parse_data_directory,
    validate_patient_data,
    get_statistics,
    print_statistics
)
from src.data.data_splitter import generate_all_splits
from src.utils.logger import setup_logger
from src.utils.seed import set_seed


def main():
    parser = argparse.ArgumentParser(description='数据预处理脚本')
    parser.add_argument(
        '--root_dir',
        type=str,
        default=r"D:\data\raw\Grade",
        help='数据根目录'
    )
    parser.add_argument(
        '--modalities',
        nargs='+',
        default=['A', 'P'],
        help='使用的模态'
    )
    parser.add_argument(
        '--output_dir',
        type=str,
        default=r"d:\code\feature_extract\data\splits",
        help='输出目录'
    )
    parser.add_argument(
        '--train_ratio',
        type=float,
        default=0.7,
        help='训练集比例'
    )
    parser.add_argument(
        '--seed',
        type=int,
        default=42,
        help='随机种子'
    )
    parser.add_argument(
        '--log_dir',
        type=str,
        default=r"d:\code\feature_extract\outputs\feature_extract\logs",
        help='日志目录'
    )
    
    args = parser.parse_args()
    
    # 设置日志
    logger = setup_logger(
        name="feature_extract",
        log_dir=Path(args.log_dir),
        console=True
    )
    
    # 设置随机种子
    set_seed(args.seed)
    
    logger.info("=" * 60)
    logger.info("开始数据预处理")
    logger.info("=" * 60)
    logger.info(f"数据根目录: {args.root_dir}")
    logger.info(f"使用模态: {args.modalities}")
    logger.info(f"输出目录: {args.output_dir}")
    logger.info(f"训练集比例: {args.train_ratio}")
    logger.info(f"随机种子: {args.seed}")
    
    # 1. 解析数据目录
    logger.info("\n步骤 1/4: 解析数据目录...")
    patient_dict = parse_data_directory(
        root_dir=args.root_dir,
        modalities=args.modalities
    )
    
    # 2. 验证数据
    logger.info("\n步骤 2/4: 验证数据完整性...")
    valid_patients, filtered_patients = validate_patient_data(
        patient_dict=patient_dict,
        required_modalities=args.modalities
    )
    
    if filtered_patients:
        logger.info(f"\n过滤的病人ID列表（前10个）:")
        for patient_id in filtered_patients[:10]:
            logger.info(f"  {patient_id}")
        if len(filtered_patients) > 10:
            logger.info(f"  ... 还有 {len(filtered_patients) - 10} 个")
    
    # 3. 打印统计信息
    logger.info("\n步骤 3/4: 数据统计...")
    stats = get_statistics(valid_patients)
    print_statistics(stats)
    
    # 4. 生成数据划分
    logger.info("\n步骤 4/4: 生成数据划分文件...")
    generate_all_splits(
        patient_dict=valid_patients,
        modalities=args.modalities,
        output_dir=args.output_dir,
        train_ratio=args.train_ratio,
        random_state=args.seed
    )
    
    logger.info("\n" + "=" * 60)
    logger.info("数据预处理完成！")
    logger.info("=" * 60)
    logger.info(f"\n生成的CSV文件位于: {args.output_dir}")
    logger.info("下一步: 使用 train.py 开始训练模型")


if __name__ == '__main__':
    main()
