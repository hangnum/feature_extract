"""
数据解析与验证模块

解析医疗图像数据目录，构建病人级数据字典，并验证数据完整性
"""

from pathlib import Path
from typing import Dict, List, Tuple, Optional
from collections import defaultdict
import logging

logger = logging.getLogger("feature_extract")


def parse_data_directory(
    root_dir: str,
    modalities: Optional[List[str]] = None
) -> Dict[str, Dict]:
    """
    Parse medical image directory into a patient-level dictionary.

    Supports paths with or without a fold level:
      - root_dir/Grade/hospital/fold/label/patient_id/modality/image.png
      - root_dir/Grade/hospital/label/patient_id/modality/image.png
    """
    root_path = Path(root_dir)

    # 处理默认的模态列表
    if modalities is None:
        modalities = ['A', 'P']

    if not root_path.exists():
        raise FileNotFoundError(f"数据目录不存在: {root_dir}")

    logger.info(f"开始解析数据目录: {root_dir}")

    patient_data = {}

    for img_path in root_path.rglob("*.png"):
        parts = img_path.parts

        try:
            grade_idx = parts.index('Grade')
            rel_parts = parts[grade_idx + 1:]

            if len(rel_parts) < 5:
                logger.warning(f"路径层级不足: {img_path}")
                continue

            hospital = rel_parts[0]

            if rel_parts[1].startswith('grade'):
                # no fold
                label_str = rel_parts[1]
                patient_id = rel_parts[2]
                modality = rel_parts[3]
            else:
                # with fold
                if len(rel_parts) < 6:
                    logger.warning(f"路径层级不足（含fold）: {img_path}")
                    continue
                label_str = rel_parts[2]
                patient_id = rel_parts[3]
                modality = rel_parts[4]

            if modality not in modalities:
                continue

            if label_str == 'grade0':
                label = 0
            elif label_str == 'grade1':
                label = 1
            else:
                logger.warning(f"未知标签: {label_str}, 跳过")
                continue

            if patient_id not in patient_data:
                patient_data[patient_id] = {
                    'hospital': hospital,
                    'label': label,
                    'image_paths': {m: [] for m in modalities}
                }

            patient_data[patient_id]['image_paths'][modality].append(str(img_path))

        except (ValueError, IndexError) as e:
            logger.warning(f"解析失败: {img_path}, 错误: {e}")
            continue

    logger.info(f"解析完成，共找到 {len(patient_data)} 个病人")

    return patient_data

def validate_patient_data(
    patient_dict: Dict[str, Dict],
    required_modalities: Optional[List[str]] = None
) -> Tuple[Dict[str, Dict], List[str]]:
    """
    验证病人数据，过滤缺失模态的病人
    
    注意：不删除原始数据，只是在划分时不包含这些病人
    
    Args:
        patient_dict: 病人数据字典
        required_modalities: 必须的模态列表
    
    Returns:
        (有效的病人数据字典, 被过滤的病人ID列表)
    """
    # 处理默认的模态列表
    if required_modalities is None:
        required_modalities = ['A', 'P']

    valid_patients = {}
    filtered_patients = []

    for patient_id, patient_info in patient_dict.items():
        # 检查每个必须的模态是否都有数据
        is_valid = True
        for modality in required_modalities:
            if not patient_info['image_paths'].get(modality):
                is_valid = False
                break
        
        if is_valid:
            valid_patients[patient_id] = patient_info
        else:
            filtered_patients.append(patient_id)
            missing_modalities = [
                m for m in required_modalities 
                if not patient_info['image_paths'].get(m)
            ]
            logger.info(
                f"病人 {patient_id} 缺失模态 {missing_modalities}，不计入本次任务"
            )
    
    logger.info(f"有效病人数: {len(valid_patients)}")
    logger.info(f"过滤病人数: {len(filtered_patients)}")
    
    return valid_patients, filtered_patients


def get_statistics(patient_dict: Dict[str, Dict]) -> Dict:
    """
    获取数据集统计信息
    
    Args:
        patient_dict: 病人数据字典
    
    Returns:
        统计信息字典
    """
    stats = {
        'total_patients': len(patient_dict),
        'by_hospital': defaultdict(int),
        'by_label': defaultdict(int),
        'images_per_modality': defaultdict(int)
    }
    
    for patient_id, patient_info in patient_dict.items():
        stats['by_hospital'][patient_info['hospital']] += 1
        stats['by_label'][patient_info['label']] += 1
        
        for modality, paths in patient_info['image_paths'].items():
            stats['images_per_modality'][modality] += len(paths)
    
    return dict(stats)


def print_statistics(stats: Dict) -> None:
    """
    打印统计信息
    
    Args:
        stats: 统计信息字典
    """
    logger.info("=" * 50)
    logger.info("数据集统计信息")
    logger.info("=" * 50)
    logger.info(f"总病人数: {stats['total_patients']}")
    
    logger.info("\n按医院分布:")
    for hospital, count in stats['by_hospital'].items():
        logger.info(f"  {hospital}: {count}")
    
    logger.info("\n按标签分布:")
    for label, count in stats['by_label'].items():
        label_name = "grade0" if label == 0 else "grade1"
        logger.info(f"  {label_name}: {count}")
    
    logger.info("\n按模态图片数:")
    for modality, count in stats['images_per_modality'].items():
        logger.info(f" {modality}: {count}")
    
    logger.info("=" * 50)
