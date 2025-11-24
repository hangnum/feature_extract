"""
数据集划分模块

按照医院和病人ID划分训练集、验证集和测试集
"""

from pathlib import Path
from typing import Dict, List, Tuple
import pandas as pd
import logging
from sklearn.model_selection import train_test_split

logger = logging.getLogger("feature_extract")


def split_by_hospital(
    patient_dict: Dict[str, Dict],
    jm_hospital: str = 'JM',
    train_ratio: float = 0.7,
    random_state: int = 42
) -> Tuple[Dict, Dict, Dict]:
    """
    按医院划分数据集
    
    - JM医院: 按病人ID进行分层抽样，7:3划分训练集和验证集
    - 其他医院: 全部作为测试集（外验）
    
    Args:
        patient_dict: 病人数据字典
        jm_hospital: JM医院的名称
        train_ratio: 训练集比例
        random_state: 随机种子
    
    Returns:
        (训练集, 验证集, 测试集) 三个病人数据字典
    """
    logger.info(f"开始划分数据集 (训练集比例: {train_ratio})")
    
    # 分离JM医院和其他医院的数据
    jm_patients = {}
    other_patients = {}
    
    for patient_id, patient_info in patient_dict.items():
        if patient_info['hospital'] == jm_hospital:
            jm_patients[patient_id] = patient_info
        else:
            other_patients[patient_id] = patient_info
    
    logger.info(f"JM医院病人数: {len(jm_patients)}")
    logger.info(f"其他医院病人数: {len(other_patients)}")
    
    # JM医院按7:3划分
    if len(jm_patients) == 0:
        logger.warning("未找到JM医院的数据！")
        train_patients = {}
        val_patients = {}
    else:
        jm_patient_ids = list(jm_patients.keys())
        jm_labels = [jm_patients[pid]['label'] for pid in jm_patient_ids]
        
        # 分层抽样
        train_ids, val_ids = train_test_split(
            jm_patient_ids,
            train_size=train_ratio,
            stratify=jm_labels,
            random_state=random_state
        )
        
        train_patients = {pid: jm_patients[pid] for pid in train_ids}
        val_patients = {pid: jm_patients[pid] for pid in val_ids}
        
        logger.info(f"训练集病人数: {len(train_patients)}")
        logger.info(f"验证集病人数: {len(val_patients)}")
    
    # 其他医院作为测试集
    test_patients = other_patients
    logger.info(f"测试集病人数: {len(test_patients)}")
    
    # 打印每个集合的标签分布
    for name, patients in [('训练集', train_patients), 
                           ('验证集', val_patients), 
                           ('测试集', test_patients)]:
        if patients:
            label_counts = {}
            for patient_info in patients.values():
                label = patient_info['label']
                label_counts[label] = label_counts.get(label, 0) + 1
            logger.info(f"{name}标签分布: {label_counts}")
    
    return train_patients, val_patients, test_patients


def generate_split_csv(
    train_patients: Dict[str, Dict],
    val_patients: Dict[str, Dict],
    test_patients: Dict[str, Dict],
    modality: str,
    output_dir: str
) -> None:
    """
    生成训练集、验证集、测试集的CSV文件
    
    CSV格式: image_path,label
    
    Args:
        train_patients: 训练集病人数据
        val_patients: 验证集病人数据
        test_patients: 测试集病人数据
        modality: 模态名称 (A or P)
        output_dir: 输出目录
    """
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    def create_csv(patients: Dict[str, Dict], split_name: str) -> None:
        """创建单个CSV文件"""
        data = []
        
        for patient_id, patient_info in patients.items():
            image_paths = patient_info['image_paths'].get(modality, [])
            label = patient_info['label']
            
            for img_path in image_paths:
                data.append({
                    'patient_id': patient_id,
                    'image_path': img_path,
                    'label': label
                })
        
        # 始终生成CSV，即便为空也生成文件，便于下游检测是否存在外验/测试集
        df = pd.DataFrame(data, columns=['patient_id', 'image_path', 'label'])
        csv_path = output_path / f"{split_name}_{modality}.csv"
        df.to_csv(csv_path, index=False, encoding='utf-8')
        if data:
            logger.info(f"保存 {split_name}_{modality}.csv: {len(df)} 条记录")
        else:
            logger.warning(f"{split_name}_{modality} 没有数据，已生成空文件: {csv_path}")
    
    # 生成三个CSV文件
    create_csv(train_patients, 'train')
    create_csv(val_patients, 'val')
    create_csv(test_patients, 'test')


def generate_all_splits(
    patient_dict: Dict[str, Dict],
    modalities: List[str],
    output_dir: str,
    train_ratio: float = 0.7,
    random_state: int = 42
) -> None:
    """
    生成所有模态的数据划分CSV文件
    
    Args:
        patient_dict: 病人数据字典
        modalities: 模态列表
        output_dir: 输出目录
        train_ratio: 训练集比例
        random_state: 随机种子
    """
    logger.info(f"开始生成数据划分文件，模态: {modalities}")
    
    # 划分数据集
    train_patients, val_patients, test_patients = split_by_hospital(
        patient_dict,
        train_ratio=train_ratio,
        random_state=random_state
    )
    
    # 为每个模态生成CSV文件
    for modality in modalities:
        logger.info(f"\n处理模态: {modality}")
        generate_split_csv(
            train_patients,
            val_patients,
            test_patients,
            modality,
            output_dir
        )
    
    logger.info(f"\n所有数据划分文件已保存至: {output_dir}")
