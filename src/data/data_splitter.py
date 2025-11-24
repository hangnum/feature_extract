"""
Dataset splitting utilities.

Split patients into train/val/test with hospital-based logic and stratified
sampling for the JM hospital.
"""

from pathlib import Path
from typing import Dict, List, Tuple
import logging
import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedShuffleSplit

logger = logging.getLogger("feature_extract")


def _stratified_train_val_split(
    patient_ids: List[str],
    labels: List[int],
    train_ratio: float,
    random_state: int
) -> Tuple[List[str], List[str]]:
    """
    Stratified split for JM patients. Falls back to deterministic random split
    if stratification is impossible (e.g., label counts < 2).
    """
    label_counts = pd.Series(labels).value_counts().to_dict()
    logger.info(f"JM label counts: {label_counts}")

    if len(set(labels)) < 2 or any(count < 2 for count in label_counts.values()):
        logger.warning("Insufficient label diversity for stratified split, using random split instead.")
        rng = np.random.default_rng(random_state)
        indices = rng.permutation(len(patient_ids))
        split_idx = int(len(patient_ids) * train_ratio)
        train_idx, val_idx = indices[:split_idx], indices[split_idx:]
        return [patient_ids[i] for i in train_idx], [patient_ids[i] for i in val_idx]

    splitter = StratifiedShuffleSplit(
        n_splits=1,
        train_size=train_ratio,
        random_state=random_state
    )
    train_idx, val_idx = next(splitter.split(patient_ids, labels))
    return [patient_ids[i] for i in train_idx], [patient_ids[i] for i in val_idx]


def _log_label_distribution(name: str, patients: Dict[str, Dict]) -> None:
    """Log label counts for a split."""
    if not patients:
        logger.info(f"{name} is empty.")
        return
    label_counts: Dict[int, int] = {}
    for patient_info in patients.values():
        label = patient_info['label']
        label_counts[label] = label_counts.get(label, 0) + 1
    logger.info(f"{name} label distribution: {label_counts}")


def split_by_hospital(
    patient_dict: Dict[str, Dict],
    jm_hospital: str = 'JM',
    train_ratio: float = 0.7,
    random_state: int = 42
) -> Tuple[Dict, Dict, Dict]:
    """
    Split patients by hospital.

    - JM hospital: stratified split (train/val).
    - Other hospitals: test (external validation).
    """
    logger.info(f"Splitting dataset (train_ratio={train_ratio})")

    jm_patients: Dict[str, Dict] = {}
    other_patients: Dict[str, Dict] = {}

    for patient_id, patient_info in patient_dict.items():
        if patient_info['hospital'] == jm_hospital:
            jm_patients[patient_id] = patient_info
        else:
            other_patients[patient_id] = patient_info

    logger.info(f"JM patients: {len(jm_patients)}")
    logger.info(f"Other hospital patients (test): {len(other_patients)}")

    if len(jm_patients) == 0:
        logger.warning("No JM patients found; skipping train/val split.")
        train_patients: Dict[str, Dict] = {}
        val_patients: Dict[str, Dict] = {}
    else:
        jm_patient_ids = list(jm_patients.keys())
        jm_labels = [jm_patients[pid]['label'] for pid in jm_patient_ids]

        train_ids, val_ids = _stratified_train_val_split(
            jm_patient_ids,
            jm_labels,
            train_ratio,
            random_state
        )

        train_patients = {pid: jm_patients[pid] for pid in train_ids}
        val_patients = {pid: jm_patients[pid] for pid in val_ids}

        logger.info(f"Train patients: {len(train_patients)}")
        logger.info(f"Val patients: {len(val_patients)}")

    test_patients = other_patients
    logger.info(f"Test patients: {len(test_patients)}")

    _log_label_distribution("Train", train_patients)
    _log_label_distribution("Val", val_patients)
    _log_label_distribution("Test", test_patients)

    return train_patients, val_patients, test_patients


def generate_split_csv(
    train_patients: Dict[str, Dict],
    val_patients: Dict[str, Dict],
    test_patients: Dict[str, Dict],
    modality: str,
    output_dir: str
) -> None:
    """
    Generate CSVs for train/val/test splits.

    CSV columns: patient_id, image_path, label
    """
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    def create_csv(patients: Dict[str, Dict], split_name: str) -> None:
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

        df = pd.DataFrame(data, columns=['patient_id', 'image_path', 'label'])
        csv_path = output_path / f"{split_name}_{modality}.csv"
        df.to_csv(csv_path, index=False, encoding='utf-8')
        if data:
            logger.info(f"Saved {split_name}_{modality}.csv with {len(df)} rows")
        else:
            logger.warning(f"{split_name}_{modality} is empty; created placeholder at {csv_path}")

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
    Generate split CSVs for all modalities.
    """
    logger.info(f"Generating split CSVs for modalities: {modalities}")

    train_patients, val_patients, test_patients = split_by_hospital(
        patient_dict,
        train_ratio=train_ratio,
        random_state=random_state
    )

    for modality in modalities:
        logger.info(f"\nProcessing modality {modality}")
        generate_split_csv(
            train_patients,
            val_patients,
            test_patients,
            modality,
            output_dir
        )

    logger.info(f"\nSplit files saved to: {output_dir}")
