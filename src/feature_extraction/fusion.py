"""
Feature fusion utilities.

Reads per-modality patient features, pools them to (1, m), concatenates across
modalities, and trains/evaluates an L2-regularized classifier.
"""

from pathlib import Path
from typing import Dict, List, Tuple, Optional
import logging
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from src.training.metrics import calculate_metrics

logger = logging.getLogger("feature_extract")


def _load_modality_features(csv_path: Path) -> Tuple[Dict[str, np.ndarray], Dict[str, int]]:
    """
    Load patient-level features for a modality and pool slices by mean.
    """
    if not csv_path.exists():
        raise FileNotFoundError(f"Feature info not found: {csv_path}")

    df = pd.read_csv(csv_path)
    features: Dict[str, np.ndarray] = {}
    labels: Dict[str, int] = {}

    for _, row in df.iterrows():
        feature_array = np.load(row['feature_path'])
        pooled = feature_array.mean(axis=0)
        features[row['patient_id']] = pooled
        labels[row['patient_id']] = int(row['label'])

    logger.info(f"Loaded {len(features)} patients from {csv_path}")
    return features, labels


def _select_csv(csv_dir: Path, modality: str, use_aligned: bool) -> Optional[Path]:
    """
    Pick the CSV path for a modality, preferring aligned files when requested.
    """
    candidates = []
    if use_aligned:
        candidates.append(csv_dir / 'aligned' / f'aligned_features_{modality}.csv')
    candidates.append(csv_dir / f'features_{modality}.csv')

    for candidate in candidates:
        if candidate.exists():
            return candidate
    return None


def build_fusion_matrix(
    modalities: List[str],
    csv_dir: Path,
    use_aligned: bool = False
) -> Tuple[np.ndarray, np.ndarray, List[str]]:
    """
    Build concatenated feature matrix for a split.
    """
    modality_features: Dict[str, Dict[str, np.ndarray]] = {}
    modality_labels: Dict[str, Dict[str, int]] = {}

    for modality in modalities:
        csv_path = _select_csv(csv_dir, modality, use_aligned)
        if csv_path is None:
            raise FileNotFoundError(f"No feature CSV found for modality {modality} in {csv_dir}")
        feats, labels = _load_modality_features(csv_path)
        modality_features[modality] = feats
        modality_labels[modality] = labels

    # Intersection of patient IDs across modalities
    common_patients = set.intersection(*[set(f.keys()) for f in modality_features.values()])
    if not common_patients:
        raise ValueError("No common patients across modalities; cannot fuse features.")

    patient_ids = sorted(common_patients)
    feature_list = []
    labels = []

    for pid in patient_ids:
        concat_feats = np.concatenate([modality_features[m][pid] for m in modalities], axis=0)
        feature_list.append(concat_feats)

        # Consistency check for labels across modalities
        label = modality_labels[modalities[0]][pid]
        for m in modalities[1:]:
            if modality_labels[m][pid] != label:
                raise ValueError(f"Label mismatch for patient {pid} between modalities.")
        labels.append(label)

    X = np.vstack(feature_list)
    y = np.array(labels)
    logger.info(
        f"Built fusion matrix from {csv_dir} | patients: {len(patient_ids)} | "
        f"feature_dim: {X.shape[1]}"
    )
    return X, y, patient_ids


def train_l2_classifier(
    X_train: np.ndarray,
    y_train: np.ndarray,
    C: float = 1.0,
    max_iter: int = 1000,
    random_state: int = 42
) -> Pipeline:
    """
    Train L2-regularized logistic regression with standard scaling.
    """
    clf = Pipeline([
        ("scaler", StandardScaler()),
        ("logreg", LogisticRegression(
            penalty="l2",
            C=C,
            max_iter=max_iter,
            solver="liblinear",
            class_weight="balanced",
            random_state=random_state
        ))
    ])
    clf.fit(X_train, y_train)
    return clf


def evaluate_classifier(
    model: Pipeline,
    X: np.ndarray,
    y: np.ndarray
) -> Tuple[np.ndarray, np.ndarray, Dict[str, float]]:
    """
    Evaluate classifier and return preds, probs, and metrics.
    """
    probs = model.predict_proba(X)[:, 1]
    preds = (probs >= 0.5).astype(int)
    metrics = calculate_metrics(y_true=y, y_pred=preds, y_prob=probs)
    return preds, probs, metrics
