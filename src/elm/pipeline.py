"""
ELM 端到端流水线

覆盖步骤：
1) 病人级特征聚合：从切片级 feature_map 求均值，得到 Xtrain/Xtest/Xtest1。
2) 归一化：只用训练集均值/方差，零方差特征平滑为 1。
3) Mann–Whitney U 检验筛特征：同步筛三套数据，支持兜底 top-K。
4) ELM 搜索：随机隐藏层节点，筛选 AUC/间隔满足约束的候选模型。
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
import scipy.io as sio
from scipy.stats import mannwhitneyu


@dataclass
class Split:
    X: np.ndarray
    y: np.ndarray


@dataclass
class AggregatedSplits:
    train: Split
    test: Split
    test1: Split


@dataclass
class NormalizationStats:
    mean: np.ndarray
    std: np.ndarray


@dataclass
class UTestResult:
    p_train: np.ndarray
    p_test: np.ndarray
    p_test1: np.ndarray
    selected_indices: np.ndarray
    filtered: AggregatedSplits


@dataclass
class DataTypeConfig:
    """每种数据类型的配置，包含路径、日志文件和键名。"""

    name: str  # 例如 "CT" / "BL"
    feature_root: Path  # 例如 /.../jiangmen_CT_CMTA/VGG16/feature_extract_3_1
    log_root: Path  # 例如 /.../jiangmen_CT_CMTA
    log_files: Dict[str, str]  # {"train": "train_CT_log.txt", ...}
    split_dirs: Dict[str, str]  # {"train": "train_data", "test": "test_data", "test1": "test1_data"}
    mat_key: str = "feature_map"


class LabelEncoder:
    """最小化标签编码器，确保标签为 0/1。"""

    def __init__(self) -> None:
        self._map: Dict[str, int] = {}

    def encode(self, label: str) -> int:
        if label not in self._map:
            self._map[label] = len(self._map)
        return self._map[label]

    @property
    def mapping(self) -> Dict[str, int]:
        return dict(self._map)

    def assert_binary(self) -> None:
        if len(self._map) != 2:
            raise ValueError(f"Expected exactly 2 classes for binary task, got {len(self._map)}: {self._map}")


def extract_patient_id(line: str) -> str:
    """从日志行提取病人 ID，默认取倒数第二段。"""
    parts = line.strip().split("/")
    if len(parts) >= 2:
        return parts[-2]
    return parts[-1]


def load_feature_map(mat_path: Path, key: str) -> np.ndarray:
    data = sio.loadmat(mat_path)
    if key not in data:
        raise KeyError(f"{mat_path} missing key '{key}'")
    feature_map = np.asarray(data[key])
    if feature_map.ndim != 2:
        raise ValueError(f"{mat_path} expected 2D feature_map, got shape {feature_map.shape}")
    return feature_map


def build_feature_index(dataset_root: Path, mat_key: str) -> Dict[str, List[Tuple[str, Path]]]:
    """预索引特征文件，返回 patient_id -> [(method, mat_path), ...]。"""
    index: Dict[str, List[Tuple[str, Path]]] = {}
    for method_dir in dataset_root.iterdir():
        if not method_dir.is_dir():
            continue
        for patient_dir in method_dir.iterdir():
            if not patient_dir.is_dir():
                continue
            mat_path = patient_dir / f"{patient_dir.name}.mat"
            if not mat_path.exists():
                continue
            # 读一遍键确保可用，早失败
            _ = load_feature_map(mat_path, mat_key)
            index.setdefault(patient_dir.name, []).append((method_dir.name, mat_path))
    return index


def aggregate_split_from_log(
    log_path: Path,
    dataset_root: Path,
    label_encoder: LabelEncoder,
    mat_key: str,
) -> Split:
    feature_index = build_feature_index(dataset_root, mat_key)
    features: List[np.ndarray] = []
    labels: List[int] = []
    missing: List[str] = []
    duplicates: List[str] = []

    with log_path.open(encoding="utf-8") as f:
        for line_no, line in enumerate(f, 1):
            if not line.strip():
                continue
            patient_id = extract_patient_id(line)
            candidates = feature_index.get(patient_id)
            if not candidates:
                missing.append(f"{patient_id} (line {line_no})")
                continue
            if len(candidates) > 1:
                duplicates.append(patient_id)
            method, mat_path = candidates[0]
            feature_map = load_feature_map(mat_path, mat_key)
            column_means = feature_map.mean(axis=0)
            features.append(column_means.astype(np.float64))
            labels.append(label_encoder.encode(method))

    if missing:
        raise FileNotFoundError(f"Missing feature_map for patients: {', '.join(missing)}")
    if duplicates:
        print(f"[warn] Multiple entries found for patients (using first match): {', '.join(sorted(set(duplicates)))}")

    X = np.vstack(features)
    y = np.asarray(labels, dtype=np.float64)
    return Split(X=X, y=y)


def aggregate_for_data_type(cfg: DataTypeConfig) -> AggregatedSplits:
    encoder = LabelEncoder()
    splits: Dict[str, Split] = {}
    for split_name, log_file in cfg.log_files.items():
        log_path = cfg.log_root / log_file
        dataset_dir = cfg.feature_root / cfg.split_dirs[split_name]
        splits[split_name] = aggregate_split_from_log(log_path, dataset_dir, encoder, cfg.mat_key)
    encoder.assert_binary()
    train = splits["train"]
    test = splits["test"]
    test1 = splits["test1"]
    return AggregatedSplits(train=train, test=test, test1=test1)


def compute_normalization(train_X: np.ndarray) -> NormalizationStats:
    mean = train_X.mean(axis=0)
    std = train_X.std(axis=0)
    std[std == 0] = 1.0
    return NormalizationStats(mean=mean, std=std)


def apply_normalization(split: Split, stats: NormalizationStats) -> Split:
    X_norm = (split.X - stats.mean) / stats.std
    return Split(X=X_norm, y=split.y.copy())


def normalize_splits(splits: AggregatedSplits) -> Tuple[AggregatedSplits, NormalizationStats]:
    stats = compute_normalization(splits.train.X)
    normalized = AggregatedSplits(
        train=apply_normalization(splits.train, stats),
        test=apply_normalization(splits.test, stats),
        test1=apply_normalization(splits.test1, stats),
    )
    return normalized, stats


def mann_whitney_pvalues(X: np.ndarray, y: np.ndarray) -> np.ndarray:
    y_values = np.unique(y)
    if set(y_values.tolist()) != {0.0, 1.0}:
        raise ValueError(f"Mann-Whitney U-test expects binary labels 0/1, got {y_values}")
    pos = X[y == 1]
    neg = X[y == 0]
    if pos.shape[0] == 0 or neg.shape[0] == 0:
        raise ValueError("Both classes must have at least one sample for U-test.")
    p_values = []
    for i in range(X.shape[1]):
        _, p = mannwhitneyu(pos[:, i], neg[:, i], alternative="two-sided")
        p_values.append(p)
    return np.asarray(p_values)


def utest_select_features(
    splits: AggregatedSplits,
    alpha_train: float = 0.05,
    alpha_test: float = 0.05,
    fallback_topk: int = 5,
) -> UTestResult:
    p_train = mann_whitney_pvalues(splits.train.X, splits.train.y)
    p_test = mann_whitney_pvalues(splits.test.X, splits.test.y)
    p_test1 = mann_whitney_pvalues(splits.test1.X, splits.test1.y)

    mask = (p_train < alpha_train) & (p_test < alpha_test)
    indices = np.nonzero(mask)[0]
    if indices.size == 0:
        combined = p_train + p_test
        order = np.argsort(combined)
        k = min(fallback_topk, combined.size)
        indices = order[:k]

    def _filter(split: Split) -> Split:
        return Split(X=split.X[:, indices], y=split.y.copy())

    filtered = AggregatedSplits(train=_filter(splits.train), test=_filter(splits.test), test1=_filter(splits.test1))
    return UTestResult(
        p_train=p_train,
        p_test=p_test,
        p_test1=p_test1,
        selected_indices=indices,
        filtered=filtered,
    )


def sigmoid(x: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-x))


def _hidden_layer(X: np.ndarray, weights: np.ndarray, bias: np.ndarray) -> np.ndarray:
    return sigmoid(np.dot(X, weights) + bias)


def train_elm(X: np.ndarray, y: np.ndarray, n_hidden: int, rng: np.random.Generator) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    if n_hidden <= 0:
        raise ValueError("n_hidden must be positive.")
    weights = rng.normal(scale=1.0, size=(X.shape[1], n_hidden))
    bias = rng.normal(scale=1.0, size=(n_hidden,))
    H = _hidden_layer(X, weights, bias)
    lw = np.linalg.pinv(H) @ y
    return weights, bias, lw


def predict_elm(X: np.ndarray, weights: np.ndarray, bias: np.ndarray, lw: np.ndarray) -> np.ndarray:
    H = _hidden_layer(X, weights, bias)
    return H @ lw


def roc_curve_from_scores(y_true: np.ndarray, scores: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    order = np.argsort(-scores)
    y_true_sorted = y_true[order]
    scores_sorted = scores[order]
    thresholds = np.r_[np.inf, scores_sorted, -np.inf]

    pos = (y_true_sorted == 1).astype(np.int64)
    neg = 1 - pos
    tp = np.cumsum(pos)
    fp = np.cumsum(neg)
    tp = np.r_[0, tp]
    fp = np.r_[0, fp]

    tpr = tp / max(tp[-1], 1)
    fpr = fp / max(fp[-1], 1)
    return fpr, tpr, thresholds


def auc_score(y_true: np.ndarray, scores: np.ndarray) -> float:
    fpr, tpr, _ = roc_curve_from_scores(y_true, scores)
    diffs = fpr[1:] - fpr[:-1]
    avg_heights = (tpr[1:] + tpr[:-1]) * 0.5
    return float(np.sum(diffs * avg_heights))


def youden_index(y_true: np.ndarray, scores: np.ndarray) -> Tuple[float, float]:
    fpr, tpr, thresholds = roc_curve_from_scores(y_true, scores)
    diff = tpr - fpr
    idx = int(np.argmax(diff))
    return float(thresholds[idx]), float(auc_score(y_true, scores))


@dataclass
class ELMCandidate:
    hidden_nodes: int
    seed: int
    train_auc: float
    test_auc: float
    test1_auc: float
    train_cut: float
    test_cut: float
    test1_cut: float
    weights: np.ndarray
    bias: np.ndarray
    output_weights: np.ndarray


@dataclass
class PipelineResult:
    aggregated: AggregatedSplits
    normalized: AggregatedSplits
    normalization_stats: NormalizationStats
    utest: UTestResult
    elm_candidates: List[ELMCandidate]


def search_elm_models(
    splits: AggregatedSplits,
    n_trials: int = 200,
    hidden_range: Tuple[int, int] = (2, 5),
    auc_floor: float = 0.70,
    max_gap: float = 0.05,
    random_state: Optional[int] = None,
) -> List[ELMCandidate]:
    rng = np.random.default_rng(random_state)
    candidates: List[ELMCandidate] = []
    for trial in range(n_trials):
        n_hidden = rng.integers(hidden_range[0], hidden_range[1] + 1)
        weights, bias, lw = train_elm(splits.train.X, splits.train.y, int(n_hidden), rng)
        train_scores = predict_elm(splits.train.X, weights, bias, lw)
        test_scores = predict_elm(splits.test.X, weights, bias, lw)
        test1_scores = predict_elm(splits.test1.X, weights, bias, lw)

        train_cut, train_auc = youden_index(splits.train.y, train_scores)
        test_cut, test_auc = youden_index(splits.test.y, test_scores)
        test1_cut, test1_auc = youden_index(splits.test1.y, test1_scores)

        if (
            test_auc >= auc_floor
            and test1_auc >= auc_floor
            and train_auc >= test_auc
            and abs(test_auc - test1_auc) < max_gap
            and abs(train_auc - test_auc) < max_gap
            and abs(train_auc - test1_auc) < max_gap
            and train_auc >= 0.80
        ):
            candidates.append(
                ELMCandidate(
                    hidden_nodes=int(n_hidden),
                    seed=int(rng.integers(0, 2**32 - 1)),
                    train_auc=train_auc,
                    test_auc=test_auc,
                    test1_auc=test1_auc,
                    train_cut=train_cut,
                    test_cut=test_cut,
                    test1_cut=test1_cut,
                    weights=weights,
                    bias=bias,
                    output_weights=lw,
                )
            )
    candidates.sort(key=lambda c: c.test_auc, reverse=True)
    return candidates


def save_feature_map_mat(path: Path, splits: AggregatedSplits) -> None:
    payload = {
        "Xtrain": splits.train.X,
        "Ytrain": splits.train.y,
        "Xtest": splits.test.X,
        "Ytest": splits.test.y,
        "Xtest1": splits.test1.X,
        "Ytest1": splits.test1.y,
    }
    sio.savemat(path, payload)


def save_normalized_mat(path: Path, splits: AggregatedSplits, stats: NormalizationStats) -> None:
    payload = {
        "Xtrain": splits.train.X,
        "Ytrain": splits.train.y,
        "Xtest": splits.test.X,
        "Ytest": splits.test.y,
        "Xtest1": splits.test1.X,
        "Ytest1": splits.test1.y,
        "meanstd": np.vstack([stats.mean, stats.std]),
    }
    sio.savemat(path, payload)


def save_utest_mat(path: Path, utest: UTestResult) -> None:
    payload = {
        "Xtrain": utest.filtered.train.X,
        "Ytrain": utest.filtered.train.y,
        "Xtest": utest.filtered.test.X,
        "Ytest": utest.filtered.test.y,
        "Xtest1": utest.filtered.test1.X,
        "Ytest1": utest.filtered.test1.y,
        "trainp_data": utest.p_train,
        "testp_data": utest.p_test,
        "testp_data1": utest.p_test1,
        "index_utest": utest.selected_indices + 1,
    }
    sio.savemat(path, payload)


def run_for_data_type(
    cfg: DataTypeConfig,
    output_dir: Path,
    auc_floor: float = 0.70,
    max_gap: float = 0.05,
    alpha_train: float = 0.05,
    alpha_test: float = 0.05,
    elm_trials: int = 200,
    hidden_range: Tuple[int, int] = (2, 5),
    random_state: Optional[int] = None,
) -> PipelineResult:
    """完整流水线入口，返回中间结果并落盘 .mat 文件。"""
    output_dir.mkdir(parents=True, exist_ok=True)

    aggregated = aggregate_for_data_type(cfg)
    save_feature_map_mat(output_dir / f"feature_{cfg.name}_map.mat", aggregated)

    normalized, stats = normalize_splits(aggregated)
    save_normalized_mat(output_dir / f"feature_{cfg.name}_normalized.mat", normalized, stats)

    utest = utest_select_features(normalized, alpha_train=alpha_train, alpha_test=alpha_test)
    save_utest_mat(output_dir / f"feature_{cfg.name}_utest.mat", utest)

    elm_candidates = search_elm_models(
        utest.filtered,
        n_trials=elm_trials,
        hidden_range=hidden_range,
        auc_floor=auc_floor,
        max_gap=max_gap,
        random_state=random_state,
    )

    return PipelineResult(
        aggregated=aggregated,
        normalized=normalized,
        normalization_stats=stats,
        utest=utest,
        elm_candidates=elm_candidates,
    )


if __name__ == "__main__":
    print("请使用 src.elm.cli 入口运行完整流水线。")
