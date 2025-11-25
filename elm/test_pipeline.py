"""Synthetic tests for the pipeline using temporary data."""

import shutil
import tempfile
from pathlib import Path
import unittest

import numpy as np
import scipy.io as sio

from pipeline import AggregatedSplits, DataTypeConfig, aggregate_for_data_type, normalize_splits, search_elm_models, utest_select_features


def _make_patient(mat_root: Path, method: str, patient: str, n_slices: int, n_feats: int, mean_shift: float) -> None:
    """Create a dummy feature_map with a controllable mean shift per class."""
    patient_dir = mat_root / method / patient
    patient_dir.mkdir(parents=True, exist_ok=True)
    feature_map = np.random.normal(loc=mean_shift, scale=1.0, size=(n_slices, n_feats))
    sio.savemat(patient_dir / f"{patient}.mat", {"feature_map": feature_map})


def _write_log(log_path: Path, patients: list[str]) -> None:
    # Fake paths with patient ID as the penultimate segment to match extract_patient_id
    with log_path.open("w", encoding="utf-8") as f:
        for pid in patients:
            f.write(f"/some/path/{pid}/image.png\n")


def _build_dummy_data(base: Path) -> DataTypeConfig:
    feature_root = base / "features" / "CT" / "VGG16" / "feature_extract_3_1"
    log_root = base / "logs" / "CT"
    for split_dir in ("train_data", "test_data", "test1_data"):
        (feature_root / split_dir).mkdir(parents=True, exist_ok=True)
    log_root.mkdir(parents=True, exist_ok=True)

    # Two classes: neg (0), pos (1) with clear separation
    splits = {
        "train": (["neg_a", "neg_b", "neg_c"], ["pos_a", "pos_b", "pos_c"]),
        "test": (["neg_d", "neg_e"], ["pos_d", "pos_e"]),
        "test1": (["neg_f", "neg_g"], ["pos_f", "pos_g"]),
    }
    n_feats = 12
    for split_name, (negs, poss) in splits.items():
        split_dir = feature_root / f"{split_name}_data"
        # class 0
        for pid in negs:
            _make_patient(split_dir, "neg", pid, n_slices=3, n_feats=n_feats, mean_shift=0.0)
        # class 1
        for pid in poss:
            _make_patient(split_dir, "pos", pid, n_slices=3, n_feats=n_feats, mean_shift=4.0)
        _write_log(log_root / f"{split_name}_CT_log.txt", negs + poss)

    cfg = DataTypeConfig(
        name="CT",
        feature_root=feature_root,
        log_root=log_root,
        log_files={
            "train": "train_CT_log.txt",
            "test": "test_CT_log.txt",
            "test1": "test1_CT_log.txt",
        },
        split_dirs={"train": "train_data", "test": "test_data", "test1": "test1_data"},
        mat_key="feature_map",
    )
    return cfg


class PipelineSmokeTest(unittest.TestCase):
    def setUp(self) -> None:
        self.tempdir = Path(tempfile.mkdtemp(prefix="pipeline_test_"))
        np.random.seed(123)
        self.cfg = _build_dummy_data(self.tempdir)

    def tearDown(self) -> None:
        shutil.rmtree(self.tempdir, ignore_errors=True)

    def test_full_flow_on_synthetic_data(self) -> None:
        aggregated = aggregate_for_data_type(self.cfg)
        self.assertIsInstance(aggregated, AggregatedSplits)
        # Shapes
        self.assertEqual(aggregated.train.X.shape[0], 6)
        self.assertEqual(aggregated.test.X.shape[0], 4)
        self.assertEqual(aggregated.test1.X.shape[0], 4)
        # Labels must be binary {0,1}
        self.assertSetEqual(set(np.unique(aggregated.train.y)), {0.0, 1.0})

        normalized, stats = normalize_splits(aggregated)
        self.assertEqual(normalized.train.X.shape, aggregated.train.X.shape)
        # std should not contain zeros after smoothing
        self.assertTrue(np.all(stats.std > 0))

        # Relax thresholds for small synthetic sample sizes to reduce flakiness.
        utest = utest_select_features(normalized, alpha_train=0.2, alpha_test=0.2)
        self.assertGreater(utest.selected_indices.size, 0)
        self.assertEqual(utest.filtered.train.X.shape[1], utest.selected_indices.size)

        # Relax thresholds to avoid flakiness, but separation is strong so AUC should be high
        candidates = search_elm_models(
            utest.filtered,
            n_trials=200,
            hidden_range=(2, 5),
            auc_floor=0.60,
            max_gap=0.2,
            random_state=42,
        )
        self.assertGreater(len(candidates), 0, "Should find at least one good ELM candidate on separable data.")


if __name__ == "__main__":
    unittest.main()
