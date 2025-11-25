"""Generate a synthetic dataset locally and run the full pipeline.

This avoids any external data/downloads. It is useful for smoke-testing in a
fresh conda env (e.g., `py9`) after installing numpy/scipy.
"""

from __future__ import annotations

import argparse
import shutil
import tempfile
from pathlib import Path
from typing import Tuple

import numpy as np
import scipy.io as sio

from pipeline import DataTypeConfig, run_for_data_type


def _make_patient(mat_root: Path, method: str, patient: str, n_slices: int, n_feats: int, mean_shift: float, rng: np.random.Generator) -> None:
    patient_dir = mat_root / method / patient
    patient_dir.mkdir(parents=True, exist_ok=True)
    feature_map = rng.normal(loc=mean_shift, scale=1.0, size=(n_slices, n_feats))
    sio.savemat(patient_dir / f"{patient}.mat", {"feature_map": feature_map})


def _write_log(log_path: Path, patients: list[str]) -> None:
    with log_path.open("w", encoding="utf-8") as f:
        for pid in patients:
            # fake path; extract_patient_id uses the penultimate segment
            f.write(f"/dummy/{pid}/image.png\n")


def build_synthetic_cfg(base: Path, n_feats: int = 50, mean_shift: float = 3.0) -> DataTypeConfig:
    feature_root = base / "features" / "SYN" / "VGG16" / "feature_extract_3_1"
    log_root = base / "logs" / "SYN"
    for split_dir in ("train_data", "test_data", "test1_data"):
        (feature_root / split_dir).mkdir(parents=True, exist_ok=True)
    log_root.mkdir(parents=True, exist_ok=True)

    # simple balanced splits
    splits = {
        "train": (["neg_a", "neg_b", "neg_c", "neg_d"], ["pos_a", "pos_b", "pos_c", "pos_d"]),
        "test": (["neg_e", "neg_f"], ["pos_e", "pos_f"]),
        "test1": (["neg_g", "neg_h"], ["pos_g", "pos_h"]),
    }
    rng = np.random.default_rng(123)
    for split_name, (negs, poss) in splits.items():
        split_dir = feature_root / f"{split_name}_data"
        for pid in negs:
            _make_patient(split_dir, "neg", pid, n_slices=4, n_feats=n_feats, mean_shift=0.0, rng=rng)
        for pid in poss:
            _make_patient(split_dir, "pos", pid, n_slices=4, n_feats=n_feats, mean_shift=mean_shift, rng=rng)
        _write_log(log_root / f"{split_name}_SYN_log.txt", negs + poss)

    return DataTypeConfig(
        name="SYN",
        feature_root=feature_root,
        log_root=log_root,
        log_files={
            "train": "train_SYN_log.txt",
            "test": "test_SYN_log.txt",
            "test1": "test1_SYN_log.txt",
        },
        split_dirs={"train": "train_data", "test": "test_data", "test1": "test1_data"},
        mat_key="feature_map",
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run pipeline on synthetic data for smoke testing.")
    parser.add_argument("--output", type=Path, default=Path("outputs/synthetic"), help="Where to write .mat outputs.")
    parser.add_argument("--n-feats", type=int, default=50, help="Feature dimension.")
    parser.add_argument("--mean-shift", type=float, default=3.0, help="Class separation for synthetic data.")
    parser.add_argument("--n-trials", type=int, default=200, help="ELM search trials.")
    parser.add_argument("--hidden-min", type=int, default=2, help="ELM hidden layer lower bound.")
    parser.add_argument("--hidden-max", type=int, default=5, help="ELM hidden layer upper bound.")
    parser.add_argument("--auc-floor", type=float, default=0.7, help="Minimum AUC threshold.")
    parser.add_argument("--max-gap", type=float, default=0.1, help="Max AUC gap between splits.")
    parser.add_argument("--random-state", type=int, default=42, help="Random seed.")
    parser.add_argument("--alpha-train", type=float, default=0.2, help="U-test p-value threshold on train.")
    parser.add_argument("--alpha-test", type=float, default=0.2, help="U-test p-value threshold on test.")
    parser.add_argument("--keep-temp", action="store_true", help="Keep the synthetic data directory for inspection.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    tempdir = Path(tempfile.mkdtemp(prefix="synthetic_pipeline_"))
    cfg = build_synthetic_cfg(tempdir, n_feats=args.n_feats, mean_shift=args.mean_shift)

    result = run_for_data_type(
        cfg,
        output_dir=args.output,
        auc_floor=args.auc_floor,
        max_gap=args.max_gap,
        alpha_train=args.alpha_train,
        alpha_test=args.alpha_test,
        elm_trials=args.n_trials,
        hidden_range=(args.hidden_min, args.hidden_max),
        random_state=args.random_state,
    )

    print(f"Synthetic data generated under: {tempdir}")
    print(f"Outputs written to: {args.output}")
    print(f"Selected features: {len(result.utest.selected_indices)}")
    print(f"ELM candidates found: {len(result.elm_candidates)}")

    if not args.keep_temp:
        shutil.rmtree(tempdir, ignore_errors=True)


if __name__ == "__main__":
    main()
