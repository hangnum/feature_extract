"""
Feature fusion script.

Pools per-modality patient features, concatenates across modalities, and trains
an L2-regularized logistic regression classifier.
"""

import sys
from pathlib import Path
import argparse
import pandas as pd

# Add project root
project_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(project_root))

from src.feature_extraction.fusion import (
    build_fusion_matrix,
    train_l2_classifier,
    evaluate_classifier
)
from src.utils.logger import setup_logger
from src.utils.seed import set_seed


def main():
    parser = argparse.ArgumentParser(description="Multimodal feature fusion with L2 classifier")
    parser.add_argument('--feature_dir', type=str, default=None, help='Directory containing feature splits')
    parser.add_argument('--modalities', nargs='+', default=['A', 'P'], help='Modalities to fuse')
    parser.add_argument('--use_aligned', action='store_true', help='Prefer aligned feature CSVs')
    parser.add_argument('--C', type=float, default=1.0, help='Inverse of L2 regularization strength')
    parser.add_argument('--max_iter', type=int, default=1000, help='Max iterations for logistic regression')
    parser.add_argument('--random_state', type=int, default=42, help='Random seed')
    parser.add_argument('--output_dir', type=str, default=None, help='Output directory for metrics/predictions')
    args = parser.parse_args()

    set_seed(args.random_state)

    if args.feature_dir is None:
        args.feature_dir = str(project_root / 'data' / 'features')
    if args.output_dir is None:
        args.output_dir = str(project_root / 'outputs' / 'feature_extract' / 'fusion')

    log_dir = Path(project_root) / 'outputs' / 'feature_extract' / 'logs'
    logger = setup_logger(
        name="feature_extract",
        log_dir=log_dir,
        console=True
    )

    feature_dir = Path(args.feature_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    logger.info("=" * 60)
    logger.info("Feature Fusion (L2 Logistic Regression)")
    logger.info("=" * 60)
    logger.info(f"Modalities: {args.modalities}")
    logger.info(f"Feature dir: {feature_dir}")
    logger.info(f"Use aligned CSVs: {args.use_aligned}")
    logger.info(f"C: {args.C} | max_iter: {args.max_iter} | seed: {args.random_state}")

    # Load train split and fit classifier
    train_dir = feature_dir / 'train'
    X_train, y_train, train_ids = build_fusion_matrix(
        modalities=args.modalities,
        csv_dir=train_dir,
        use_aligned=args.use_aligned
    )
    logger.info(f"Training samples: {len(train_ids)} | Feature dim: {X_train.shape[1]}")

    model = train_l2_classifier(
        X_train=X_train,
        y_train=y_train,
        C=args.C,
        max_iter=args.max_iter,
        random_state=args.random_state
    )

    all_metrics = []

    def evaluate_split(split_name: str, X, y, patient_ids):
        preds, probs, metrics = evaluate_classifier(model, X, y)
        metrics['split'] = split_name
        all_metrics.append(metrics)

        preds_df = pd.DataFrame({
            'patient_id': patient_ids,
            'label': y,
            'prob': probs,
            'pred': preds
        })
        preds_path = output_dir / f"fusion_{split_name}_preds.csv"
        preds_df.to_csv(preds_path, index=False)
        logger.info(
            f"[{split_name}] "
            f"AUC: {metrics['auc']:.4f} | Acc: {metrics['accuracy']:.4f} | "
            f"Sensitivity: {metrics['sensitivity']:.4f} | Specificity: {metrics['specificity']:.4f} "
            f"| saved preds: {preds_path}"
        )

    # Evaluate train split
    evaluate_split('train', X_train, y_train, train_ids)

    # Optional val/test splits
    for split in ['val', 'test']:
        split_dir = feature_dir / split
        if not split_dir.exists():
            logger.warning(f"Skip {split}: directory not found ({split_dir})")
            continue
        try:
            X_split, y_split, split_ids = build_fusion_matrix(
                modalities=args.modalities,
                csv_dir=split_dir,
                use_aligned=args.use_aligned
            )
        except FileNotFoundError as e:
            logger.warning(f"Skip {split}: {e}")
            continue
        except ValueError as e:
            logger.warning(f"Skip {split}: {e}")
            continue

        evaluate_split(split, X_split, y_split, split_ids)

    metrics_path = output_dir / "fusion_metrics.csv"
    pd.DataFrame(all_metrics).to_csv(metrics_path, index=False)
    logger.info(f"Metrics saved to {metrics_path}")
    logger.info("Fusion completed.")


if __name__ == '__main__':
    main()
