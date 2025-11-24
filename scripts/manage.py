"""
Management CLI with subcommands to run preprocess, train, extract, and fuse.

All defaults come from the config file, with CLI overrides when provided.
"""

import argparse
import subprocess
import sys
from pathlib import Path
from typing import List

project_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(project_root))

from src.utils.config import Config
from src.utils.logger import setup_logger
from src.utils.seed import set_seed
from src.data.data_parser import (
    parse_data_directory,
    validate_patient_data,
    get_statistics,
    print_statistics
)
from src.data.data_splitter import generate_all_splits


def load_config(config_path: Path) -> Config:
    if config_path.exists():
        return Config.from_yaml(str(config_path))
    return Config()


def run_subprocess(cmd: List[str], logger) -> None:
    logger.info(f"Run: {' '.join(cmd)}")
    result = subprocess.run(cmd)
    if result.returncode != 0:
        raise SystemExit(result.returncode)


def handle_preprocess(args, config: Config, logger) -> None:
    root_dir = args.root_dir or config.data.root_dir
    modalities = args.modalities or config.data.modalities
    output_dir = args.output_dir or str(project_root / 'data' / 'splits')
    train_ratio = args.train_ratio if args.train_ratio is not None else config.data.train_ratio
    seed = args.seed if args.seed is not None else config.experiment.seed
    log_dir = args.log_dir or str(project_root / 'outputs' / 'feature_extract' / 'logs')

    logger.info("=== Preprocess ===")
    logger.info(f"root_dir={root_dir}")
    logger.info(f"modalities={modalities}")
    logger.info(f"output_dir={output_dir}")
    logger.info(f"train_ratio={train_ratio}")
    logger.info(f"seed={seed}")

    set_seed(seed)

    patient_dict = parse_data_directory(root_dir=root_dir, modalities=modalities)
    valid_patients, filtered_patients = validate_patient_data(
        patient_dict=patient_dict,
        required_modalities=modalities
    )

    if filtered_patients:
        logger.info(f"Filtered patients (first 10): {filtered_patients[:10]}")
        if len(filtered_patients) > 10:
            logger.info(f"... and {len(filtered_patients) - 10} more")

    stats = get_statistics(valid_patients)
    print_statistics(stats)

    generate_all_splits(
        patient_dict=valid_patients,
        modalities=modalities,
        output_dir=output_dir,
        train_ratio=train_ratio,
        random_state=seed
    )


def handle_train(args, logger) -> None:
    cmd = [
        sys.executable,
        str(project_root / 'scripts' / 'train.py'),
        '--modality', args.modality,
        '--config', args.config
    ]

    if args.model:
        cmd += ['--model', args.model]
    if args.epochs is not None:
        cmd += ['--epochs', str(args.epochs)]
    if args.batch_size is not None:
        cmd += ['--batch_size', str(args.batch_size)]
    if args.learning_rate is not None:
        cmd += ['--learning_rate', str(args.learning_rate)]
    if args.loss_type:
        cmd += ['--loss_type', args.loss_type]
    if args.device:
        cmd += ['--device', args.device]
    if args.disable_early_stop:
        cmd.append('--disable_early_stop')
    if args.resume:
        cmd.append('--resume')

    run_subprocess(cmd, logger)


def handle_extract(args, config: Config, logger) -> None:
    default_checkpoint = Path(config.experiment.output_dir) / 'checkpoints' / 'best_model.pth'
    cmd = [
        sys.executable,
        str(project_root / 'scripts' / 'extract_features.py'),
        '--modality', args.modality,
        '--model', args.model or config.model.name,
        '--checkpoint', str(args.checkpoint or default_checkpoint),
        '--device', args.device or config.training.device
    ]

    if args.output_dir:
        cmd += ['--output_dir', args.output_dir]
    if args.batch_size is not None:
        cmd += ['--batch_size', str(args.batch_size)]
    if args.align:
        cmd.append('--align')

    run_subprocess(cmd, logger)


def handle_fuse(args, config: Config, logger) -> None:
    cmd = [
        sys.executable,
        str(project_root / 'scripts' / 'fuse_features.py'),
    ]

    feature_dir = args.feature_dir or str(project_root / 'data' / 'features')
    output_dir = args.output_dir or str(project_root / 'outputs' / 'feature_extract' / 'fusion')
    cmd += ['--feature_dir', feature_dir, '--output_dir', output_dir]

    modalities = args.modalities or config.data.modalities
    if modalities:
        cmd += ['--modalities'] + modalities
    if args.use_aligned:
        cmd.append('--use_aligned')
    if args.C is not None:
        cmd += ['--C', str(args.C)]
    if args.max_iter is not None:
        cmd += ['--max_iter', str(args.max_iter)]
    if args.random_state is not None:
        cmd += ['--random_state', str(args.random_state)]

    run_subprocess(cmd, logger)


def main():
    parser = argparse.ArgumentParser(description="Project management CLI")
    parser.add_argument(
        '--config',
        type=str,
        default=str(project_root / 'config' / 'default_config.yaml'),
        help='Config file path (defaults to config/default_config.yaml)'
    )
    subparsers = parser.add_subparsers(dest='command', required=True)

    # preprocess
    preprocess_parser = subparsers.add_parser('preprocess', help='Parse data and generate splits')
    preprocess_parser.add_argument('--root_dir', type=str, default=None, help='Data root directory')
    preprocess_parser.add_argument('--modalities', nargs='+', default=None, help='Modalities to include')
    preprocess_parser.add_argument('--output_dir', type=str, default=None, help='Output directory for splits')
    preprocess_parser.add_argument('--train_ratio', type=float, default=None, help='Train ratio for JM hospital')
    preprocess_parser.add_argument('--seed', type=int, default=None, help='Random seed')
    preprocess_parser.add_argument('--log_dir', type=str, default=None, help='Log directory')

    # train
    train_parser = subparsers.add_parser('train', help='Train a single modality model')
    train_parser.add_argument('--modality', type=str, default='A', choices=['A', 'P'], help='Modality to train')
    train_parser.add_argument('--model', type=str, default=None, help='Model name (override config)')
    train_parser.add_argument('--epochs', type=int, default=None, help='Override epochs')
    train_parser.add_argument('--batch_size', type=int, default=None, help='Override batch size')
    train_parser.add_argument('--learning_rate', type=float, default=None, help='Override learning rate')
    train_parser.add_argument('--loss_type', type=str, default=None, help='Override loss type')
    train_parser.add_argument('--device', type=str, default=None, help='Device')
    train_parser.add_argument('--disable_early_stop', action='store_true', help='Disable early stopping')
    train_parser.add_argument('--resume', action='store_true', help='Resume from checkpoint')

    # extract
    extract_parser = subparsers.add_parser('extract', help='Extract patient-level features')
    extract_parser.add_argument('--modality', type=str, default='A', choices=['A', 'P'], help='Modality to extract')
    extract_parser.add_argument('--model', type=str, default=None, help='Model name for feature extractor')
    extract_parser.add_argument('--checkpoint', type=str, default=None, help='Checkpoint path')
    extract_parser.add_argument('--output_dir', type=str, default=None, help='Feature output directory')
    extract_parser.add_argument('--batch_size', type=int, default=None, help='Batch size')
    extract_parser.add_argument('--device', type=str, default=None, help='Device')
    extract_parser.add_argument('--align', action='store_true', help='Align multimodal features after extraction')

    # fuse
    fuse_parser = subparsers.add_parser('fuse', help='Fuse multimodal features and train L2 classifier')
    fuse_parser.add_argument('--feature_dir', type=str, default=None, help='Directory containing feature splits')
    fuse_parser.add_argument('--modalities', nargs='+', default=None, help='Modalities to fuse')
    fuse_parser.add_argument('--use_aligned', action='store_true', help='Use aligned feature CSVs if available')
    fuse_parser.add_argument('--C', type=float, default=None, help='Inverse regularization strength')
    fuse_parser.add_argument('--max_iter', type=int, default=None, help='Max iterations for logistic regression')
    fuse_parser.add_argument('--random_state', type=int, default=None, help='Random seed')
    fuse_parser.add_argument('--output_dir', type=str, default=None, help='Output directory for fusion outputs')

    args = parser.parse_args()

    config_path = Path(args.config)
    config = load_config(config_path)

    log_dir = Path(project_root) / 'outputs' / 'feature_extract' / 'logs'
    logger = setup_logger(
        name="feature_extract",
        log_dir=log_dir,
        console=True
    )

    if args.command == 'preprocess':
        handle_preprocess(args, config, logger)
    elif args.command == 'train':
        handle_train(args, logger)
    elif args.command == 'extract':
        handle_extract(args, config, logger)
    elif args.command == 'fuse':
        handle_fuse(args, config, logger)


if __name__ == '__main__':
    main()
