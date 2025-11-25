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
    logger.warning("Old fusion pipeline has been removed; please use CMTA-based fusion instead.")


def handle_cmta(args, config: Config, logger) -> None:
    """处理CMTA训练"""
    # 加载CMTA特定配置
    cmta_config_path = args.config or str(project_root / 'config' / 'best_hparams' / 'cmta.yaml')
    cmta_config = load_config(Path(cmta_config_path))

    cmd = [
        sys.executable,
        str(project_root / 'scripts' / 'train_cmta.py'),
        '--config', cmta_config_path
    ]

    # 数据目录
    data_dir = args.data_dir or config.data.root_dir
    cmd += ['--data_dir', data_dir]

    # 模态配置
    modalities = args.modalities or cmta_config.data.modalities
    if modalities:
        cmd += ['--modalities'] + modalities

    # 模型配置
    if args.model_size:
        cmd += ['--model_size', args.model_size]
    elif hasattr(cmta_config.model.cmta, 'model_size'):
        cmd += ['--model_size', cmta_config.model.cmta.model_size]

    # 训练参数
    if args.batch_size is not None:
        cmd += ['--batch_size', str(args.batch_size)]
    elif hasattr(cmta_config.training, 'batch_size'):
        cmd += ['--batch_size', str(cmta_config.training.batch_size)]

    if args.epochs is not None:
        cmd += ['--epochs', str(args.epochs)]
    elif hasattr(cmta_config.training, 'epochs'):
        cmd += ['--epochs', str(cmta_config.training.epochs)]

    if args.learning_rate is not None:
        cmd += ['--learning_rate', str(args.learning_rate)]
    elif hasattr(cmta_config.training, 'learning_rate'):
        cmd += ['--learning_rate', str(cmta_config.training.learning_rate)]

    if args.alpha is not None:
        cmd += ['--alpha', str(args.alpha)]
    elif hasattr(cmta_config.training.cmta, 'alpha'):
        cmd += ['--alpha', str(cmta_config.training.cmta.alpha)]

    if args.beta is not None:
        cmd += ['--beta', str(args.beta)]
    elif hasattr(cmta_config.training.cmta, 'beta'):
        cmd += ['--beta', str(cmta_config.training.cmta.beta)]

    # 其他参数
    if args.seed is not None:
        cmd += ['--seed', str(args.seed)]
    elif hasattr(cmta_config.experiment, 'seed'):
        cmd += ['--seed', str(cmta_config.experiment.seed)]

    device = args.device or cmta_config.training.device
    cmd += ['--device', device]

    if args.resume:
        cmd += ['--resume', args.resume]

    # 输出目录
    output_dir = getattr(cmta_config.experiment, 'output_dir', './outputs/cmta')
    cmd += ['--output_dir', output_dir]

    run_subprocess(cmd, logger)


def handle_elm(args, logger) -> None:
    """运行 ELM 流水线"""
    cmd = [
        sys.executable,
        str(project_root / 'scripts' / 'run_elm.py'),
        '--data-type', args.data_type
    ]

    if args.elm_config:
        cmd += ['--config', args.elm_config]
    else:
        cmd += ['--config', str(project_root / 'config' / 'elm_config.json')]

    if args.output is not None:
        cmd += ['--output', args.output]
    if args.n_trials is not None:
        cmd += ['--n-trials', str(args.n_trials)]
    if args.hidden_min is not None:
        cmd += ['--hidden-min', str(args.hidden_min)]
    if args.hidden_max is not None:
        cmd += ['--hidden-max', str(args.hidden_max)]
    if args.auc_floor is not None:
        cmd += ['--auc-floor', str(args.auc_floor)]
    if args.max_gap is not None:
        cmd += ['--max-gap', str(args.max_gap)]
    if args.random_state is not None:
        cmd += ['--random-state', str(args.random_state)]
    if args.alpha_train is not None:
        cmd += ['--alpha-train', str(args.alpha_train)]
    if args.alpha_test is not None:
        cmd += ['--alpha-test', str(args.alpha_test)]

    run_subprocess(cmd, logger)


def handle_visualize(args, logger) -> None:
    """绘制训练曲线与指标"""
    cmd = [
        sys.executable,
        str(project_root / 'scripts' / 'visualize_results.py'),
        '--history_csv', args.history_csv
    ]
    if args.output_dir is not None:
        cmd += ['--output_dir', args.output_dir]
    run_subprocess(cmd, logger)


def handle_elm(args, logger) -> None:
    """运行 ELM 流水线"""
    cmd = [
        sys.executable,
        str(project_root / 'scripts' / 'run_elm.py'),
        '--data-type', args.data_type
    ]

    if args.elm_config:
        cmd += ['--config', args.elm_config]
    else:
        cmd += ['--config', str(project_root / 'config' / 'elm_config.json')]

    if args.output is not None:
        cmd += ['--output', args.output]
    if args.n_trials is not None:
        cmd += ['--n-trials', str(args.n_trials)]
    if args.hidden_min is not None:
        cmd += ['--hidden-min', str(args.hidden_min)]
    if args.hidden_max is not None:
        cmd += ['--hidden-max', str(args.hidden_max)]
    if args.auc_floor is not None:
        cmd += ['--auc-floor', str(args.auc_floor)]
    if args.max_gap is not None:
        cmd += ['--max-gap', str(args.max_gap)]
    if args.random_state is not None:
        cmd += ['--random-state', str(args.random_state)]
    if args.alpha_train is not None:
        cmd += ['--alpha-train', str(args.alpha_train)]
    if args.alpha_test is not None:
        cmd += ['--alpha-test', str(args.alpha_test)]

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

    # cmta
    cmta_parser = subparsers.add_parser('cmta', help='Train CMTA multimodal fusion model')
    cmta_parser.add_argument('--data_dir', type=str, default=None, help='CMTA data directory')
    cmta_parser.add_argument('--modalities', nargs='+', default=None, help='Modalities to use')
    cmta_parser.add_argument('--model_size', type=str, default=None, choices=['small', 'large'], help='Model size')
    cmta_parser.add_argument('--batch_size', type=int, default=None, help='Batch size')
    cmta_parser.add_argument('--epochs', type=int, default=None, help='Training epochs')
    cmta_parser.add_argument('--learning_rate', type=float, default=None, help='Learning rate')
    cmta_parser.add_argument('--alpha', type=float, default=None, help='Cohort loss weight')
    cmta_parser.add_argument('--beta', type=float, default=None, help='Auxiliary loss weight')
    cmta_parser.add_argument('--seed', type=int, default=None, help='Random seed')
    cmta_parser.add_argument('--device', type=str, default=None, help='Training device')
    cmta_parser.add_argument('--resume', type=str, default=None, help='Resume from checkpoint')
    cmta_parser.add_argument('--config', type=str, default=None, help='CMTA specific config file')

    # elm
    elm_parser = subparsers.add_parser('elm', help='Run ELM feature aggregation + U-test + ELM search')
    elm_parser.add_argument('--data_type', type=str, required=True, help='Data type key in elm config (e.g., CT/BL)')
    elm_parser.add_argument('--elm_config', type=str, default=None, help='Path to elm JSON config')
    elm_parser.add_argument('--output', type=str, default=None, help='Directory to save .mat outputs')
    elm_parser.add_argument('--n_trials', type=int, default=None, help='Override ELM search trials')
    elm_parser.add_argument('--hidden_min', type=int, default=None, help='Override hidden nodes lower bound')
    elm_parser.add_argument('--hidden_max', type=int, default=None, help='Override hidden nodes upper bound')
    elm_parser.add_argument('--auc_floor', type=float, default=None, help='Minimum AUC threshold')
    elm_parser.add_argument('--max_gap', type=float, default=None, help='Max AUC gap between splits')
    elm_parser.add_argument('--random_state', type=int, default=None, help='Random seed for ELM search')
    elm_parser.add_argument('--alpha_train', type=float, default=None, help='U-test p-value threshold on train')
    elm_parser.add_argument('--alpha_test', type=float, default=None, help='U-test p-value threshold on test')

    # visualize
    viz_parser = subparsers.add_parser('visualize', help='Plot training metrics from CSV history')
    viz_parser.add_argument('--history_csv', type=str, required=True, help='Path to training history CSV')
    viz_parser.add_argument('--output_dir', type=str, default=None, help='Output directory for plots')

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
    elif args.command == 'cmta':
        handle_cmta(args, config, logger)
    elif args.command == 'elm':
        handle_elm(args, logger)
    elif args.command == 'visualize':
        handle_visualize(args, logger)


if __name__ == '__main__':
    main()
