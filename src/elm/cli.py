"""
ELM 命令行入口

支持基于 JSON 配置完成特征聚合 + U 检验 + ELM 搜索，并输出 .mat 文件。
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from .config import build_datatype_config, build_elm_config, load_json_config
from .pipeline import run_for_data_type


def _default_config_path() -> Path:
    """返回优先可用的配置文件路径。"""
    project_root = Path(__file__).resolve().parents[2]
    candidates = [
        project_root / "config" / "elm_config.json",
        project_root / "config" / "elm_config.example.json",
    ]
    for path in candidates:
        if path.exists():
            return path
    return candidates[0]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="运行特征聚合 + U 检验 + ELM 流水线。")
    parser.add_argument("--config", type=Path, default=None, help="JSON 配置路径，默认尝试 config/elm_config.json。")
    parser.add_argument("--data-type", required=True, help="配置中的 data_types 键，例如 CT/BL。")
    parser.add_argument("--output", type=Path, default=None, help="保存生成 .mat 文件的目录（默认 config.experiment.output_dir 或 outputs/elm）。")
    parser.add_argument("--n-trials", type=int, help="ELM 搜索次数覆盖。")
    parser.add_argument("--hidden-min", type=int, help="ELM 隐藏节点下界覆盖。")
    parser.add_argument("--hidden-max", type=int, help="ELM 隐藏节点上界覆盖。")
    parser.add_argument("--auc-floor", type=float, help="最小 AUC 阈值覆盖。")
    parser.add_argument("--max-gap", type=float, help="不同切分间 AUC 最大差覆盖。")
    parser.add_argument("--random-state", type=int, help="随机种子。")
    parser.add_argument("--alpha-train", type=float, help="训练集 U 检验阈值。")
    parser.add_argument("--alpha-test", type=float, help="验证集 U 检验阈值。")
    return parser.parse_args()


def main(args: argparse.Namespace | None = None) -> None:
    if args is None:
        args = parse_args()

    config_path = args.config or _default_config_path()
    if not config_path.exists():
        raise FileNotFoundError(f"未找到配置文件: {config_path}")

    cfg_dict = load_json_config(config_path)
    dt_cfg = build_datatype_config(cfg_dict, args.data_type)
    elm_cfg = build_elm_config(cfg_dict)

    if args.n_trials is not None:
        elm_cfg.n_trials = args.n_trials
    if args.hidden_min is not None or args.hidden_max is not None:
        lo = args.hidden_min if args.hidden_min is not None else elm_cfg.hidden_range[0]
        hi = args.hidden_max if args.hidden_max is not None else elm_cfg.hidden_range[1]
        elm_cfg.hidden_range = (lo, hi)
    if args.auc_floor is not None:
        elm_cfg.auc_floor = args.auc_floor
    if args.max_gap is not None:
        elm_cfg.max_gap = args.max_gap
    if args.random_state is not None:
        elm_cfg.random_state = args.random_state
    alpha_train = args.alpha_train if args.alpha_train is not None else 0.05
    alpha_test = args.alpha_test if args.alpha_test is not None else 0.05

    output_dir = args.output
    if output_dir is None:
        project_root = Path(__file__).resolve().parents[2]
        output_dir = project_root / "outputs" / "elm"

    result = run_for_data_type(
        dt_cfg,
        output_dir=output_dir,
        auc_floor=elm_cfg.auc_floor,
        max_gap=elm_cfg.max_gap,
        alpha_train=alpha_train,
        alpha_test=alpha_test,
        elm_trials=elm_cfg.n_trials,
        hidden_range=elm_cfg.hidden_range,
        random_state=elm_cfg.random_state,
    )

    print(
        f"数据类型 {dt_cfg.name} 处理完成；筛选特征 {len(result.utest.selected_indices)} 个；"
        f"满足筛选条件的 ELM 候选数 {len(result.elm_candidates)}。"
    )


if __name__ == "__main__":
    main()
