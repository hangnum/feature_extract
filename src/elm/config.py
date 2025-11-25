"""
ELM 配置读取工具（JSON）

提供从 JSON 构建 DataTypeConfig 与 ELM 搜索配置的能力。
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

from .pipeline import DataTypeConfig


@dataclass
class ElmConfig:
    n_trials: int = 200
    hidden_range: Tuple[int, int] = (2, 5)
    auc_floor: float = 0.70
    max_gap: float = 0.05
    random_state: Optional[int] = None


def _require(obj: Dict[str, Any], key: str) -> Any:
    if key not in obj:
        raise KeyError(f"Missing required config key: {key}")
    return obj[key]


def load_json_config(path: Path) -> Dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def build_datatype_config(config: Dict[str, Any], name: str) -> DataTypeConfig:
    data_types = _require(config, "data_types")
    if name not in data_types:
        raise KeyError(f"Data type '{name}' not found in config.")
    cfg = data_types[name]
    return DataTypeConfig(
        name=name,
        feature_root=Path(_require(cfg, "feature_root")),
        log_root=Path(_require(cfg, "log_root")),
        log_files=dict(_require(cfg, "log_files")),
        split_dirs=dict(_require(cfg, "split_dirs")),
        mat_key=cfg.get("mat_key", "feature_map"),
    )


def build_elm_config(config: Dict[str, Any]) -> ElmConfig:
    elm_cfg = config.get("elm", {})
    hidden_range = tuple(elm_cfg.get("hidden_range", (2, 5)))
    if len(hidden_range) != 2:
        raise ValueError("elm.hidden_range must be a length-2 list/tuple.")
    return ElmConfig(
        n_trials=int(elm_cfg.get("n_trials", 200)),
        hidden_range=(int(hidden_range[0]), int(hidden_range[1])),
        auc_floor=float(elm_cfg.get("auc_floor", 0.70)),
        max_gap=float(elm_cfg.get("max_gap", 0.05)),
        random_state=elm_cfg.get("random_state"),
    )
