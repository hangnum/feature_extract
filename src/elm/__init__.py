"""
ELM 流水线包

提供特征聚合、标准化、U 检验筛选与 ELM 搜索能力。
"""

from .config import ElmConfig, build_datatype_config, build_elm_config, load_json_config
from .pipeline import (
    AggregatedSplits,
    DataTypeConfig,
    ELMCandidate,
    NormalizationStats,
    PipelineResult,
    Split,
    UTestResult,
    aggregate_for_data_type,
    normalize_splits,
    run_for_data_type,
    search_elm_models,
    utest_select_features,
)
