"""
ELM 流水线运行脚本

基于 JSON 配置完成特征聚合、U 检验筛选与 ELM 搜索。
"""

import sys
from pathlib import Path

# 将项目根目录加入路径，便于导入 src 模块
project_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(project_root))

from src.elm.cli import main


if __name__ == "__main__":
    main()
