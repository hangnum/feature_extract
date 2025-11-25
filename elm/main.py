"""
兼容入口：调用 src.elm.cli.main，建议改用 scripts/run_elm.py。
"""

import sys
from pathlib import Path

project_root = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(project_root))

from src.elm.cli import main


if __name__ == "__main__":
    main()
