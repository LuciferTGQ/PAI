from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.evaluation.main_system_matrix import run_main_system_matrix


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the frozen 3x4 IB main system matrix")
    parser.add_argument("--config", default="configs/main_system_matrix.yaml")
    args = parser.parse_args()
    run_main_system_matrix(args.config)


if __name__ == "__main__":
    main()
