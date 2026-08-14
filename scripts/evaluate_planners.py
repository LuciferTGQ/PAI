from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.evaluation.planner_comparison import evaluate_planners


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate CEM, iCEM, and MPPI on development seeds")
    parser.add_argument("--config", default="configs/planner_ib_m1000_gru.yaml")
    args = parser.parse_args()
    result = evaluate_planners(args.config)
    for method, section in result["methods"].items():
        print(method, section["summary"])


if __name__ == "__main__":
    main()
