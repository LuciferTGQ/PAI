from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.evaluation.strategy_heldout import evaluate_heldout_strategies


def main() -> None:
    parser = argparse.ArgumentParser(description="One-time M-1000 held-out strategy report")
    parser.add_argument("--config", default="configs/strategy_heldout_ib_m1000_gru.yaml")
    args = parser.parse_args()
    results = evaluate_heldout_strategies(args.config)
    for method, section in results["methods"].items():
        print(method, section["summary"])


if __name__ == "__main__":
    main()
