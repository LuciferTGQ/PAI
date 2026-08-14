from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.evaluation.mppi_stability import evaluate_mppi_stability


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate the diagnosed MPPI stability correction")
    parser.add_argument("--config", default="configs/mppi_stable_ib_m1000_gru.yaml")
    args = parser.parse_args()
    result = evaluate_mppi_stability(args.config)
    print(result["summary"])


if __name__ == "__main__":
    main()
