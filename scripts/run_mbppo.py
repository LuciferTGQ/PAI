from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.evaluation.mbppo_experiment import run_mbppo_experiment


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Train and evaluate NeoRL-style MB-PPO with/without behavior KL"
    )
    parser.add_argument("--config", default="configs/mbppo_ib_m1000_gru.yaml")
    args = parser.parse_args()
    results = run_mbppo_experiment(args.config)
    for method, section in results["methods"].items():
        print(method, section["summary"])


if __name__ == "__main__":
    main()
