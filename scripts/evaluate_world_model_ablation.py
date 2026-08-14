from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.evaluation.ablation import evaluate_ablations_from_config


def main() -> None:
    parser = argparse.ArgumentParser(description="Run IB world-model input ablations")
    parser.add_argument("--config", default="configs/world_model_ib_m1000.yaml")
    args = parser.parse_args()
    print(json.dumps(evaluate_ablations_from_config(args.config), indent=2))


if __name__ == "__main__":
    main()
