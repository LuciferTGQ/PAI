from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.evaluation.simulator import evaluate_strategy_from_config


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate IB behavior, BC, and WM-CEM policies")
    parser.add_argument("--config", default="configs/strategy_ib_m100.yaml")
    parser.add_argument("--world-model-checkpoint", default=None)
    parser.add_argument("--output-suffix", default="")
    args = parser.parse_args()
    results = evaluate_strategy_from_config(
        args.config, args.world_model_checkpoint, args.output_suffix
    )
    for name in ("original_behavior_reference", "official_bc", "world_model_cem"):
        print(name, results[name]["summary"])


if __name__ == "__main__":
    main()
