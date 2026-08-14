from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.evaluation.prediction import evaluate_from_config


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate one-step and multi-step IB prediction")
    parser.add_argument("--config", default="configs/world_model_ib_m100.yaml")
    parser.add_argument("--checkpoint", default=None)
    parser.add_argument("--output-suffix", default="")
    args = parser.parse_args()
    evaluate_from_config(args.config, args.checkpoint, args.output_suffix)


if __name__ == "__main__":
    main()
