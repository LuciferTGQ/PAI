from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.evaluation.icem_parameter_check import evaluate_icem_parameter_check


def main() -> None:
    parser = argparse.ArgumentParser(description="Minimal reference iCEM parameter check")
    parser.add_argument("--config", default="configs/icem_reference_parameter_check.yaml")
    args = parser.parse_args()
    result = evaluate_icem_parameter_check(args.config)
    for row in result["episodes"]:
        print(row["dataset_scale"], row["config_variant"], row["episode_return"])


if __name__ == "__main__":
    main()
