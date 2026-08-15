from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.evaluation.end_to_end_scaling import evaluate_end_to_end_scaling


def main() -> None:
    parser = argparse.ArgumentParser(description="Fixed GRU + CEM end-to-end data scaling")
    parser.add_argument("--config", default="configs/end_to_end_data_scaling.yaml")
    parser.add_argument("--scales", nargs="+", type=int, default=[100, 1000, 10000])
    args = parser.parse_args()
    results = evaluate_end_to_end_scaling(args.config, scales=args.scales)
    for scale, section in results["data_scales"].items():
        print(f"M-{scale}", section.get("summary"))


if __name__ == "__main__":
    main()
