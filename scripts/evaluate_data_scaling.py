from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.evaluation.data_scaling import evaluate_data_scaling


def main() -> None:
    parser = argparse.ArgumentParser(description="Validation-only 4x3 World Model scaling table")
    parser.add_argument("--scales", nargs="+", type=int, default=[100, 1000, 10000])
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()
    result = evaluate_data_scaling(ROOT, scales=args.scales, device=args.device)
    for scale, selections in result["selections"].items():
        print(f"M-{scale}")
        for architecture, selection in selections.items():
            row = selection["selected"]
            print(
                architecture,
                f"epoch={row['epoch']}",
                f"one={row['one_step_NRMSE']:.6f}",
                f"rollout={row['mean_NRMSE_H5_H10_H20']:.6f}",
            )


if __name__ == "__main__":
    main()
