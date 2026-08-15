from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.evaluation.data_scaling import ARCHITECTURES_5X3, evaluate_data_scaling


def main() -> None:
    parser = argparse.ArgumentParser(description="Validation-only fair 5x3 World Model matrix")
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()
    result = evaluate_data_scaling(
        ROOT,
        scales=(100, 1000, 10000),
        device=args.device,
        architectures=ARCHITECTURES_5X3,
        output_prefix="world_model_5x3_common_validation",
        reuse_m1000_legacy=False,
        candidate_validation_path="data/raw/ib-medium-1000-val.npz",
        candidate_fixed_std=True,
    )
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
