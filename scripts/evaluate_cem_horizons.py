from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.evaluation.cem_horizon import evaluate_cem_horizons


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Select CEM horizon with frozen M-1000 dynamics"
    )
    parser.add_argument("--config", default="configs/cem_horizon_ib_m1000_gru.yaml")
    args = parser.parse_args()
    evaluate_cem_horizons(args.config)


if __name__ == "__main__":
    main()
