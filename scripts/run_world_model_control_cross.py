from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.evaluation.world_model_control_cross import run_world_model_control_cross


def main() -> None:
    parser = argparse.ArgumentParser(description="Run M-1000 World Model to control cross matrix")
    parser.add_argument("--config", default="configs/world_model_control_cross_m1000.yaml")
    args = parser.parse_args()
    run_world_model_control_cross(args.config)


if __name__ == "__main__":
    main()
