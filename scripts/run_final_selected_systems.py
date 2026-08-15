from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.evaluation.final_selected_systems import run_final_selected_systems


def main() -> None:
    parser = argparse.ArgumentParser(description="Run final fresh-seed selected systems")
    parser.add_argument("--config", default="configs/final_selected_systems.yaml")
    args = parser.parse_args()
    run_final_selected_systems(args.config)


if __name__ == "__main__":
    main()
