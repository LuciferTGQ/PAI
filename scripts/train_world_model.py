from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.world_model.trainer import train_from_config


def main() -> None:
    parser = argparse.ArgumentParser(description="Train the IB Temporal Transformer world model")
    parser.add_argument("--config", default="configs/world_model_ib_m100.yaml")
    args = parser.parse_args()
    train_from_config(args.config)


if __name__ == "__main__":
    main()

