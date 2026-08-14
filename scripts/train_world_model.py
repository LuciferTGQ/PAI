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
    parser.add_argument("--resume", default=None)
    parser.add_argument("--epochs", type=int, default=None)
    args = parser.parse_args()
    train_from_config(args.config, resume_from=args.resume, target_epochs=args.epochs)


if __name__ == "__main__":
    main()
