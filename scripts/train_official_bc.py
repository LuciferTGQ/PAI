from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.strategy.official_bc import train_official_bc
from src.utils.config import load_config


def main() -> None:
    parser = argparse.ArgumentParser(description="Train the fixed-commit NeoRL OfflineRL BC")
    parser.add_argument("--config", default="configs/official_bc_ib_m100.yaml")
    args = parser.parse_args()
    config, _ = load_config(args.config)
    train_official_bc(config)


if __name__ == "__main__":
    main()
