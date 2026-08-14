from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.evaluation.checkpoint_selection import evaluate_m1000_architecture_study


def main() -> None:
    parser = argparse.ArgumentParser(description="Validation-only M-1000 world-model selection")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--batch-size", type=int, default=2048)
    parser.add_argument("--rollout-stride", type=int, default=10)
    args = parser.parse_args()
    result = evaluate_m1000_architecture_study(
        ROOT,
        device=args.device,
        batch_size=args.batch_size,
        rollout_stride=args.rollout_stride,
    )
    print(json.dumps(result["global_selection"], indent=2))


if __name__ == "__main__":
    main()
