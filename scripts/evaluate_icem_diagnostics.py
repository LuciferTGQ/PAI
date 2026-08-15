from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.evaluation.icem_diagnostics import evaluate_icem_diagnostics


def main() -> None:
    parser = argparse.ArgumentParser(description="Reference-validated iCEM diagnostics")
    parser.add_argument("--config", default="configs/icem_reference_diagnostics.yaml")
    args = parser.parse_args()
    result = evaluate_icem_diagnostics(args.config)
    for dataset, summary in result["summaries"].items():
        print(dataset, summary)


if __name__ == "__main__":
    main()
