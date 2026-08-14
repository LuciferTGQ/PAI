from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.evaluation.epoch_study import analyze_epoch_study


if __name__ == "__main__":
    print(json.dumps(analyze_epoch_study(ROOT), indent=2))
