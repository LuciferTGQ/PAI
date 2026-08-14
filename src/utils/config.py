from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, Tuple

import yaml


def project_root() -> Path:
    return Path(__file__).resolve().parents[2]


def resolve_path(root: Path, value: str | Path) -> Path:
    path = Path(value)
    return path if path.is_absolute() else root / path


def load_config(path: str | Path) -> Tuple[Dict[str, Any], Path]:
    root = project_root()
    config_path = resolve_path(root, path)
    with config_path.open("r", encoding="utf-8") as handle:
        config = yaml.safe_load(handle)
    return config, root

