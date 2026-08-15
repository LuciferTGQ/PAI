from __future__ import annotations

import argparse
import sys
from pathlib import Path

import torch
import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.utils.config import resolve_path
from src.world_model.trainer import train_from_config


DEFAULT_CONFIGS = [
    "configs/world_model_ib_m100_transformer2_fair_50e.yaml",
    "configs/world_model_ib_m1000_transformer2_fair_50e.yaml",
    "configs/world_model_ib_m10000_transformer2_fair_10e.yaml",
    "configs/world_model_ib_m100_mlp_50e.yaml",
    "configs/world_model_ib_m100_gru_50e.yaml",
    "configs/world_model_ib_m100_lstm_50e.yaml",
    "configs/world_model_ib_m100_transformer4_50e.yaml",
    "configs/world_model_ib_m10000_mlp_10e.yaml",
    "configs/world_model_ib_m10000_gru_10e.yaml",
    "configs/world_model_ib_m10000_lstm_10e.yaml",
    "configs/world_model_ib_m10000_transformer4_10e.yaml",
]


def _checkpoint_epoch(path: Path) -> int:
    return int(torch.load(path, map_location="cpu", weights_only=False)["epoch"])


def _latest_resume(config_path: Path) -> tuple[Path | None, int, int]:
    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    target = int(config["training"]["epochs"])
    checkpoint = resolve_path(ROOT, config["outputs"]["checkpoint"])
    last_value = config["outputs"].get("last_checkpoint")
    last = resolve_path(ROOT, last_value) if last_value else None
    candidates: list[Path] = []
    if last is not None and last.exists():
        candidates.append(last)
    stem = checkpoint.stem[:-5] if checkpoint.stem.endswith("_best") else checkpoint.stem
    candidates.extend(checkpoint.parent.glob(f"{stem}_epoch_*.pt"))
    if not candidates:
        return None, 0, target
    latest = max(candidates, key=_checkpoint_epoch)
    return latest, _checkpoint_epoch(latest), target


def main() -> None:
    parser = argparse.ArgumentParser(description="Resume-safe M-100/M-10000 World Model matrix")
    parser.add_argument("configs", nargs="*", default=DEFAULT_CONFIGS)
    args = parser.parse_args()
    for value in args.configs:
        config_path = resolve_path(ROOT, value)
        resume, epoch, target = _latest_resume(config_path)
        if epoch >= target:
            print(f"reuse completed config={config_path.name} epoch={epoch}")
            continue
        print(
            f"train config={config_path.name} resume={resume if resume else 'none'} "
            f"current_epoch={epoch} target={target}"
        )
        train_from_config(config_path, resume_from=resume, target_epochs=target)


if __name__ == "__main__":
    main()
