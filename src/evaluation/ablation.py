from __future__ import annotations

import json
from pathlib import Path
from typing import Dict

import numpy as np

from src.data.ib_dataset import load_ib_npz
from src.utils.config import load_config, resolve_path
from src.world_model.interface import FrozenWorldModel


def _metrics(prediction: np.ndarray, target: np.ndarray) -> Dict[str, float]:
    error = prediction - target
    return {
        "mse": float(np.mean(np.square(error))),
        "mae": float(np.mean(np.abs(error))),
    }


def _predict_batched(
    model: FrozenWorldModel,
    observations: np.ndarray,
    actions: np.ndarray,
    batch_size: int,
) -> np.ndarray:
    return np.concatenate(
        [
            model.predict_next_frame(
                observations[start : start + batch_size],
                actions[start : start + batch_size],
            )
            for start in range(0, len(observations), batch_size)
        ],
        axis=0,
    )


def evaluate_ablations_from_config(config_path: str | Path) -> Dict[str, object]:
    config, root = load_config(config_path)
    data_config = config["data"]
    eval_config = config["evaluation"]
    output_config = config["outputs"]
    data = load_ib_npz(resolve_path(root, data_config["val_path"]))
    model = FrozenWorldModel(
        resolve_path(root, output_config["checkpoint"]), config["training"]["device"]
    )
    observations = data["obs"].astype(np.float32, copy=False)
    actions = data["action"].astype(np.float32, copy=False)
    targets = data["next_obs"][:, : model.frame_dim].astype(np.float32, copy=False)
    batch_size = int(eval_config["batch_size"])
    rng = np.random.default_rng(int(config["seed"]))

    predictions = _predict_batched(model, observations, actions, batch_size)
    shuffled_predictions = _predict_batched(
        model, observations, actions[rng.permutation(len(actions))], batch_size
    )
    zero_action_predictions = _predict_batched(
        model, observations, np.zeros_like(actions), batch_size
    )
    latest_frame = observations[:, : model.frame_dim]
    latest_only_history = np.repeat(
        latest_frame[:, None, :], model.history_len, axis=1
    ).reshape(len(observations), model.obs_dim)
    latest_only_predictions = _predict_batched(
        model, latest_only_history, actions, batch_size
    )

    results: Dict[str, object] = {
        "samples": int(len(targets)),
        "model": _metrics(predictions, targets),
        "persistence": _metrics(latest_frame, targets),
        "shuffled_action": _metrics(shuffled_predictions, targets),
        "zero_action": _metrics(zero_action_predictions, targets),
        "latest_frame_repeated": _metrics(latest_only_predictions, targets),
    }
    model_mse = results["model"]["mse"]
    persistence_mse = results["persistence"]["mse"]
    results["mse_reduction_vs_persistence"] = float(
        (persistence_mse - model_mse) / persistence_mse
    )
    output_path = resolve_path(root, output_config["ablation_json"])
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as handle:
        json.dump(results, handle, indent=2)
    return results
