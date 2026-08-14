from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Dict, Iterable

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from src.data.ib_dataset import load_ib_npz, trajectory_spans, validate_ib_semantics
from src.utils.config import load_config, resolve_path
from src.world_model.interface import FrozenWorldModel


def _metric_dict(actual: np.ndarray, predicted: np.ndarray, target_std: np.ndarray) -> Dict[str, float]:
    error = predicted - actual
    return {
        "mse": float(np.mean(np.square(error))),
        "mae": float(np.mean(np.abs(error))),
        "rmse": float(np.sqrt(np.mean(np.square(error)))),
        "nrmse_std": float(np.sqrt(np.mean(np.square(error / target_std)))),
    }


def _write_json(path: Path, values: Dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(values, indent=2, ensure_ascii=False), encoding="utf-8")


def _write_rows(path: Path, fieldnames: Iterable[str], rows: Iterable[Dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def evaluate_one_step(
    world_model: FrozenWorldModel,
    val_data: Dict[str, np.ndarray],
    variable_names: list[str],
    batch_size: int,
    json_path: Path,
    csv_path: Path,
    figure_path: Path,
    plot_samples: int,
    selected_variables: list[str],
) -> Dict[str, object]:
    predictions = []
    for start in range(0, len(val_data["obs"]), batch_size):
        stop = min(start + batch_size, len(val_data["obs"]))
        predictions.append(
            world_model.predict_next_frame(val_data["obs"][start:stop], val_data["action"][start:stop])
        )
    predicted = np.concatenate(predictions, axis=0)
    actual = val_data["next_obs"][:, : world_model.frame_dim]
    result: Dict[str, object] = _metric_dict(actual, predicted, world_model.stats.target_std)
    per_variable = {}
    rows = []
    for i, name in enumerate(variable_names):
        metrics = _metric_dict(actual[:, i], predicted[:, i], world_model.stats.target_std[i])
        per_variable[name] = metrics
        rows.append({"variable": name, **metrics})
    result["per_variable"] = per_variable
    result["samples"] = int(len(actual))
    _write_json(json_path, result)
    _write_rows(csv_path, ("variable", "mse", "mae", "rmse", "nrmse_std"), rows)

    figure_path.parent.mkdir(parents=True, exist_ok=True)
    count = min(plot_samples, len(actual))
    fig, axes = plt.subplots(len(selected_variables), 1, figsize=(9, 3.2 * len(selected_variables)), sharex=True)
    axes = np.atleast_1d(axes)
    for axis, name in zip(axes, selected_variables):
        index = variable_names.index(name)
        axis.plot(actual[:count, index], label="actual", linewidth=1.5)
        axis.plot(predicted[:count, index], label="predicted", linewidth=1.2, alpha=0.85)
        axis.set_ylabel(name)
        axis.grid(alpha=0.25)
    axes[0].legend()
    axes[-1].set_xlabel("Validation transition")
    fig.suptitle("One-step prediction")
    fig.tight_layout()
    fig.savefig(figure_path, dpi=180)
    plt.close(fig)
    return result


def evaluate_multi_step(
    world_model: FrozenWorldModel,
    val_data: Dict[str, np.ndarray],
    variable_names: list[str],
    horizons: list[int],
    stride: int,
    batch_size: int,
    json_path: Path,
    csv_path: Path,
    horizon_figure_path: Path,
    rollout_figure_path: Path,
    selected_variables: list[str],
) -> Dict[str, object]:
    spans = trajectory_spans(val_data["index"], len(val_data["obs"]))
    max_horizon = max(horizons)
    starts = np.asarray(
        [
            start
            for span_start, span_end in spans
            for start in range(int(span_start), int(span_end - max_horizon + 1), stride)
        ],
        dtype=np.int64,
    )
    if starts.size == 0:
        raise ValueError("No valid multi-step rollout starts")

    squared = {h: [] for h in horizons}
    absolute = {h: [] for h in horizons}
    for batch_start in range(0, len(starts), batch_size):
        batch_indices = starts[batch_start : batch_start + batch_size]
        histories = val_data["obs"][batch_indices].copy()
        for step in range(max_horizon):
            actions = val_data["action"][batch_indices + step]
            predicted = world_model.predict_next_frame(histories, actions)
            actual = val_data["next_obs"][batch_indices + step, : world_model.frame_dim]
            horizon = step + 1
            if horizon in squared:
                error = predicted - actual
                squared[horizon].append(np.square(error))
                absolute[horizon].append(np.abs(error))
            histories = np.concatenate((predicted, histories[:, : -world_model.frame_dim]), axis=1)

    rows = []
    result: Dict[str, object] = {"rollout_starts": int(len(starts)), "stride": int(stride), "horizons": {}}
    for horizon in horizons:
        sq = np.concatenate(squared[horizon], axis=0)
        ab = np.concatenate(absolute[horizon], axis=0)
        row = {
            "horizon": horizon,
            "mse": float(sq.mean()),
            "mae": float(ab.mean()),
            "rmse": float(np.sqrt(sq.mean())),
            "nrmse_std": float(np.sqrt(np.mean(sq / np.square(world_model.stats.target_std)))),
        }
        rows.append(row)
        result["horizons"][str(horizon)] = row
    _write_json(json_path, result)
    _write_rows(csv_path, ("horizon", "mse", "mae", "rmse", "nrmse_std"), rows)

    horizon_figure_path.parent.mkdir(parents=True, exist_ok=True)
    plt.figure(figsize=(7, 4.5))
    plt.plot([row["horizon"] for row in rows], [row["nrmse_std"] for row in rows], marker="o")
    plt.xlabel("Autoregressive rollout horizon")
    plt.ylabel("NRMSE (training target std)")
    plt.title("Multi-step prediction error accumulation")
    plt.grid(alpha=0.25)
    plt.tight_layout()
    plt.savefig(horizon_figure_path, dpi=180)
    plt.close()

    selected_start = int(spans[0, 0])
    actions = val_data["action"][selected_start : selected_start + max_horizon]
    predicted_rollout = world_model.rollout(val_data["obs"][selected_start], actions)
    actual_rollout = val_data["next_obs"][selected_start : selected_start + max_horizon, : world_model.frame_dim]
    fig, axes = plt.subplots(len(selected_variables), 1, figsize=(9, 3.2 * len(selected_variables)), sharex=True)
    axes = np.atleast_1d(axes)
    for axis, name in zip(axes, selected_variables):
        index = variable_names.index(name)
        axis.plot(actual_rollout[:, index], label="actual", linewidth=1.5)
        axis.plot(predicted_rollout[:, index], label="autoregressive prediction", linewidth=1.2)
        axis.set_ylabel(name)
        axis.grid(alpha=0.25)
    axes[0].legend()
    axes[-1].set_xlabel("Rollout step")
    fig.suptitle(f"Selected validation rollout (horizon={max_horizon})")
    fig.tight_layout()
    fig.savefig(rollout_figure_path, dpi=180)
    plt.close(fig)
    return result


def evaluate_from_config(
    config_path: str | Path,
    checkpoint_override: str | Path | None = None,
    output_suffix: str = "",
) -> Dict[str, object]:
    config, project_root = load_config(config_path)
    data_cfg = config["data"]
    eval_cfg = config["evaluation"]
    output_cfg = config["outputs"]
    val_data = load_ib_npz(resolve_path(project_root, data_cfg["val_path"]))
    print("val_audit", validate_ib_semantics(val_data, data_cfg["history_len"], data_cfg["frame_dim"]))
    checkpoint_path = checkpoint_override or output_cfg["checkpoint"]
    world_model = FrozenWorldModel(
        resolve_path(project_root, checkpoint_path), config["training"]["device"]
    )

    def output_path(key: str) -> Path:
        path = resolve_path(project_root, output_cfg[key])
        if output_suffix:
            path = path.with_name(f"{path.stem}{output_suffix}{path.suffix}")
        return path

    one_step = evaluate_one_step(
        world_model,
        val_data,
        data_cfg["variable_names"],
        eval_cfg["batch_size"],
        output_path("one_step_json"),
        output_path("one_step_csv"),
        output_path("one_step_figure"),
        eval_cfg["one_step_plot_samples"],
        eval_cfg["selected_variables"],
    )
    multi_step = evaluate_multi_step(
        world_model,
        val_data,
        data_cfg["variable_names"],
        eval_cfg["horizons"],
        eval_cfg["rollout_stride"],
        eval_cfg["batch_size"],
        output_path("multi_step_json"),
        output_path("multi_step_csv"),
        output_path("horizon_figure"),
        output_path("rollout_figure"),
        eval_cfg["selected_variables"],
    )
    print("one_step", json.dumps(one_step, indent=2))
    print("multi_step", json.dumps(multi_step, indent=2))
    return {"one_step": one_step, "multi_step": multi_step}
