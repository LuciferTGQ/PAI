from __future__ import annotations

import csv
import json
import time
from pathlib import Path
from typing import Any, Dict, Iterable

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

from src.data.ib_dataset import load_ib_npz, trajectory_spans, validate_ib_semantics
from src.world_model.interface import FrozenWorldModel


ROLLOUT_HORIZONS = (5, 10, 20, 50)
SELECTION_HORIZONS = (5, 10, 20)
ONE_STEP_TOLERANCE = 1.10


def _relative_path(path: Path, root: Path) -> str:
    try:
        return path.resolve().relative_to(root.resolve()).as_posix()
    except ValueError:
        return str(path.resolve())


def _checkpoint_epoch(path: Path) -> int:
    checkpoint = torch.load(path, map_location="cpu", weights_only=False)
    if "epoch" not in checkpoint:
        raise ValueError(f"Checkpoint has no epoch metadata: {path}")
    return int(checkpoint["epoch"])


def _deduplicate_by_epoch(paths: Iterable[Path]) -> list[Path]:
    by_epoch: Dict[int, Path] = {}
    for path in paths:
        if path.exists():
            by_epoch[_checkpoint_epoch(path)] = path
    return [by_epoch[epoch] for epoch in sorted(by_epoch)]


def discover_m1000_candidates(root: Path) -> Dict[str, list[Path]]:
    checkpoint_dir = root / "outputs" / "checkpoints"
    legacy_transformer = [
        checkpoint_dir / "world_model_ib_m1000.pt",
        checkpoint_dir / "world_model_ib_m1000_100e_best.pt",
        checkpoint_dir / "world_model_ib_m1000_100e_last.pt",
    ]
    candidates: Dict[str, list[Path]] = {
        "Transformer-2L": _deduplicate_by_epoch(legacy_transformer)
    }
    for architecture, prefix in (
        ("MLP", "world_model_ib_m1000_mlp"),
        ("GRU", "world_model_ib_m1000_gru"),
        ("LSTM", "world_model_ib_m1000_lstm"),
        ("Transformer-4L", "world_model_ib_m1000_transformer4"),
    ):
        paths = [checkpoint_dir / f"{prefix}_best.pt", checkpoint_dir / f"{prefix}_last.pt"]
        paths.extend(sorted(checkpoint_dir.glob(f"{prefix}_epoch_*.pt")))
        candidates[architecture] = _deduplicate_by_epoch(paths)
    missing = [name for name, paths in candidates.items() if not paths]
    if missing:
        raise FileNotFoundError(f"No checkpoints found for: {', '.join(missing)}")
    return candidates


def _one_step_nrmse(
    world_model: FrozenWorldModel,
    val_data: Dict[str, np.ndarray],
    batch_size: int,
    nrmse_std: np.ndarray | None = None,
) -> float:
    denominator = world_model.stats.target_std if nrmse_std is None else nrmse_std
    squared_normalized_errors = []
    for start in range(0, len(val_data["obs"]), batch_size):
        stop = min(start + batch_size, len(val_data["obs"]))
        predicted = world_model.predict_next_frame(
            val_data["obs"][start:stop], val_data["action"][start:stop]
        )
        actual = val_data["next_obs"][start:stop, : world_model.frame_dim]
        squared_normalized_errors.append(
            np.square((predicted - actual) / denominator)
        )
    return float(np.sqrt(np.concatenate(squared_normalized_errors).mean()))


def _multi_step_nrmse(
    world_model: FrozenWorldModel,
    val_data: Dict[str, np.ndarray],
    horizons: Iterable[int],
    stride: int,
    batch_size: int,
    max_rollout_starts: int | None = None,
    nrmse_std: np.ndarray | None = None,
) -> tuple[Dict[int, float], int]:
    horizons = tuple(sorted(int(value) for value in horizons))
    max_horizon = max(horizons)
    spans = trajectory_spans(val_data["index"], len(val_data["obs"]))
    starts = np.asarray(
        [
            start
            for span_start, span_end in spans
            for start in range(int(span_start), int(span_end - max_horizon + 1), stride)
        ],
        dtype=np.int64,
    )
    if starts.size == 0:
        raise ValueError("No valid multi-step validation starts")
    if max_rollout_starts is not None and len(starts) > max_rollout_starts:
        positions = np.linspace(0, len(starts) - 1, max_rollout_starts, dtype=np.int64)
        starts = starts[positions]
    squared_errors: Dict[int, list[np.ndarray]] = {horizon: [] for horizon in horizons}
    denominator = world_model.stats.target_std if nrmse_std is None else nrmse_std
    for batch_start in range(0, len(starts), batch_size):
        indices = starts[batch_start : batch_start + batch_size]
        histories = val_data["obs"][indices].copy()
        for step in range(max_horizon):
            predicted = world_model.predict_next_frame(
                histories, val_data["action"][indices + step]
            )
            horizon = step + 1
            if horizon in squared_errors:
                actual = val_data["next_obs"][indices + step, : world_model.frame_dim]
                squared_errors[horizon].append(
                    np.square((predicted - actual) / denominator)
                )
            histories = np.concatenate(
                (predicted, histories[:, : -world_model.frame_dim]), axis=1
            )
    metrics = {
        horizon: float(np.sqrt(np.concatenate(values).mean()))
        for horizon, values in squared_errors.items()
    }
    return metrics, int(len(starts))


def evaluate_checkpoint(
    architecture: str,
    checkpoint_path: Path,
    val_data: Dict[str, np.ndarray],
    root: Path,
    device: str,
    batch_size: int,
    rollout_stride: int,
    max_rollout_starts: int | None = None,
    nrmse_std: np.ndarray | None = None,
) -> Dict[str, Any]:
    started = time.perf_counter()
    metadata = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    world_model = FrozenWorldModel(checkpoint_path, device=device)
    one_step = _one_step_nrmse(world_model, val_data, batch_size, nrmse_std)
    multi_step, rollout_starts = _multi_step_nrmse(
        world_model,
        val_data,
        ROLLOUT_HORIZONS,
        rollout_stride,
        batch_size,
        max_rollout_starts,
        nrmse_std,
    )
    parameter_count = int(
        metadata.get(
            "parameter_count",
            sum(parameter.numel() for parameter in world_model.model.parameters()),
        )
    )
    result: Dict[str, Any] = {
        "architecture": architecture,
        "epoch": int(metadata["epoch"]),
        "checkpoint": _relative_path(checkpoint_path, root),
        "parameter_count": parameter_count,
        "one_step_NRMSE": one_step,
        **{f"NRMSE_H{horizon}": value for horizon, value in multi_step.items()},
        "mean_NRMSE_H5_H10_H20": float(
            np.mean([multi_step[horizon] for horizon in SELECTION_HORIZONS])
        ),
        "rollout_starts": rollout_starts,
        "evaluation_seconds": time.perf_counter() - started,
    }
    del world_model
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return result


def select_by_validation_protocol(
    records: list[Dict[str, Any]], tolerance: float = ONE_STEP_TOLERANCE
) -> Dict[str, Any]:
    if not records:
        raise ValueError("Cannot select from an empty checkpoint list")
    best_one_step = min(float(record["one_step_NRMSE"]) for record in records)
    threshold = tolerance * best_one_step
    eligible = [
        record for record in records if float(record["one_step_NRMSE"]) <= threshold
    ]
    selected = min(
        eligible,
        key=lambda record: (
            float(record["mean_NRMSE_H5_H10_H20"]),
            float(record["one_step_NRMSE"]),
            int(record["epoch"]),
        ),
    )
    return {
        "best_one_step_NRMSE": best_one_step,
        "one_step_threshold": threshold,
        "eligible_epochs": [int(record["epoch"]) for record in eligible],
        "selected": selected,
    }


def _extension_recommendation(records: list[Dict[str, Any]], selected_epoch: int) -> bool:
    periodic = sorted(
        [record for record in records if int(record["epoch"]) % 5 == 0],
        key=lambda record: int(record["epoch"]),
    )
    if selected_epoch < 45 or len(periodic) < 3:
        return False
    latest = periodic[-3:]
    epoch_values = np.asarray([record["epoch"] for record in latest], dtype=np.float64)
    rollout_values = np.asarray(
        [record["mean_NRMSE_H5_H10_H20"] for record in latest], dtype=np.float64
    )
    slope = float(np.polyfit(epoch_values, rollout_values, 1)[0])
    relative_improvement = float((rollout_values[-2] - rollout_values[-1]) / rollout_values[-2])
    return slope < 0.0 and relative_improvement >= 0.01


def _write_csv(path: Path, records: list[Dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = list(records[0])
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(records)


def _write_figure(
    output_base: Path,
    records: list[Dict[str, Any]],
    selections: Dict[str, Any],
    global_selection: Dict[str, Any],
) -> None:
    """Render the validation-selection argument as an editable publication figure."""
    output_base.parent.mkdir(parents=True, exist_ok=True)
    plt.rcParams.update(
        {
            "font.family": "sans-serif",
            "font.sans-serif": ["Arial", "DejaVu Sans", "Liberation Sans"],
            "svg.fonttype": "none",
            "pdf.fonttype": 42,
            "font.size": 7,
            "axes.spines.right": False,
            "axes.spines.top": False,
            "axes.linewidth": 0.8,
            "legend.frameon": False,
        }
    )
    colors = {
        "MLP": "#7884B4",
        "GRU": "#B64342",
        "LSTM": "#9A4D8E",
        "Transformer-2L": "#42949E",
        "Transformer-4L": "#0F4D92",
    }
    figure = plt.figure(figsize=(7.2, 6.0))
    grid = figure.add_gridspec(2, 2, width_ratios=(1.12, 1.0), hspace=0.38, wspace=0.38)
    axes = [figure.add_subplot(grid[:, 0]), figure.add_subplot(grid[0, 1]), figure.add_subplot(grid[1, 1])]

    selected_rows = {
        architecture: values["selected"] for architecture, values in selections.items()
    }
    threshold = float(global_selection["one_step_threshold"])
    chosen = global_selection["selected"]
    axes[0].axvspan(threshold, max(threshold * 1.035, 0.205), color="#F6CFCB", alpha=0.35, lw=0)
    axes[0].axvline(threshold, color="#B64342", linestyle="--", linewidth=1.0)
    for architecture, row in selected_rows.items():
        is_chosen = architecture == chosen["architecture"]
        axes[0].scatter(
            row["one_step_NRMSE"],
            row["mean_NRMSE_H5_H10_H20"],
            s=95 if is_chosen else 48,
            marker="*" if is_chosen else "o",
            color=colors[architecture],
            edgecolor="white",
            linewidth=0.7,
            zorder=4,
        )
        axes[0].annotate(
            f"{architecture}\n(e{row['epoch']})",
            (row["one_step_NRMSE"], row["mean_NRMSE_H5_H10_H20"]),
            xytext=(5, 5 if architecture != "Transformer-4L" else -17),
            textcoords="offset points",
            color=colors[architecture],
            fontsize=6.5,
        )
    axes[0].annotate(
        "1.10× best one-step threshold",
        xy=(threshold, axes[0].get_ylim()[1]),
        xytext=(-3, -4),
        textcoords="offset points",
        ha="right",
        va="top",
        fontsize=6.2,
        color="#B64342",
    )
    axes[0].set(
        title="Validation-only deployment selection",
        xlabel="One-step NRMSE",
        ylabel="Mean rollout NRMSE (H5/H10/H20)",
    )

    for architecture in sorted({record["architecture"] for record in records}):
        values = sorted(
            [record for record in records if record["architecture"] == architecture],
            key=lambda record: record["epoch"],
        )
        epochs = [record["epoch"] for record in values]
        axes[1].plot(
            epochs,
            [record["mean_NRMSE_H5_H10_H20"] for record in values],
            marker="o", markersize=2.8, linewidth=1.1,
            color=colors[architecture],
            label=architecture,
        )
        selected = selections[architecture]["selected"]
        axes[1].scatter(
            selected["epoch"], selected["mean_NRMSE_H5_H10_H20"],
            s=55, marker="*", color=colors[architecture], edgecolor="white", linewidth=0.5, zorder=5
        )
    axes[1].set(
        title="Checkpoint rollout trend",
        xlabel="Epoch",
        ylabel="mean NRMSE (H5/H10/H20)",
    )

    horizon_values = np.asarray(ROLLOUT_HORIZONS)
    axes[2].axvspan(20, 50, color="#D8D8D8", alpha=0.25, lw=0)
    for architecture, row in selected_rows.items():
        axes[2].plot(
            horizon_values,
            [row[f"NRMSE_H{horizon}"] for horizon in horizon_values],
            marker="o", markersize=3.0, linewidth=1.2,
            color=colors[architecture], label=architecture,
        )
    axes[2].axvline(20, color="#767676", linestyle=":", linewidth=0.8)
    axes[2].text(34.5, axes[2].get_ylim()[0], "H50: diagnostic only", ha="center", va="bottom", fontsize=6.2, color="#606060")
    axes[2].set(
        title="Selected-checkpoint stability",
        xlabel="Autoregressive horizon",
        ylabel="NRMSE",
        xticks=list(ROLLOUT_HORIZONS),
    )

    for label, axis in zip(("a", "b", "c"), axes):
        axis.grid(alpha=0.25)
        axis.text(-0.12, 1.04, label, transform=axis.transAxes, fontsize=8, fontweight="bold")
    axes[1].legend(fontsize=5.8, ncol=2, loc="upper right")
    axes[2].legend(fontsize=5.8, ncol=2, loc="upper left")
    figure.subplots_adjust(left=0.09, right=0.99, bottom=0.09, top=0.95)
    figure.savefig(output_base.with_suffix(".svg"), bbox_inches="tight")
    figure.savefig(output_base.with_suffix(".pdf"), bbox_inches="tight")
    figure.savefig(output_base.with_suffix(".png"), dpi=300, bbox_inches="tight")
    plt.close(figure)


def evaluate_m1000_architecture_study(
    root: str | Path,
    device: str = "cuda",
    batch_size: int = 2048,
    rollout_stride: int = 10,
) -> Dict[str, Any]:
    root = Path(root)
    val_data = load_ib_npz(root / "data" / "raw" / "ib-medium-100-val.npz")
    audit = validate_ib_semantics(val_data, history_len=30, frame_dim=6)
    candidates = discover_m1000_candidates(root)
    records: list[Dict[str, Any]] = []
    for architecture, checkpoint_paths in candidates.items():
        for checkpoint_path in checkpoint_paths:
            print(f"evaluating architecture={architecture} checkpoint={checkpoint_path.name}")
            record = evaluate_checkpoint(
                architecture,
                checkpoint_path,
                val_data,
                root,
                device,
                batch_size,
                rollout_stride,
            )
            records.append(record)
            print(json.dumps(record, indent=2))

    architecture_selections = {
        architecture: select_by_validation_protocol(
            [record for record in records if record["architecture"] == architecture]
        )
        for architecture in candidates
    }
    model_rows = [values["selected"] for values in architecture_selections.values()]
    global_selection = select_by_validation_protocol(model_rows)
    extension = {
        architecture: _extension_recommendation(
            [record for record in records if record["architecture"] == architecture],
            int(selection["selected"]["epoch"]),
        )
        for architecture, selection in architecture_selections.items()
        if architecture != "Transformer-2L"
    }
    result = {
        "protocol": {
            "selection_data": "ib-medium-100-val only",
            "one_step_tolerance": ONE_STEP_TOLERANCE,
            "selection_horizons": list(SELECTION_HORIZONS),
            "diagnostic_only_horizon": 50,
            "strategy_or_simulator_metrics_used": False,
            "checkpoint_selection": "within architecture, then across architectures",
        },
        "validation_audit": audit,
        "architecture_selections": architecture_selections,
        "global_selection": global_selection,
        "resume_to_75_recommended": extension,
        "all_checkpoint_metrics": records,
    }
    metrics_dir = root / "outputs" / "metrics"
    metrics_dir.mkdir(parents=True, exist_ok=True)
    json_path = metrics_dir / "world_model_ib_m1000_architecture_selection.json"
    json_path.write_text(json.dumps(result, indent=2, ensure_ascii=False), encoding="utf-8")
    _write_csv(metrics_dir / "world_model_ib_m1000_architecture_checkpoints.csv", records)
    _write_csv(metrics_dir / "world_model_ib_m1000_architecture_models.csv", model_rows)
    _write_figure(
        root / "outputs" / "figures" / "world_model_ib_m1000_architecture_selection",
        records,
        architecture_selections,
        global_selection,
    )
    print(f"selected_deployment={global_selection['selected']['checkpoint']}")
    return result
