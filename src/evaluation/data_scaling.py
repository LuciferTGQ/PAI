from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any, Dict, Iterable

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np

from src.data.ib_dataset import load_ib_npz, validate_ib_semantics
from src.evaluation.checkpoint_selection import (
    evaluate_checkpoint,
    select_by_validation_protocol,
)


ARCHITECTURES = {
    "MLP": "mlp",
    "GRU": "gru",
    "LSTM": "lstm",
    "Transformer-4L": "transformer4",
}


def _epoch(path: Path) -> int:
    import torch

    return int(torch.load(path, map_location="cpu", weights_only=False)["epoch"])


def _candidates(root: Path, scale: int, suffix: str) -> list[Path]:
    directory = root / "outputs" / "checkpoints"
    prefix = f"world_model_ib_m{scale}_{suffix}"
    paths = [directory / f"{prefix}_best.pt", directory / f"{prefix}_last.pt"]
    paths.extend(sorted(directory.glob(f"{prefix}_epoch_*.pt")))
    by_epoch = {_epoch(path): path for path in paths if path.exists()}
    return [by_epoch[epoch] for epoch in sorted(by_epoch)]


def _write_json(path: Path, payload: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(f"{path.suffix}.tmp")
    temporary.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    temporary.replace(path)


def _write_csv(path: Path, rows: list[Dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def _reuse_m1000(root: Path) -> tuple[list[Dict[str, Any]], Dict[str, Any]]:
    source_path = root / "outputs" / "metrics" / "world_model_ib_m1000_architecture_selection.json"
    source = json.loads(source_path.read_text(encoding="utf-8"))
    records = [
        {"data_scale": 1000, **record}
        for record in source["all_checkpoint_metrics"]
        if record["architecture"] in ARCHITECTURES
    ]
    selections = {
        architecture: source["architecture_selections"][architecture]
        for architecture in ARCHITECTURES
    }
    return records, selections


def _extension_needed(scale: int, records: list[Dict[str, Any]], selected_epoch: int) -> bool:
    if scale == 100:
        boundary = 45
    elif scale == 10000:
        boundary = 9
    else:
        return False
    if selected_epoch < boundary:
        return False
    latest = sorted(records, key=lambda row: int(row["epoch"]))[-3:]
    if len(latest) < 3:
        return False
    values = np.asarray([row["mean_NRMSE_H5_H10_H20"] for row in latest])
    relative = (values[-2] - values[-1]) / max(values[-2], 1e-12)
    return bool(np.all(np.diff(values) < 0) and relative >= 0.01)


def _plot(path: Path, selected_rows: list[Dict[str, Any]]) -> None:
    mpl.rcParams.update(
        {
            "font.family": "sans-serif",
            "font.sans-serif": ["Arial", "DejaVu Sans"],
            "font.size": 8,
            "svg.fonttype": "none",
            "pdf.fonttype": 42,
            "axes.spines.top": False,
            "axes.spines.right": False,
        }
    )
    colors = {
        "MLP": "#7884B4",
        "GRU": "#B64342",
        "LSTM": "#9A4D8E",
        "Transformer-4L": "#0F4D92",
    }
    figure, axes = plt.subplots(1, 2, figsize=(8.3, 3.6))
    figure.suptitle(
        "Dynamics accuracy across offline training-data scales",
        fontsize=11,
        fontweight="bold",
        y=1.02,
    )
    for architecture in ARCHITECTURES:
        rows = sorted(
            [row for row in selected_rows if row["architecture"] == architecture],
            key=lambda row: int(row["data_scale"]),
        )
        if len(rows) != 3:
            continue
        scales = [row["data_scale"] for row in rows]
        axes[0].plot(scales, [row["one_step_NRMSE"] for row in rows], marker="o",
                     color=colors[architecture], label=architecture)
        axes[1].plot(scales, [row["mean_NRMSE_H5_H10_H20"] for row in rows], marker="o",
                     color=colors[architecture], label=architecture)
    for axis, title, ylabel in (
        (axes[0], "a  One-step accuracy", "One-step NRMSE"),
        (axes[1], "b  Autoregressive accuracy", "Mean NRMSE (H5/H10/H20)"),
    ):
        axis.set_xscale("log")
        axis.set_xticks([100, 1000, 10000], ["100", "1,000", "10,000"])
        axis.set_xlabel("Training trajectories")
        axis.set_ylabel(ylabel)
        axis.set_title(title, loc="left", fontweight="bold")
        axis.grid(alpha=0.22)
    axes[0].legend(frameon=False, fontsize=7.2)
    figure.tight_layout()
    base = path.with_suffix("")
    base.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(base.with_suffix(".png"), dpi=300, bbox_inches="tight")
    figure.savefig(base.with_suffix(".pdf"), bbox_inches="tight")
    figure.savefig(base.with_suffix(".svg"), bbox_inches="tight")
    plt.close(figure)


def evaluate_data_scaling(
    root: str | Path,
    scales: Iterable[int] = (100, 1000, 10000),
    device: str = "cuda",
    batch_size: int = 2048,
    rollout_stride: int = 10,
    max_rollout_starts: int = 10_000,
) -> Dict[str, Any]:
    root = Path(root)
    scales = [int(scale) for scale in scales]
    metrics_dir = root / "outputs" / "metrics"
    progress_path = metrics_dir / "world_model_data_scaling.json"
    progress: Dict[str, Any] = {
        "protocol": {
            "one_step_tolerance": 1.10,
            "selection_horizons": [5, 10, 20],
            "diagnostic_only_horizon": 50,
            "rollout_stride": rollout_stride,
            "max_rollout_starts": max_rollout_starts,
            "strategy_or_simulator_metrics_used": False,
            "architectures": list(ARCHITECTURES),
        },
        "validation_audits": {},
        "all_checkpoint_metrics": [],
        "selections": {},
        "extension_recommended": {},
        "common_validation_metrics": [],
        "fixed_common_validation_metrics": [],
    }
    if progress_path.exists():
        existing = json.loads(progress_path.read_text(encoding="utf-8"))
        if existing.get("protocol") == progress["protocol"]:
            progress = existing

    records = progress["all_checkpoint_metrics"]
    keys = {
        (int(row["data_scale"]), row["architecture"], int(row["epoch"]))
        for row in records
    }
    if 1000 in scales and not any(key[0] == 1000 for key in keys):
        reused_records, reused_selections = _reuse_m1000(root)
        records.extend(reused_records)
        progress["selections"]["1000"] = reused_selections
        keys.update(
            (1000, row["architecture"], int(row["epoch"])) for row in reused_records
        )
        _write_json(progress_path, progress)

    validation_paths = {
        100: root / "data" / "raw" / "ib-medium-10-val.npz",
        10000: root / "data" / "raw" / "ib-medium-1000-val.npz",
    }
    for scale in scales:
        if scale == 1000:
            continue
        val_data = load_ib_npz(validation_paths[scale])
        progress["validation_audits"][str(scale)] = validate_ib_semantics(val_data)
        for architecture, suffix in ARCHITECTURES.items():
            paths = _candidates(root, scale, suffix)
            if not paths:
                raise FileNotFoundError(f"No M-{scale} {architecture} checkpoints")
            for checkpoint in paths:
                epoch = _epoch(checkpoint)
                key = (scale, architecture, epoch)
                if key in keys:
                    print(f"reuse scale={scale} architecture={architecture} epoch={epoch}")
                    continue
                print(f"evaluate scale={scale} architecture={architecture} epoch={epoch}")
                row = evaluate_checkpoint(
                    architecture,
                    checkpoint,
                    val_data,
                    root,
                    device,
                    batch_size,
                    rollout_stride,
                    max_rollout_starts,
                )
                row = {"data_scale": scale, **row}
                records.append(row)
                keys.add(key)
                _write_json(progress_path, progress)
        del val_data

    for scale in scales:
        scale_records = [row for row in records if int(row["data_scale"]) == scale]
        if not scale_records:
            continue
        selections = {
            architecture: select_by_validation_protocol(
                [row for row in scale_records if row["architecture"] == architecture]
            )
            for architecture in ARCHITECTURES
        }
        progress["selections"][str(scale)] = selections
        progress["extension_recommended"][str(scale)] = {
            architecture: _extension_needed(
                scale,
                [row for row in scale_records if row["architecture"] == architecture],
                int(selection["selected"]["epoch"]),
            )
            for architecture, selection in selections.items()
        }
    _write_json(progress_path, progress)
    if records:
        _write_csv(metrics_dir / "world_model_data_scaling_checkpoints.csv", records)
    selected_rows = [
        {"data_scale": int(scale), **selection["selected"]}
        for scale, selections in progress["selections"].items()
        for selection in selections.values()
    ]
    selected_rows = sorted(
        selected_rows,
        key=lambda row: (int(row["data_scale"]), row["architecture"]),
    )
    if selected_rows:
        _write_csv(metrics_dir / "world_model_data_scaling_models.csv", selected_rows)
    if {int(scale) for scale in progress["selections"]} == {100, 1000, 10000}:
        common_path = root / "data" / "raw" / "ib-medium-1000-val.npz"
        common_data = load_ib_npz(common_path)
        common_rows = progress.setdefault("common_validation_metrics", [])
        common_keys = {
            (int(row["data_scale"]), row["architecture"], int(row["epoch"]))
            for row in common_rows
        }
        for selected in selected_rows:
            key = (
                int(selected["data_scale"]),
                selected["architecture"],
                int(selected["epoch"]),
            )
            if key in common_keys:
                continue
            checkpoint = root / selected["checkpoint"]
            print(
                f"common validation scale={key[0]} architecture={key[1]} epoch={key[2]}"
            )
            row = evaluate_checkpoint(
                selected["architecture"],
                checkpoint,
                common_data,
                root,
                device,
                batch_size,
                rollout_stride,
                max_rollout_starts,
            )
            common_rows.append({"data_scale": key[0], **row})
            common_keys.add(key)
            _write_json(progress_path, progress)
        common_rows = sorted(
            common_rows,
            key=lambda row: (int(row["data_scale"]), row["architecture"]),
        )
        _write_csv(
            metrics_dir / "world_model_data_scaling_common_validation_model_scale_std.csv",
            common_rows,
        )
        common_targets = common_data["next_obs"][:, :6].astype(np.float64)
        fixed_std = common_targets.std(axis=0)
        fixed_std[fixed_std < 1e-6] = 1.0
        fixed_std = fixed_std.astype(np.float32)
        del common_targets
        progress["common_validation_protocol"] = {
            "data": "data/raw/ib-medium-1000-val.npz",
            "target_std": fixed_std.tolist(),
            "fixed_across_all_data_scales_and_architectures": True,
            "checkpoint_selection_uses_this_metric": False,
        }
        fixed_rows = progress.setdefault("fixed_common_validation_metrics", [])
        fixed_keys = {
            (int(row["data_scale"]), row["architecture"], int(row["epoch"]))
            for row in fixed_rows
        }
        for selected in selected_rows:
            key = (
                int(selected["data_scale"]),
                selected["architecture"],
                int(selected["epoch"]),
            )
            if key in fixed_keys:
                continue
            checkpoint = root / selected["checkpoint"]
            print(
                f"fixed-std common validation scale={key[0]} "
                f"architecture={key[1]} epoch={key[2]}"
            )
            row = evaluate_checkpoint(
                selected["architecture"],
                checkpoint,
                common_data,
                root,
                device,
                batch_size,
                rollout_stride,
                max_rollout_starts,
                fixed_std,
            )
            fixed_rows.append({"data_scale": key[0], **row})
            fixed_keys.add(key)
            _write_json(progress_path, progress)
        del common_data
        fixed_rows = sorted(
            fixed_rows,
            key=lambda row: (int(row["data_scale"]), row["architecture"]),
        )
        _write_csv(
            metrics_dir / "world_model_data_scaling_common_validation_fixed_std.csv",
            fixed_rows,
        )
        common_by_key = {
            (int(row["data_scale"]), row["architecture"]): row for row in common_rows
        }
        fixed_by_key = {
            (int(row["data_scale"]), row["architecture"]): row for row in fixed_rows
        }
        combined_rows = []
        for selected in selected_rows:
            common = common_by_key[(int(selected["data_scale"]), selected["architecture"])]
            fixed = fixed_by_key[(int(selected["data_scale"]), selected["architecture"])]
            combined_rows.append(
                {
                    **selected,
                    "common_one_step_NRMSE": common["one_step_NRMSE"],
                    "common_NRMSE_H5": common["NRMSE_H5"],
                    "common_NRMSE_H10": common["NRMSE_H10"],
                    "common_NRMSE_H20": common["NRMSE_H20"],
                    "common_NRMSE_H50": common["NRMSE_H50"],
                    "common_mean_NRMSE_H5_H10_H20": common[
                        "mean_NRMSE_H5_H10_H20"
                    ],
                    "fixed_common_one_step_NRMSE": fixed["one_step_NRMSE"],
                    "fixed_common_NRMSE_H5": fixed["NRMSE_H5"],
                    "fixed_common_NRMSE_H10": fixed["NRMSE_H10"],
                    "fixed_common_NRMSE_H20": fixed["NRMSE_H20"],
                    "fixed_common_NRMSE_H50": fixed["NRMSE_H50"],
                    "fixed_common_mean_NRMSE_H5_H10_H20": fixed[
                        "mean_NRMSE_H5_H10_H20"
                    ],
                }
            )
        _write_csv(metrics_dir / "world_model_data_scaling_models.csv", combined_rows)
        _plot(
            root / "outputs" / "figures" / "world_model_data_scaling",
            fixed_rows,
        )
    return progress
