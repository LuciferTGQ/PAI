from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any, Dict

import matplotlib.pyplot as plt
import numpy as np
from scipy import stats


def _read_json(path: Path) -> Dict[str, Any]:
    with path.open(encoding="utf-8") as handle:
        return json.load(handle)


def _paired_summary(candidate: np.ndarray, reference: np.ndarray) -> Dict[str, Any]:
    difference = candidate - reference
    interval = stats.t.interval(
        0.95,
        len(difference) - 1,
        loc=float(difference.mean()),
        scale=float(stats.sem(difference)),
    )
    return {
        "mean_improvement": float(difference.mean()),
        "std_improvement": float(difference.std(ddof=1)),
        "ci95": [float(interval[0]), float(interval[1])],
        "paired_t_p": float(stats.ttest_rel(candidate, reference).pvalue),
        "wins": int(np.count_nonzero(difference > 0)),
        "episodes": int(len(difference)),
    }


def analyze_epoch_study(root: str | Path) -> Dict[str, Any]:
    root = Path(root)
    metrics = root / "outputs" / "metrics"
    figures = root / "outputs" / "figures"
    with (metrics / "world_model_ib_m1000_100e_training.csv").open(
        encoding="utf-8"
    ) as handle:
        history = [
            {"epoch": int(row["epoch"]), "train": float(row["train_mse"]),
             "validation": float(row["val_mse"])}
            for row in csv.DictReader(handle)
        ]
    train = np.asarray([row["train"] for row in history])
    validation = np.asarray([row["validation"] for row in history])
    epochs = np.asarray([row["epoch"] for row in history])
    best_index = int(validation.argmin())

    prediction_files = {
        "epoch_4": ("world_model_ib_m1000_one_step.json", "world_model_ib_m1000_multi_step.json"),
        "epoch_11": ("world_model_ib_m1000_100e_one_step.json", "world_model_ib_m1000_100e_multi_step.json"),
        "epoch_100": ("world_model_ib_m1000_100e_one_step_last.json", "world_model_ib_m1000_100e_multi_step_last.json"),
    }
    prediction = {
        name: {"one_step": _read_json(metrics / files[0]),
               "multi_step": _read_json(metrics / files[1])}
        for name, files in prediction_files.items()
    }
    strategy_files = {
        "epoch_4": "strategy_ib_m1000.json",
        "epoch_11": "strategy_ib_m1000_wm100e_best.json",
        "epoch_100": "strategy_ib_m1000_wm100e_last.json",
    }
    strategy = {name: _read_json(metrics / filename) for name, filename in strategy_files.items()}
    bc_returns = np.asarray(strategy["epoch_4"]["official_bc"]["returns"])
    control_returns = {
        name: np.asarray(values["world_model_cem"]["returns"])
        for name, values in strategy.items()
    }

    result: Dict[str, Any] = {
        "training": {
            "epochs": int(len(history)),
            "best_epoch": int(epochs[best_index]),
            "best_validation_mse": float(validation[best_index]),
            "epoch_5_validation_mse": float(validation[4]),
            "epoch_100_validation_mse": float(validation[-1]),
            "epoch_100_train_mse": float(train[-1]),
            "epoch_100_generalization_gap": float(validation[-1] - train[-1]),
            "validation_slope_last_20": float(np.polyfit(epochs[-20:], validation[-20:], 1)[0]),
        },
        "prediction": prediction,
        "control": {
            name: {
                "mean": float(values.mean()),
                "std": float(values.std()),
                "vs_bc": _paired_summary(values, bc_returns),
            }
            for name, values in control_returns.items()
        },
        "paired_control": {
            "epoch_11_vs_epoch_4": _paired_summary(
                control_returns["epoch_11"], control_returns["epoch_4"]
            ),
            "epoch_100_vs_epoch_4": _paired_summary(
                control_returns["epoch_100"], control_returns["epoch_4"]
            ),
            "epoch_100_vs_epoch_11": _paired_summary(
                control_returns["epoch_100"], control_returns["epoch_11"]
            ),
        },
    }

    output_json = metrics / "world_model_ib_m1000_100e_analysis.json"
    with output_json.open("w", encoding="utf-8") as handle:
        json.dump(result, handle, indent=2)

    figure, axes = plt.subplots(2, 2, figsize=(12, 8.5))
    axes[0, 0].plot(epochs, train, label="train")
    axes[0, 0].plot(epochs, validation, label="validation")
    axes[0, 0].axvline(epochs[best_index], color="tab:green", linestyle="--", label="one-step best")
    axes[0, 0].set(title="100-epoch training trend", xlabel="Epoch", ylabel="Normalized MSE")
    axes[0, 0].legend()

    axes[0, 1].plot(epochs, validation, color="tab:orange")
    for epoch in (4, 11, 100):
        axes[0, 1].scatter(epoch, validation[epoch - 1], label=f"epoch {epoch}")
    axes[0, 1].set(title="Validation trend", xlabel="Epoch", ylabel="Normalized MSE")
    axes[0, 1].legend()

    for name, values in prediction.items():
        horizons = values["multi_step"]["horizons"]
        x = [int(horizon) for horizon in horizons]
        y = [horizons[str(horizon)]["mse"] for horizon in x]
        axes[1, 0].plot(x, y, marker="o", label=name.replace("_", " "))
    axes[1, 0].set(title="Free-running prediction", xlabel="Horizon", ylabel="MSE")
    axes[1, 0].legend()

    axes[1, 1].boxplot(
        [bc_returns, control_returns["epoch_4"], control_returns["epoch_11"], control_returns["epoch_100"]],
        labels=["BC", "epoch 4", "epoch 11", "epoch 100"],
        showmeans=True,
    )
    axes[1, 1].set(title="Matched-seed simulator return", ylabel="Return (higher is better)")
    for axis in axes.flat:
        axis.grid(alpha=0.25)
    figure.tight_layout()
    figures.mkdir(parents=True, exist_ok=True)
    figure.savefig(figures / "world_model_ib_m1000_100e_analysis.png", dpi=170)
    plt.close(figure)
    return result
