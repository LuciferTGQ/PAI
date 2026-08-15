from __future__ import annotations

import csv
import json
import math
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any, Iterable

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch, Rectangle


ROOT = Path(__file__).resolve().parents[1]
REPORT = ROOT / "report"
FIGURES = REPORT / "figures"
GENERATED = REPORT / "generated"
SOURCE_DATA = REPORT / "source_data"
FIGURES.mkdir(parents=True, exist_ok=True)
GENERATED.mkdir(parents=True, exist_ok=True)
SOURCE_DATA.mkdir(parents=True, exist_ok=True)
sys.path.insert(0, str(ROOT))

from src.data.ib_dataset import load_ib_npz, trajectory_spans  # noqa: E402
from src.world_model.interface import FrozenWorldModel  # noqa: E402


ARCHITECTURES = ["MLP", "GRU", "LSTM", "Transformer-2L", "Transformer-4L"]
SCALES = ["M100", "M1000", "M10000"]
STRATEGIES = ["CEM", "iCEM", "MPPI", "MB-PPO"]
ARCH_COLORS = {
    "MLP": "#7A7A7A",
    "GRU": "#3B6FB6",
    "LSTM": "#8C6BB1",
    "Transformer-2L": "#2A9D8F",
    "Transformer-4L": "#E69F00",
}
STRATEGY_COLORS = {
    "BC": "#6C757D",
    "CEM": "#3B6FB6",
    "iCEM": "#E69F00",
    "MPPI": "#2A9D8F",
    "MB-PPO": "#8C6BB1",
}


def configure_style() -> None:
    mpl.rcParams.update(
        {
            "font.family": "sans-serif",
            "font.sans-serif": ["Arial", "Microsoft YaHei", "DejaVu Sans"],
            "font.size": 8,
            "axes.titlesize": 9,
            "axes.labelsize": 8,
            "xtick.labelsize": 7,
            "ytick.labelsize": 7,
            "legend.fontsize": 7,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.linewidth": 0.8,
            "legend.frameon": False,
            "svg.fonttype": "none",
            "pdf.fonttype": 42,
            "figure.facecolor": "white",
            "axes.facecolor": "white",
            "savefig.facecolor": "white",
        }
    )


def save_figure(fig: mpl.figure.Figure, stem: str) -> None:
    for suffix, kwargs in (
        ("pdf", {}),
        ("svg", {}),
        ("png", {"dpi": 600}),
    ):
        fig.savefig(FIGURES / f"{stem}.{suffix}", bbox_inches="tight", **kwargs)
    plt.close(fig)


def normalize_generated_text() -> None:
    """Keep generated text assets deterministic and free of line-end whitespace."""
    paths = list(FIGURES.glob("*.svg")) + list(GENERATED.glob("*.tex"))
    for path in paths:
        content = path.read_text(encoding="utf-8")
        normalized = "\n".join(line.rstrip() for line in content.splitlines()) + "\n"
        path.write_text(normalized, encoding="utf-8")


def panel_label(ax: mpl.axes.Axes, label: str) -> None:
    ax.text(
        -0.12,
        1.04,
        label,
        transform=ax.transAxes,
        fontsize=10,
        fontweight="bold",
        va="bottom",
    )


def read_world_models() -> pd.DataFrame:
    frame = pd.read_csv(ROOT / "outputs/metrics/world_model_5x3_common_validation_models.csv")
    frame = frame[frame["selected_within_architecture"] == True].copy()  # noqa: E712
    frame["scale"] = "M" + frame["data_scale"].astype(str)
    return frame


def figure_1_framework() -> None:
    fig, ax = plt.subplots(figsize=(7.0, 3.35))
    ax.set_xlim(0, 14)
    ax.set_ylim(0, 7)
    ax.axis("off")
    phases = [
        (0.3, 4.7, 3.1, 1.25, "Historical trajectories", "30 x 6 history + 3D action", "#EAF1F8"),
        (4.1, 4.7, 3.2, 1.25, "World Model", "predict next 6D frame", "#E4F3F0"),
        (8.1, 4.7, 2.4, 1.25, "Frozen model", "recursive rollout", "#E4F3F0"),
        (11.1, 4.7, 2.5, 1.25, "NeoRL simulator", "1000-step control", "#F8EFD9"),
    ]
    for x, y, w, h, title, sub, color in phases:
        box = FancyBboxPatch(
            (x, y), w, h, boxstyle="round,pad=0.08,rounding_size=0.1",
            facecolor=color, edgecolor="#4C5964", linewidth=1.0,
        )
        ax.add_patch(box)
        ax.text(x + w / 2, y + 0.77, title, ha="center", va="center", fontweight="bold", fontsize=9)
        ax.text(x + w / 2, y + 0.32, sub, ha="center", va="center", color="#4C5964", fontsize=7)
    for x1, x2 in ((3.4, 4.1), (7.3, 8.1), (10.5, 11.1)):
        ax.add_patch(FancyArrowPatch((x1, 5.33), (x2, 5.33), arrowstyle="-|>", mutation_scale=12, color="#4C5964"))

    gate = FancyBboxPatch(
        (4.25, 2.55), 5.95, 1.15, boxstyle="round,pad=0.08,rounding_size=0.1",
        facecolor="#F5F6F7", edgecolor="#2A9D8F", linewidth=1.5,
    )
    ax.add_patch(gate)
    ax.text(7.225, 3.25, "Dynamics validation only", ha="center", va="center", fontweight="bold", color="#1D746B")
    ax.text(7.225, 2.82, "one-step gate -> minimize mean(H5, H10, H20); H50 diagnostic", ha="center", va="center", fontsize=7)
    ax.add_patch(FancyArrowPatch((5.7, 4.7), (5.7, 3.7), arrowstyle="-|>", mutation_scale=11, color="#2A9D8F"))
    ax.add_patch(FancyArrowPatch((8.7, 3.7), (8.7, 4.7), arrowstyle="-|>", mutation_scale=11, color="#2A9D8F"))

    plan = FancyBboxPatch(
        (3.0, 0.45), 3.45, 1.05, boxstyle="round,pad=0.06,rounding_size=0.08",
        facecolor="#EAF1F8", edgecolor="#3B6FB6", linewidth=1.0,
    )
    rl = FancyBboxPatch(
        (7.15, 0.45), 3.45, 1.05, boxstyle="round,pad=0.06,rounding_size=0.08",
        facecolor="#EEE8F5", edgecolor="#8C6BB1", linewidth=1.0,
    )
    ax.add_patch(plan)
    ax.add_patch(rl)
    ax.text(4.725, 1.12, "Model-Based Planning", ha="center", fontweight="bold")
    ax.text(4.725, 0.72, "CEM / iCEM / MPPI", ha="center", fontsize=7)
    ax.text(8.875, 1.12, "Model-Based RL", ha="center", fontweight="bold")
    ax.text(8.875, 0.72, "MB-PPO + behavior KL", ha="center", fontsize=7)
    ax.add_patch(FancyArrowPatch((8.7, 4.7), (5.0, 1.5), arrowstyle="-|>", mutation_scale=10, color="#3B6FB6", connectionstyle="arc3,rad=0.14"))
    ax.add_patch(FancyArrowPatch((8.9, 4.7), (8.9, 1.5), arrowstyle="-|>", mutation_scale=10, color="#8C6BB1"))
    ax.add_patch(FancyArrowPatch((6.45, 0.98), (11.8, 4.7), arrowstyle="-|>", mutation_scale=10, color="#3B6FB6", connectionstyle="arc3,rad=-0.14"))
    ax.add_patch(FancyArrowPatch((10.6, 0.98), (12.5, 4.7), arrowstyle="-|>", mutation_scale=10, color="#8C6BB1", connectionstyle="arc3,rad=-0.12"))
    ax.text(11.9, 3.2, "reward is used here,\nnot for model selection", ha="center", va="center", fontsize=7, color="#A65F00")
    save_figure(fig, "fig1_framework")


def figure_2_heatmaps(wm: pd.DataFrame) -> None:
    one = wm.pivot(index="architecture", columns="scale", values="one_step_NRMSE").loc[ARCHITECTURES, SCALES]
    rollout = wm.pivot(index="architecture", columns="scale", values="mean_NRMSE_H5_H10_H20").loc[ARCHITECTURES, SCALES]
    selected = {("GRU", "M100"), ("Transformer-2L", "M1000"), ("Transformer-2L", "M10000")}
    fig, axes = plt.subplots(1, 2, figsize=(7.1, 3.55), constrained_layout=True)
    for label, ax, data, title, fmt in (
        ("a", axes[0], one, "One-step NRMSE", ".3f"),
        ("b", axes[1], rollout, "Mean NRMSE at H5/H10/H20", ".3f"),
    ):
        image = ax.imshow(data.values, cmap="Blues", aspect="auto")
        ax.set_xticks(range(len(SCALES)), SCALES)
        ax.set_yticks(range(len(ARCHITECTURES)), ARCHITECTURES)
        ax.set_title(title, fontweight="bold", pad=8)
        for row in range(len(ARCHITECTURES)):
            for col in range(len(SCALES)):
                value = data.iloc[row, col]
                text_color = "white" if value > np.quantile(data.values, 0.68) else "#20262C"
                ax.text(col, row, format(value, fmt), ha="center", va="center", color=text_color, fontsize=7)
                if (ARCHITECTURES[row], SCALES[col]) in selected:
                    ax.add_patch(Rectangle((col - 0.48, row - 0.48), 0.96, 0.96, fill=False, edgecolor="#E69F00", linewidth=2.0))
        cbar = fig.colorbar(image, ax=ax, fraction=0.045, pad=0.03)
        cbar.ax.tick_params(labelsize=6)
        panel_label(ax, label)
    axes[0].set_ylabel("Architecture")
    fig.text(0.5, -0.01, "Lower is better; orange borders mark validation-selected models", ha="center", fontsize=7, color="#4C5964")
    save_figure(fig, "fig2_world_model_heatmaps")


def figure_3_multistep(wm: pd.DataFrame) -> None:
    subset = wm[wm["scale"] == "M1000"].set_index("architecture").loc[ARCHITECTURES]
    horizons = [1, 5, 10, 20, 50]
    x = np.arange(len(horizons))
    fig, ax = plt.subplots(figsize=(6.6, 3.7))
    for architecture in ARCHITECTURES:
        values = [subset.loc[architecture, f"NRMSE_H{h}"] for h in horizons]
        ax.plot(
            x, values, marker="o", markersize=4.8, linewidth=1.5,
            color=ARCH_COLORS[architecture], label=architecture,
        )
    ax.set_xticks(x, [f"H{h}" for h in horizons])
    ax.set_xlabel("Evaluated recursive-rollout horizon")
    ax.set_ylabel("NRMSE")
    ax.set_title("M1000: recursive error accumulation", fontweight="bold", loc="left")
    ax.grid(axis="y", alpha=0.22)
    ax.legend(ncol=3, loc="upper left")
    ax.text(0.99, 0.02, "Markers are the five measured horizons; lines are visual guides only.", transform=ax.transAxes, ha="right", va="bottom", fontsize=6.5, color="#4C5964")
    save_figure(fig, "fig3_multistep_error")


def representative_rollout() -> pd.DataFrame:
    output = GENERATED / "representative_h50_rollout.csv"
    if output.exists():
        return pd.read_csv(output)
    data = load_ib_npz(ROOT / "data/raw/ib-medium-1000-val.npz")
    spans = trajectory_spans(data["index"], len(data["obs"]))
    starts = np.asarray(
        [start for left, right in spans for start in range(int(left), int(right - 50 + 1))],
        dtype=np.int64,
    )
    candidate_positions = np.linspace(0, len(starts) - 1, 101, dtype=np.int64)
    candidates = starts[candidate_positions]
    checkpoint = ROOT / "outputs/checkpoints/world_model_ib_m1000_transformer2_fair_epoch_020.pt"
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = FrozenWorldModel(checkpoint, device=device)
    histories = data["obs"][candidates].copy()
    predicted_steps = []
    actual_steps = []
    for step in range(50):
        predicted = model.predict_next_frame(histories, data["action"][candidates + step])
        actual = data["next_obs"][candidates + step, :6]
        predicted_steps.append(predicted.copy())
        actual_steps.append(actual.copy())
        histories = np.concatenate((predicted, histories[:, :-6]), axis=1)
    predicted_array = np.stack(predicted_steps, axis=1)
    actual_array = np.stack(actual_steps, axis=1)
    normalized = (predicted_array - actual_array) / model.stats.target_std.reshape(1, 1, -1)
    errors = np.sqrt(np.mean(np.square(normalized), axis=(1, 2)))
    median_error = np.median(errors)
    representative = int(np.argmin(np.abs(errors - median_error)))
    names = ["setpoint", "velocity", "gain", "shift", "fatigue", "consumption"]
    rows = []
    for step in range(50):
        for index, name in enumerate(names):
            rows.append(
                {
                    "step": step + 1,
                    "variable": name,
                    "ground_truth": float(actual_array[representative, step, index]),
                    "prediction": float(predicted_array[representative, step, index]),
                    "candidate_start": int(candidates[representative]),
                    "candidate_rollout_nrmse": float(errors[representative]),
                    "selection_rule": "closest to median H50 rollout NRMSE among 101 deterministic starts",
                }
            )
    frame = pd.DataFrame(rows)
    frame.to_csv(output, index=False)
    return frame


def figure_4_rollout(trace: pd.DataFrame) -> None:
    variables = [("velocity", "Velocity"), ("fatigue", "Fatigue"), ("consumption", "Consumption")]
    fig, axes = plt.subplots(3, 1, figsize=(7.0, 5.6), sharex=True, constrained_layout=True)
    for label, ax, (key, display) in zip("abc", axes, variables):
        data = trace[trace["variable"] == key]
        ax.plot(data["step"], data["ground_truth"], color="#252A2E", linewidth=1.5, label="Ground truth")
        ax.plot(data["step"], data["prediction"], color="#2A9D8F", linewidth=1.5, linestyle="--", label="Prediction")
        ax.set_ylabel(display)
        ax.grid(alpha=0.2)
        panel_label(ax, label)
    axes[0].legend(loc="upper left", ncol=2)
    axes[-1].set_xlabel("Recursive rollout step")
    error = float(trace["candidate_rollout_nrmse"].iloc[0])
    fig.suptitle(f"Representative H50 rollout (median-error candidate; NRMSE={error:.3f})", fontsize=10, fontweight="bold")
    fig.text(0.99, 0.002, "Recorded actions; median-error deterministic validation candidate", ha="right", fontsize=6.5, color="#4C5964")
    save_figure(fig, "fig4_representative_rollout")


def _method_name(value: str) -> str:
    return "MB-PPO" if value == "MB-PPO+KL" else value


def _point_summary(ax: mpl.axes.Axes, values: Iterable[float], x: float, color: str, jitter_seed: int) -> None:
    array = np.asarray(list(values), dtype=np.float64)
    rng = np.random.default_rng(jitter_seed)
    jitter = rng.uniform(-0.085, 0.085, size=len(array))
    ax.scatter(np.full(len(array), x) + jitter, array, s=20, facecolor="white", edgecolor=color, linewidth=1.0, alpha=0.95, zorder=3)
    ax.errorbar(x, array.mean(), yerr=array.std(), fmt="o", color=color, markerfacecolor=color, markersize=5, capsize=3, linewidth=1.2, zorder=4)


def figure_5_strategy() -> None:
    episodes = pd.read_csv(ROOT / "outputs/metrics/main_system_matrix_development_episodes.csv")
    episodes["strategy"] = episodes["strategy"].map(_method_name)
    fig, axes = plt.subplots(1, 3, figsize=(7.2, 3.65), sharey=False)
    for panel_index, (label, scale, ax) in enumerate(zip("abc", ["M-100", "M-1000", "M-10000"], axes)):
        subset = episodes[episodes["dataset_scale"] == scale]
        for index, strategy in enumerate(STRATEGIES):
            values = subset[subset["strategy"] == strategy]["episode_return"].values
            if scale == "M-100" and strategy == "iCEM":
                ax.scatter(index, -374000, marker="v", s=35, color=STRATEGY_COLORS[strategy], clip_on=False, zorder=4)
                ax.text(index, -366000, "off-scale", rotation=90, va="bottom", ha="center", fontsize=6.2, color=STRATEGY_COLORS[strategy])
                continue
            _point_summary(ax, values, index, STRATEGY_COLORS[strategy], 100 + panel_index * 10 + index)
        ax.set_xticks(range(4), STRATEGIES, rotation=24, ha="right")
        ax.set_title(scale, fontweight="bold")
        ax.grid(axis="y", alpha=0.22)
        panel_label(ax, label)
        if scale == "M-100":
            ax.set_ylim(-375000, -255000)
            inset = ax.inset_axes([0.07, 0.08, 0.24, 0.30])
            bad = subset[subset["strategy"] == "iCEM"]["episode_return"].values
            _point_summary(inset, bad, 0, STRATEGY_COLORS["iCEM"], 222)
            inset.set_xlim(-0.3, 0.3)
            inset.set_ylim(-2_665_000, -2_630_000)
            inset.set_xticks([])
            inset.set_yticks([-2_660_000, -2_640_000])
            inset.yaxis.set_major_formatter(
                mpl.ticker.FuncFormatter(lambda value, _pos: f"{value / 1e6:.3f}M")
            )
            inset.set_title("iCEM", fontsize=6.5)
            inset.tick_params(axis="y", labelsize=5.5)
            inset.grid(axis="y", alpha=0.18)
        else:
            ax.set_ylim(-390000, -195000)
    axes[0].set_ylabel("Episode cumulative reward (higher is better)")
    fig.text(0.5, -0.025, "Dots: development seeds 42-46; filled marker and error bar: mean +/- population SD", ha="center", fontsize=6.7, color="#4C5964")
    fig.tight_layout()
    save_figure(fig, "fig5_strategy_comparison")


def figure_6_cross() -> None:
    episodes = pd.read_csv(ROOT / "outputs/metrics/world_model_control_cross_m1000_episodes.csv")
    episodes["strategy"] = episodes["strategy"].map(_method_name)
    fig, axes = plt.subplots(1, 2, figsize=(7.2, 3.55), sharey=False, constrained_layout=True)
    for panel, label, strategy in zip(axes, "ab", ["MPPI", "MB-PPO"]):
        subset = episodes[episodes["strategy"] == strategy]
        for index, architecture in enumerate(ARCHITECTURES):
            values = subset[subset["world_model_architecture"] == architecture]["episode_return"].values
            _point_summary(panel, values, index, ARCH_COLORS[architecture], 300 + index)
        panel.set_xticks(
            range(5),
            ["MLP", "GRU", "LSTM", "Transformer-\n2L", "Transformer-\n4L"],
            rotation=16,
            ha="right",
        )
        panel.set_title(strategy, fontweight="bold")
        panel.set_ylabel("Episode cumulative reward")
        panel.grid(axis="y", alpha=0.22)
        panel_label(panel, label)
    axes[0].set_ylim(-355000, -200000)
    axes[1].set_ylim(-291000, -274000)
    fig.text(0.5, -0.02, "M1000; only World Model architecture changes within each panel; n=5 seeds", ha="center", fontsize=6.7, color="#4C5964")
    save_figure(fig, "fig6_world_model_strategy_cross")


def figure_7_final() -> None:
    behavior = np.asarray(
        json.loads((ROOT / "outputs/metrics/strategy_ib_m1000.json").read_text(encoding="utf-8"))["original_behavior_reference"]["returns"],
        dtype=np.float64,
    )
    episodes = pd.read_csv(ROOT / "outputs/metrics/final_selected_systems_fresh_seeds_episodes.csv")
    groups = [
        ("BC", episodes[episodes["strategy"] == "BC"]["episode_return"].values, STRATEGY_COLORS["BC"]),
        ("M100\nGRU + CEM", episodes[episodes["dataset_scale"] == "M-100"]["episode_return"].values, STRATEGY_COLORS["CEM"]),
        ("M1000\nTransformer-2L\n+ MPPI", episodes[episodes["dataset_scale"] == "M-1000"]["episode_return"].values, STRATEGY_COLORS["MPPI"]),
        ("M10000\nTransformer-2L\n+ MPPI", episodes[episodes["dataset_scale"] == "M-10000"]["episode_return"].values, "#17766E"),
    ]
    fig, ax = plt.subplots(figsize=(7.2, 4.0))
    box = ax.boxplot(
        [behavior], positions=[0], widths=0.42, patch_artist=True, showfliers=False,
        medianprops={"color": "#333333", "linewidth": 1.2},
        boxprops={"facecolor": "#F3F4F5", "edgecolor": "#6C757D", "linewidth": 1.2},
        whiskerprops={"color": "#6C757D"}, capprops={"color": "#6C757D"},
    )
    del box
    for index, (name, values, color) in enumerate(groups, start=1):
        _point_summary(ax, values, index, color, 400 + index)
    ax.set_xticks(
        range(5),
        ["Original Behavior\n(dataset trajectories)"] + [group[0] for group in groups],
        rotation=12,
        ha="right",
    )
    ax.set_ylabel("Episode cumulative reward (higher is better)")
    ax.set_ylim(-297000, -210000)
    ax.grid(axis="y", alpha=0.22)
    ax.text(0, -295000, "n=1000\nunpaired", ha="center", va="bottom", fontsize=6.5, color="#6C757D")
    ax.text(2.5, -295000, "online NeoRL simulator: n=10 untouched seeds each", ha="center", va="bottom", fontsize=6.5, color="#4C5964")
    ax.set_title("Final long-horizon control evaluation", fontweight="bold", loc="left")
    save_figure(fig, "fig7_final_simulator")


def figure_8_kl() -> None:
    payload = json.loads((ROOT / "outputs/metrics/strategy_ib_m1000_gru_mbppo.json").read_text(encoding="utf-8"))
    with_kl = np.asarray(payload["methods"]["mbppo_with_kl"]["returns"], dtype=np.float64)
    without_kl = np.asarray(payload["methods"]["mbppo_without_kl"]["returns"], dtype=np.float64)
    hist_with = pd.read_csv(ROOT / "outputs/metrics/mbppo_ib_m1000_gru_with_kl_training.csv")
    hist_without = pd.read_csv(ROOT / "outputs/metrics/mbppo_ib_m1000_gru_without_kl_training.csv")
    fig, axes = plt.subplots(1, 2, figsize=(7.2, 3.6), constrained_layout=True)
    for index, (name, values, color) in enumerate(
        [("MB-PPO", without_kl, "#C44E52"), ("MB-PPO + KL", with_kl, STRATEGY_COLORS["MB-PPO"])]
    ):
        _point_summary(axes[0], values, index, color, 500 + index)
    axes[0].set_xticks([0, 1], ["MB-PPO\nwithout KL", "MB-PPO\nwith KL"])
    axes[0].set_ylabel("Episode cumulative reward")
    axes[0].set_title("Simulator outcome", fontweight="bold")
    axes[0].grid(axis="y", alpha=0.22)
    panel_label(axes[0], "a")
    axes[1].plot(hist_without["gradient_step"], hist_without["behavior_kl"], color="#C44E52", linewidth=1.4, label="without KL")
    axes[1].plot(hist_with["gradient_step"], hist_with["behavior_kl"], color=STRATEGY_COLORS["MB-PPO"], linewidth=1.4, label="with KL")
    axes[1].set_yscale("symlog", linthresh=0.1)
    axes[1].set_xlabel("PPO gradient step")
    axes[1].set_ylabel(r"Behavior divergence $D_{KL}(\pi_b\Vert\pi)$")
    axes[1].set_title("Policy departure from behavior", fontweight="bold")
    axes[1].grid(alpha=0.22)
    axes[1].legend()
    panel_label(axes[1], "b")
    save_figure(fig, "fig8_mbppo_kl_ablation")


def latex_num(value: float, decimals: int = 3) -> str:
    return f"{value:.{decimals}f}"


def latex_return(mean: float, std: float) -> str:
    return f"{mean:,.0f} $\\pm$ {std:,.0f}"


def generate_tables(wm: pd.DataFrame) -> None:
    architecture_rows = []
    config = {
        "MLP": ("无显式时序递归", "2$\\times$256 MLP, dropout 0.1"),
        "GRU": ("门控递归状态", "2层, hidden 128, action branch 64"),
        "LSTM": ("带记忆单元的递归状态", "2层, hidden 128, action branch 64"),
        "Transformer-2L": ("自注意力历史编码", "$d=64$, 4 heads, FFN 128, 2层"),
        "Transformer-4L": ("自注意力历史编码", "$d=64$, 4 heads, FFN 128, 4层"),
    }
    for architecture in ARCHITECTURES:
        row = wm[wm["architecture"] == architecture].iloc[0]
        architecture_rows.append(
            f"{architecture} & {config[architecture][0]} & {config[architecture][1]} & {int(row['parameter_count']):,} \\\\"
        )
    table1 = r"""
\begin{table}[H]
\centering
\caption{World Model architectures. 参数量不随数据规模改变。}
\label{tab:architectures}
\small
\begin{tabularx}{\textwidth}{lXXr}
\toprule
Architecture & Temporal modeling & Main configuration & Parameters \\
\midrule
""" + "\n".join(architecture_rows) + r"""
\bottomrule
\end{tabularx}
\end{table}
"""

    selected = wm[wm["selected_best_world_model"] == True].copy()  # noqa: E712
    selected = selected.set_index("scale").loc[SCALES].reset_index()
    rows = []
    for _, row in selected.iterrows():
        rows.append(
            f"{row['scale']} & {row['architecture']} & {int(row['epoch'])} & "
            f"{row['one_step_NRMSE']:.3f} & {row['NRMSE_H5']:.3f} & {row['NRMSE_H10']:.3f} & "
            f"{row['NRMSE_H20']:.3f} & {row['NRMSE_H50']:.3f} \\\\"
        )
    table2 = r"""
\begin{table}[H]
\centering
\caption{Validation-only protocol selected World Models. H50仅用于长期稳定性诊断。}
\label{tab:selected_wm}
\small
\begin{tabular}{lllrrrrr}
\toprule
Dataset & Architecture & Epoch & One-step & H5 & H10 & H20 & H50 \\
\midrule
""" + "\n".join(rows) + r"""
\bottomrule
\end{tabular}
\end{table}
"""

    depth = pd.read_csv(ROOT / "outputs/metrics/world_model_5x3_common_validation_transformer_depth.csv")
    depth_rows = []
    for scale in ["M-100", "M-1000", "M-10000"]:
        scale_number = int(scale.replace("M-", ""))
        row = depth[depth["data_scale"] == scale_number].iloc[0]
        depth_rows.append(
            f"{scale.replace('-', '')} & {row['transformer2_one_step_NRMSE']:.3f} / {row['transformer2_mean_NRMSE_H5_H10_H20']:.3f} & "
            f"{row['transformer4_one_step_NRMSE']:.3f} / {row['transformer4_mean_NRMSE_H5_H10_H20']:.3f} & "
            f"{row['delta_4L_minus_2L_one_step_NRMSE']:+.3f} / {row['delta_4L_minus_2L_mean_NRMSE_H5_H10_H20']:+.3f} \\\\"
        )
    table3 = r"""
\begin{table}[H]
\centering
\caption{Transformer depth ablation. 单元格为 one-step / mean(H5,H10,H20)；差值为4L减2L，负值更优。}
\label{tab:depth}
\small
\begin{tabular}{lccc}
\toprule
Scale & Transformer-2L & Transformer-4L & Difference \\
\midrule
""" + "\n".join(depth_rows) + r"""
\bottomrule
\end{tabular}
\end{table}
"""

    main = pd.read_csv(ROOT / "outputs/metrics/main_system_matrix_development_summary.csv")
    main["strategy"] = main["strategy"].map(_method_name)
    matrix_rows = []
    for scale in ["M-100", "M-1000", "M-10000"]:
        values = main[main["dataset_scale"] == scale].set_index("strategy")
        matrix_rows.append(
            f"{scale.replace('-', '')} & "
            + " & ".join(latex_return(values.loc[strategy, "mean_return"], values.loc[strategy, "std_return"]) for strategy in STRATEGIES)
            + " \\\\"
        )
    table4 = r"""
\begin{table}[H]
\centering
\caption{Development strategy matrix (seeds 42--46). 数值为 episode cumulative reward 的 mean $\pm$ population SD，越高越好。}
\label{tab:strategy_matrix}
\scriptsize
\resizebox{\textwidth}{!}{%
\begin{tabular}{lrrrr}
\toprule
Dataset & CEM & iCEM & MPPI & MB-PPO \\
\midrule
""" + "\n".join(matrix_rows) + r"""
\bottomrule
\end{tabular}}
\end{table}
"""

    final_summary = pd.read_csv(ROOT / "outputs/metrics/final_selected_systems_fresh_seeds_summary.csv")
    final_episodes = pd.read_csv(ROOT / "outputs/metrics/final_selected_systems_fresh_seeds_episodes.csv")
    bc = final_episodes[final_episodes["strategy"] == "BC"]["episode_return"].values
    behavior = np.asarray(json.loads((ROOT / "outputs/metrics/strategy_ib_m1000.json").read_text(encoding="utf-8"))["original_behavior_reference"]["returns"])
    final_rows = [
        f"原始行为数据参考值（Original Behavior） & NeoRL dataset trajectories & {latex_return(behavior.mean(), behavior.std())} & -- & -- \\\\ ",
        f"BC & NeoRL simulator & {latex_return(bc.mean(), bc.std())} & 0 & -- \\\\ ",
    ]
    names = {
        "M-100": "M100: GRU + CEM",
        "M-1000": "M1000: Transformer-2L + MPPI",
        "M-10000": "M10000: Transformer-2L + MPPI",
    }
    for _, row in final_summary.iterrows():
        final_rows.append(
            f"{names[row['dataset_scale']]} & NeoRL simulator & {latex_return(row['mean_return'], row['std_return'])} & "
            f"{row['mean_delta_vs_bc']:+,.0f} & {100 * row['win_rate_vs_bc']:.0f}\\% \\\\"
        )
    table5 = r"""
\begin{table}[H]
\centering
\caption{Final NeoRL evaluation. Online systems use untouched seeds 100--109. Original Behavior来自数据集已有轨迹，不是同seed在线部署。}
\label{tab:final}
\scriptsize
\resizebox{\textwidth}{!}{%
\begin{tabular}{llrrr}
\toprule
Method/System & Evaluation source & Episode return & $\Delta$ vs BC & Win rate \\
\midrule
""" + "\n".join(final_rows) + r"""
\bottomrule
\end{tabular}}
\end{table}
"""

    t2 = wm[(wm["scale"] == "M1000") & (wm["architecture"] == "Transformer-2L")].iloc[0]
    variable_names = ["setpoint", "velocity", "gain", "shift", "fatigue", "consumption"]
    variable_rows = []
    for variable in variable_names:
        variable_rows.append(
            f"{variable.capitalize()} & {t2[f'one_step_NRMSE_{variable}']:.3f} & "
            f"{t2[f'NRMSE_H20_{variable}']:.3f} & {t2[f'NRMSE_H50_{variable}']:.3f} \\\\"
        )
    per_variable = r"""
\begin{table}[H]
\centering
\caption{M1000 Transformer-2L的per-variable NRMSE。变量的误差积累速度明显不同。}
\label{tab:per_variable}
\small
\begin{tabular}{lrrr}
\toprule
Variable & One-step & H20 & H50 \\
\midrule
""" + "\n".join(variable_rows) + r"""
\bottomrule
\end{tabular}
\end{table}
"""

    (GENERATED / "tables.tex").write_text(
        "\n".join([table1, table2, table3, table4, table5, per_variable]), encoding="utf-8"
    )
    for name, content in (
        ("table1_architectures.tex", table1),
        ("table2_selected_world_models.tex", table2),
        ("table3_depth_ablation.tex", table3),
        ("table4_strategy_matrix.tex", table4),
        ("table5_final_evaluation.tex", table5),
        ("table_per_variable.tex", per_variable),
    ):
        (GENERATED / name).write_text(content, encoding="utf-8")


def figure_notes(wm: pd.DataFrame, trace: pd.DataFrame) -> None:
    main = pd.read_csv(ROOT / "outputs/metrics/main_system_matrix_development_summary.csv")
    cross = pd.read_csv(ROOT / "outputs/metrics/world_model_control_cross_m1000_summary.csv")
    final = pd.read_csv(ROOT / "outputs/metrics/final_selected_systems_fresh_seeds_summary.csv")
    behavior = np.asarray(json.loads((ROOT / "outputs/metrics/strategy_ib_m1000.json").read_text(encoding="utf-8"))["original_behavior_reference"]["returns"])
    selected = wm[wm["selected_best_world_model"] == True].set_index("scale")  # noqa: E712
    notes = f"""# Figure interpretation drafts

## Figure 1 — Overall Framework

Question: 历史工业轨迹如何经过World Model、策略优化和simulator评价形成闭环？

Observation: World Model selection与simulator reward之间存在明确隔离；模型冻结后才进入CEM、iCEM、MPPI或MB-PPO。

Interpretation: 该隔离避免用控制回报反向挑选dynamics模型，保留了模型比较的因果清晰度。

Limitation: 流程图描述实验协议，不提供任何性能证据。

## Figure 2 — Architecture × Dataset Scale

Question: 架构和数据规模如何共同影响单步与多步预测？

Observation: M100由GRU取得最低合规综合rollout误差；M1000和M10000由Transformer-2L胜出。M10000的Transformer-2L mean(H5,H10,H20)为{selected.loc['M10000','mean_NRMSE_H5_H10_H20']:.3f}，但不同架构并未随数据量单调改善。

Interpretation: 架构归纳偏置与数据覆盖发生交互；单步与递归rollout关注的误差模式不同。

Limitation: 每格是单训练seed下的validation结果，不能替代多训练seed鲁棒性研究。

## Figure 3 — Multi-step Error Accumulation

Question: one-step NRMSE为何不足以描述递归World Model？

Observation: M1000中五个模型的one-step都约为0.17--0.19，但H20和H50明显分离；Transformer-2L在H5--H20综合最低，而Transformer-4L在H50更低。

Interpretation: 小的单步偏差在闭环递归更新中会以变量相关方式传播，短中期稳定性需要直接评价。

Limitation: 连接线不表示未测horizon处的插值性能。

## Figure 4 — Representative H50 Rollout

Question: 代表性连续轨迹上的预测漂移是什么形态？

Observation: 所示轨迹在101个确定性候选起点中最接近中位H50误差，综合NRMSE为{trace['candidate_rollout_nrmse'].iloc[0]:.3f}；velocity、fatigue和consumption呈现不同的漂移速度。

Interpretation: 同一个整体NRMSE背后可能包含非常不同的变量级误差，工业控制应监控关键变量而非只看总分。

Limitation: 一条代表轨迹不能替代全体rollout起点的统计结果。

## Figure 5 — Strategy Comparison

Question: 每个数据规模固定其dynamics-selected World Model后，哪种策略更强？

Observation: M100的CEM最高（{main[(main.dataset_scale=='M-100') & (main.strategy=='CEM')].mean_return.iloc[0]:,.0f}）；M1000和M10000的MPPI最高。M100 iCEM约为{main[(main.dataset_scale=='M-100') & (main.strategy=='iCEM')].mean_return.iloc[0]:,.0f}，出现严重失效。

Interpretation: 规划器能力与模型质量共同决定真实控制效果；优化器可能利用低质量World Model的错误高收益区域。

Limitation: iCEM的toy objective已通过，因此该图不能推出iCEM一般不适合工业过程。

## Figure 6 — World Model × Downstream Strategy

Question: dynamics prediction ranking是否完全决定downstream control ranking？

Observation: M1000中MPPI在Transformer-4L上最高（{cross[(cross.architecture=='Transformer-4L') & (cross.strategy=='MPPI')].mean_return.iloc[0]:,.0f}），MB-PPO在LSTM上最高（{cross[(cross.architecture=='LSTM') & (cross.strategy=='MB-PPO+KL')].mean_return.iloc[0]:,.0f}），而dynamics validation选择Transformer-2L。

Interpretation: planning与learned policy对不同误差结构的敏感性不同，通用预测误差与决策效用并非完全等价。

Limitation: 这是当前五模型、两策略和一个数据规模上的观察，不是普遍定律，也不用于反向修改World Model选择。

## Figure 7 — Final NeoRL Simulator Result

Question: 最终World-Model-based systems能否提高1000步累计奖励？

Observation: Original Behavior数据轨迹均值为{behavior.mean():,.0f}；三个最终系统相对BC的改善分别为{final.iloc[0].mean_delta_vs_bc:,.0f}、{final.iloc[1].mean_delta_vs_bc:,.0f}和{final.iloc[2].mean_delta_vs_bc:,.0f}，且十个最终seed的win rate均为100%。

Interpretation: 经过validation-only模型选择和冻结策略选择后，基于World Model的优化在未使用simulator seeds上保持了长期回报提升。

Limitation: Original Behavior来自数据集已有轨迹，不是与在线系统同seed部署的配对基线；结果也只覆盖NeoRL Industrial Benchmark。

## Figure 8 — MB-PPO KL Ablation

Question: behavior KL是否抑制MB-PPO利用World Model误差？

Observation: 不加KL的平均return约为-883,709，加KL后为-284,715；dense training history显示无KL策略与行为分布的偏离显著扩大。

Interpretation: 在该M1000 GRU消融中，behavior constraint显著降低了model exploitation风险。

Limitation: 消融只覆盖一个World Model和一个训练seed，不能量化最佳KL系数或保证所有场景都有效。
"""
    (GENERATED / "figure_explanations.md").write_text(notes, encoding="utf-8")


def export_source_data(wm: pd.DataFrame, trace: pd.DataFrame) -> None:
    wm_columns = [
        "scale", "architecture", "epoch", "parameter_count", "one_step_NRMSE",
        "NRMSE_H1", "NRMSE_H5", "NRMSE_H10", "NRMSE_H20", "NRMSE_H50",
        "mean_NRMSE_H5_H10_H20", "selected_best_world_model",
    ]
    wm[wm_columns].to_csv(SOURCE_DATA / "fig2_fig3_world_model_metrics.csv", index=False)
    trace.to_csv(SOURCE_DATA / "fig4_representative_rollout.csv", index=False)
    main = pd.read_csv(ROOT / "outputs/metrics/main_system_matrix_development_episodes.csv")
    main["strategy"] = main["strategy"].map(_method_name)
    main[
        [
            "dataset_scale",
            "world_model_architecture",
            "strategy",
            "seed",
            "episode_return",
            "episode_length",
        ]
    ].to_csv(SOURCE_DATA / "fig5_strategy_episodes.csv", index=False)
    cross = pd.read_csv(ROOT / "outputs/metrics/world_model_control_cross_m1000_episodes.csv")
    cross["strategy"] = cross["strategy"].map(_method_name)
    cross[
        [
            "dataset_scale",
            "world_model_architecture",
            "strategy",
            "seed",
            "episode_return",
            "episode_length",
        ]
    ].to_csv(SOURCE_DATA / "fig6_world_model_strategy_episodes.csv", index=False)
    final = pd.read_csv(ROOT / "outputs/metrics/final_selected_systems_fresh_seeds_episodes.csv")
    behavior = np.asarray(
        json.loads((ROOT / "outputs/metrics/strategy_ib_m1000.json").read_text(encoding="utf-8"))["original_behavior_reference"]["returns"]
    )
    behavior_rows = pd.DataFrame(
        {
            "evaluation_source": "NeoRL dataset trajectories",
            "system": "Original Behavior",
            "sample_id": np.arange(len(behavior)),
            "episode_return": behavior,
        }
    )
    online_rows = final[["strategy", "dataset_scale", "seed", "episode_return"]].copy()
    online_rows["evaluation_source"] = "NeoRL simulator"
    online_rows["system"] = np.where(
        online_rows["strategy"] == "BC",
        "BC",
        online_rows["dataset_scale"].fillna("") + " selected system",
    )
    online_rows = online_rows.rename(columns={"seed": "sample_id"})
    pd.concat(
        [
            behavior_rows,
            online_rows[["evaluation_source", "system", "sample_id", "episode_return"]],
        ],
        ignore_index=True,
    ).to_csv(SOURCE_DATA / "fig7_final_evaluation.csv", index=False)
    kl_payload = json.loads((ROOT / "outputs/metrics/strategy_ib_m1000_gru_mbppo.json").read_text(encoding="utf-8"))
    kl_returns = []
    for condition, key in (("without behavior KL", "mbppo_without_kl"), ("with behavior KL", "mbppo_with_kl")):
        for seed, value in zip([42, 43, 44, 45, 46], kl_payload["methods"][key]["returns"]):
            kl_returns.append({"condition": condition, "seed": seed, "episode_return": value})
    pd.DataFrame(kl_returns).to_csv(SOURCE_DATA / "fig8_kl_ablation_returns.csv", index=False)
    histories = []
    for condition, filename in (
        ("without behavior KL", "mbppo_ib_m1000_gru_without_kl_training.csv"),
        ("with behavior KL", "mbppo_ib_m1000_gru_with_kl_training.csv"),
    ):
        history = pd.read_csv(ROOT / "outputs/metrics" / filename)
        history.insert(0, "condition", condition)
        histories.append(history)
    pd.concat(histories, ignore_index=True).to_csv(SOURCE_DATA / "fig8_kl_ablation_history.csv", index=False)

    final_summary = pd.read_csv(ROOT / "outputs/metrics/final_selected_systems_fresh_seeds_summary.csv")
    snapshot = {
        "world_model_rows": int(len(wm)),
        "development_strategy_episode_rows": int(len(main)),
        "controlled_cross_episode_rows": int(len(cross)),
        "final_online_episode_rows": int(len(final)),
        "original_behavior_trajectory_rows": int(len(behavior)),
        "final_selected_systems": final_summary.to_dict(orient="records"),
    }
    (GENERATED / "report_metrics_snapshot.json").write_text(
        json.dumps(snapshot, indent=2), encoding="utf-8"
    )


def main() -> None:
    configure_style()
    wm = read_world_models()
    figure_1_framework()
    figure_2_heatmaps(wm)
    figure_3_multistep(wm)
    trace = representative_rollout()
    figure_4_rollout(trace)
    figure_5_strategy()
    figure_6_cross()
    figure_7_final()
    figure_8_kl()
    generate_tables(wm)
    figure_notes(wm, trace)
    export_source_data(wm, trace)
    normalize_generated_text()
    print(f"generated figures={len(list(FIGURES.glob('*.pdf')))} tables=6")


if __name__ == "__main__":
    main()
