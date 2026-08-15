from __future__ import annotations

import csv
import json
import math
import sys
import time
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
            "font.sans-serif": ["Microsoft YaHei", "Arial", "DejaVu Sans"],
            "mathtext.fontset": "dejavusans",
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
        target = FIGURES / f"{stem}.{suffix}"
        temporary = FIGURES / f"{stem}.tmp.{suffix}"
        for attempt in range(3):
            try:
                fig.savefig(temporary, bbox_inches="tight", **kwargs)
                temporary.replace(target)
                break
            except OSError:
                if attempt == 2:
                    raise
                time.sleep(0.15)
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
    fig, ax = plt.subplots(figsize=(7.2, 3.45))
    ax.set_xlim(0, 16)
    ax.set_ylim(0, 7.2)
    ax.axis("off")
    phases = [
        (0.2, 5.0, 2.55, 1.15, "NeoRL 历史轨迹", "工业状态与动作记录", "#EAF1F8"),
        (3.25, 5.0, 2.55, 1.15, "历史状态与动作", "30帧历史 + 3维动作", "#EAF1F8"),
        (6.3, 5.0, 2.45, 1.15, "世界模型训练", "预测下一帧状态", "#E4F3F0"),
        (9.25, 5.0, 3.0, 1.15, "预测能力评价", "单步 + 多步递归预测", "#E4F3F0"),
        (12.75, 5.0, 2.75, 1.15, "冻结世界模型", "支持后续策略优化", "#E4F3F0"),
    ]
    for x, y, w, h, title, sub, color in phases:
        box = FancyBboxPatch(
            (x, y), w, h, boxstyle="round,pad=0.08,rounding_size=0.1",
            facecolor=color, edgecolor="#4C5964", linewidth=1.0,
        )
        ax.add_patch(box)
        ax.text(x + w / 2, y + 0.72, title, ha="center", va="center", fontweight="bold", fontsize=8.5)
        ax.text(x + w / 2, y + 0.30, sub, ha="center", va="center", color="#4C5964", fontsize=6.8)
    for x1, x2 in ((2.75, 3.25), (5.8, 6.3), (8.75, 9.25), (12.25, 12.75)):
        ax.add_patch(FancyArrowPatch((x1, 5.58), (x2, 5.58), arrowstyle="-|>", mutation_scale=11, color="#4C5964"))

    plan = FancyBboxPatch(
        (2.2, 2.55), 4.15, 1.05, boxstyle="round,pad=0.06,rounding_size=0.08",
        facecolor="#EAF1F8", edgecolor="#3B6FB6", linewidth=1.0,
    )
    rl = FancyBboxPatch(
        (9.65, 2.55), 4.15, 1.05, boxstyle="round,pad=0.06,rounding_size=0.08",
        facecolor="#EEE8F5", edgecolor="#8C6BB1", linewidth=1.0,
    )
    ax.add_patch(plan)
    ax.add_patch(rl)
    ax.text(4.275, 3.18, "基于模型的规划", ha="center", fontweight="bold")
    ax.text(4.275, 2.78, "CEM / iCEM / MPPI", ha="center", fontsize=7)
    ax.text(11.725, 3.18, "基于模型的强化学习", ha="center", fontweight="bold")
    ax.text(11.725, 2.78, "MB-PPO", ha="center", fontsize=7)
    ax.add_patch(FancyArrowPatch((14.125, 5.0), (4.3, 3.6), arrowstyle="-|>", mutation_scale=10, color="#3B6FB6", connectionstyle="arc3,rad=0.10"))
    ax.add_patch(FancyArrowPatch((14.125, 5.0), (11.7, 3.6), arrowstyle="-|>", mutation_scale=10, color="#8C6BB1", connectionstyle="arc3,rad=-0.08"))

    simulator = FancyBboxPatch(
        (5.75, 0.55), 4.5, 1.05, boxstyle="round,pad=0.06,rounding_size=0.08",
        facecolor="#F8EFD9", edgecolor="#B58116", linewidth=1.0,
    )
    ax.add_patch(simulator)
    ax.text(8.0, 1.18, "NeoRL 仿真环境", ha="center", fontweight="bold")
    ax.text(8.0, 0.78, "1000步长期累计奖励", ha="center", fontsize=7)
    ax.add_patch(FancyArrowPatch((4.3, 2.55), (6.6, 1.6), arrowstyle="-|>", mutation_scale=10, color="#3B6FB6"))
    ax.add_patch(FancyArrowPatch((11.7, 2.55), (9.4, 1.6), arrowstyle="-|>", mutation_scale=10, color="#8C6BB1"))
    save_figure(fig, "fig1_framework")


def figure_2_heatmaps(wm: pd.DataFrame) -> None:
    one = wm.pivot(index="architecture", columns="scale", values="one_step_NRMSE").loc[ARCHITECTURES, SCALES]
    rollout = wm.pivot(index="architecture", columns="scale", values="mean_NRMSE_H5_H10_H20").loc[ARCHITECTURES, SCALES]
    selected = {("GRU", "M100"), ("Transformer-2L", "M1000"), ("Transformer-2L", "M10000")}
    fig, axes = plt.subplots(1, 2, figsize=(7.1, 3.55), constrained_layout=True)
    for label, ax, data, title, fmt in (
        ("a", axes[0], one, "单步预测 NRMSE", ".3f"),
        ("b", axes[1], rollout, "H5/H10/H20 平均 NRMSE", ".3f"),
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
    axes[0].set_ylabel("模型架构")
    save_figure(fig, "fig2_world_model_heatmaps")


def figure_3_multistep(wm: pd.DataFrame) -> None:
    subset = wm[wm["scale"] == "M1000"].set_index("architecture").loc[ARCHITECTURES]
    horizons = [1, 5, 10, 20, 50]
    fig, ax = plt.subplots(figsize=(6.6, 3.7))
    for architecture in ARCHITECTURES:
        values = [subset.loc[architecture, f"NRMSE_H{h}"] for h in horizons]
        ax.plot(
            horizons, values, marker="o", markersize=4.8, linewidth=1.5,
            color=ARCH_COLORS[architecture], label=architecture,
        )
    ax.set_xticks(horizons, [str(h) for h in horizons])
    ax.set_xlabel("递归预测步数")
    ax.set_ylabel("NRMSE")
    ax.set_title("M1000：递归预测误差随预测步长的变化", fontweight="bold", loc="left")
    ax.grid(axis="y", alpha=0.22)
    ax.legend(ncol=3, loc="upper left")
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
    variables = [("velocity", "速度"), ("fatigue", "疲劳度"), ("consumption", "能耗")]
    fig, axes = plt.subplots(3, 1, figsize=(7.0, 5.6), sharex=True, constrained_layout=True)
    for label, ax, (key, display) in zip("abc", axes, variables):
        data = trace[trace["variable"] == key]
        ax.plot(data["step"], data["ground_truth"], color="#252A2E", linewidth=1.5, label="真实状态")
        ax.plot(data["step"], data["prediction"], color="#2A9D8F", linewidth=1.5, linestyle="--", label="模型预测")
        ax.set_ylabel(display)
        ax.grid(alpha=0.2)
        panel_label(ax, label)
    axes[0].legend(loc="upper left", ncol=2)
    axes[-1].set_xlabel("递归预测步数")
    fig.suptitle("代表性50步递归预测轨迹", fontsize=10, fontweight="bold")
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
    fig, axes = plt.subplots(1, 3, figsize=(7.2, 3.65), sharey=True)
    for panel_index, (label, scale, ax) in enumerate(zip("abc", ["M-100", "M-1000", "M-10000"], axes)):
        subset = episodes[episodes["dataset_scale"] == scale]
        for index, strategy in enumerate(STRATEGIES):
            values = subset[subset["strategy"] == strategy]["episode_return"].values
            if scale == "M-100" and strategy == "iCEM":
                ax.annotate(
                    "iCEM: -2.65M\n（超出显示范围）",
                    xy=(index, -378000), xytext=(index, -366000),
                    ha="center", va="bottom", fontsize=6.2, color=STRATEGY_COLORS[strategy],
                    arrowprops={"arrowstyle": "-|>", "color": STRATEGY_COLORS[strategy], "lw": 1.0},
                )
                continue
            _point_summary(ax, values, index, STRATEGY_COLORS[strategy], 100 + panel_index * 10 + index)
        ax.set_xticks(range(4), STRATEGIES, rotation=18, ha="right")
        ax.set_title(scale.replace("-", ""), fontweight="bold")
        ax.grid(axis="y", alpha=0.22)
        panel_label(ax, label)
        ax.set_ylim(-380000, -200000)
    axes[0].set_ylabel("1000步累计奖励（越高越好）")
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
            ARCHITECTURES,
            rotation=28,
            ha="right",
        )
        panel.tick_params(axis="x", labelsize=6.5)
        panel.set_title(f"固定 {strategy}", fontweight="bold")
        panel.set_ylabel("1000步累计奖励")
        panel.grid(axis="y", alpha=0.22)
        panel_label(panel, label)
    axes[0].set_ylim(-355000, -200000)
    axes[1].set_ylim(-291000, -274000)
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
        [behavior], positions=[0], widths=0.48, patch_artist=True, showfliers=False,
        medianprops={"color": "#333333", "linewidth": 1.2},
        boxprops={"facecolor": "#F3F4F5", "edgecolor": "#6C757D", "linewidth": 1.2},
        whiskerprops={"color": "#6C757D"}, capprops={"color": "#6C757D"},
    )
    del box
    for index, (name, values, color) in enumerate(groups, start=2):
        _point_summary(ax, values, index, color, 400 + index)
    ax.set_xticks(
        [0, 2, 3, 4, 5],
        ["原始行为\n（数据集历史轨迹）"] + [group[0] for group in groups],
        rotation=0,
        ha="center",
    )
    ax.set_xlim(-0.65, 5.55)
    ax.set_ylabel("1000步累计奖励（越高越好）")
    ax.set_ylim(-297000, -210000)
    ax.grid(axis="y", alpha=0.22)
    ax.axvline(1.15, color="#AAB0B5", linewidth=1.0, linestyle="--")
    ax.text(0.28, 1.03, "数据集历史表现", transform=ax.transAxes, ha="center", va="bottom", fontsize=7.3, fontweight="bold", color="#5F6B73")
    ax.text(0.72, 1.03, "NeoRL 仿真验证", transform=ax.transAxes, ha="center", va="bottom", fontsize=7.3, fontweight="bold", color="#2A6F69")
    ax.set_title("最终长期控制效果", fontweight="bold", loc="left")
    save_figure(fig, "fig7_final_simulator")


def figure_8_kl() -> None:
    payload = json.loads((ROOT / "outputs/metrics/strategy_ib_m1000_gru_mbppo.json").read_text(encoding="utf-8"))
    with_kl = np.asarray(payload["methods"]["mbppo_with_kl"]["returns"], dtype=np.float64)
    without_kl = np.asarray(payload["methods"]["mbppo_without_kl"]["returns"], dtype=np.float64)
    hist_with = pd.read_csv(ROOT / "outputs/metrics/mbppo_ib_m1000_gru_with_kl_training.csv")
    hist_without = pd.read_csv(ROOT / "outputs/metrics/mbppo_ib_m1000_gru_without_kl_training.csv")
    fig, axes = plt.subplots(1, 2, figsize=(7.2, 3.6), constrained_layout=True)
    for index, (name, values, color) in enumerate(
        [("无 KL 约束", without_kl, "#C44E52"), ("有 KL 约束", with_kl, STRATEGY_COLORS["MB-PPO"])]
    ):
        _point_summary(axes[0], values, index, color, 500 + index)
    axes[0].set_xticks([0, 1], ["MB-PPO\n无 KL 约束", "MB-PPO\n有 KL 约束"])
    axes[0].set_ylabel("1000步累计奖励")
    axes[0].set_title("仿真累计奖励", fontweight="bold")
    axes[0].grid(axis="y", alpha=0.22)
    panel_label(axes[0], "a")
    axes[1].plot(hist_without["gradient_step"], hist_without["behavior_kl"], color="#C44E52", linewidth=1.4, label="无 KL 约束")
    axes[1].plot(hist_with["gradient_step"], hist_with["behavior_kl"], color=STRATEGY_COLORS["MB-PPO"], linewidth=1.4, label="有 KL 约束")
    axes[1].set_yscale("symlog", linthresh=0.1)
    axes[1].set_xlabel("PPO 更新步数")
    axes[1].set_ylabel(r"策略偏离程度 $D_{KL}(\pi_b\Vert\pi)$\n（对数型尺度）")
    axes[1].set_title("策略与行为策略的偏离", fontweight="bold")
    axes[1].grid(alpha=0.22)
    axes[1].legend()
    panel_label(axes[1], "b")
    save_figure(fig, "fig8_mbppo_kl_ablation")


def horizon_ablation_data() -> pd.DataFrame:
    payload = json.loads(
        (ROOT / "outputs/metrics/strategy_ib_m1000_gru_cem_horizons.json").read_text(
            encoding="utf-8"
        )
    )
    selected_horizon = int(payload["selection"]["selected_horizon"])
    selected_mean = float(
        payload["horizons"][str(selected_horizon)]["summary"]["mean"]
    )
    rows = []
    for horizon_text, record in payload["horizons"].items():
        summary = record["summary"]
        horizon = int(horizon_text)
        rows.append(
            {
                "horizon": horizon,
                "episodes": int(summary["episodes"]),
                "mean_return": float(summary["mean"]),
                "std_return": float(summary["std"]),
                "median_return": float(summary["median"]),
                "delta_vs_h10": float(summary["mean"]) - selected_mean,
                "selected": horizon == selected_horizon,
            }
        )
    return pd.DataFrame(rows).sort_values("horizon").reset_index(drop=True)


def horizon_sensitivity_data(wm: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for scale in SCALES:
        subset = wm[wm["scale"] == scale]
        short_medium = subset.loc[
            subset["mean_NRMSE_H5_H10_H20"].idxmin(), "architecture"
        ]
        long_horizon = subset.loc[subset["NRMSE_H50"].idxmin(), "architecture"]
        rows.append(
            {
                "scale": scale,
                "best_h5_h10_h20_architecture": short_medium,
                "best_h50_architecture": long_horizon,
                "consistent": short_medium == long_horizon,
            }
        )
    return pd.DataFrame(rows)


def training_sufficiency_data() -> pd.DataFrame:
    specifications = (
        (
            "M100",
            "GRU",
            ROOT / "outputs/metrics/world_model_ib_m100_gru_training.csv",
            35,
        ),
        (
            "M1000",
            "Transformer-2L",
            ROOT
            / "outputs/metrics/world_model_ib_m1000_transformer2_fair_training.csv",
            20,
        ),
        (
            "M10000",
            "Transformer-2L",
            ROOT
            / "outputs/metrics/world_model_ib_m10000_transformer2_fair_training.csv",
            4,
        ),
    )
    rows = []
    for scale, architecture, path, selected_epoch in specifications:
        history = pd.read_csv(path)
        selected = history[history["epoch"] == selected_epoch].iloc[0]
        last = history.iloc[-1]
        rows.append(
            {
                "scale": scale,
                "architecture": architecture,
                "total_epochs": int(last["epoch"]),
                "selected_epoch": selected_epoch,
                "selected_train_mse": float(selected["train_mse"]),
                "selected_val_mse": float(selected["val_mse"]),
                "last_train_mse": float(last["train_mse"]),
                "last_val_mse": float(last["val_mse"]),
            }
        )
    return pd.DataFrame(rows)


def figure_9_training_sufficiency() -> None:
    history = pd.read_csv(
        ROOT
        / "outputs/metrics/world_model_ib_m1000_transformer2_fair_training.csv"
    )
    checkpoints = pd.read_csv(
        ROOT / "outputs/metrics/world_model_5x3_common_validation_checkpoints.csv"
    )
    checkpoints = checkpoints[
        (checkpoints["data_scale"] == 1000)
        & (checkpoints["architecture"] == "Transformer-2L")
    ].sort_values("epoch")
    selected_epoch = 20
    selected = history[history["epoch"] == selected_epoch].iloc[0]
    selected_checkpoint = checkpoints[checkpoints["epoch"] == selected_epoch].iloc[0]
    fig, axes = plt.subplots(1, 2, figsize=(7.2, 3.35), constrained_layout=True)

    axes[0].plot(
        history["epoch"],
        history["train_mse"],
        color="#3B6FB6",
        linewidth=1.6,
        label="训练 MSE",
    )
    axes[0].plot(
        history["epoch"],
        history["val_mse"],
        color="#E69F00",
        linewidth=1.6,
        label="验证 MSE",
    )
    axes[0].axvline(
        selected_epoch,
        color="#4C5964",
        linestyle="--",
        linewidth=1.0,
        label="架构内选定轮次",
    )
    axes[0].scatter(
        [selected_epoch, selected_epoch],
        [selected["train_mse"], selected["val_mse"]],
        color=["#3B6FB6", "#E69F00"],
        edgecolor="white",
        linewidth=0.6,
        s=30,
        zorder=4,
    )
    axes[0].set_xlabel("训练轮次")
    axes[0].set_ylabel("均方误差（MSE）")
    axes[0].set_title("正式50轮训练历史", fontweight="bold", loc="left")
    axes[0].grid(axis="y", alpha=0.22)
    axes[0].legend(ncol=1, loc="upper right")
    panel_label(axes[0], "a")

    axes[1].plot(
        checkpoints["epoch"],
        checkpoints["one_step_NRMSE"],
        color="#008E9B",
        linewidth=1.4,
        marker="o",
        markersize=4.0,
        label="单步 NRMSE",
    )
    axes[1].plot(
        checkpoints["epoch"],
        checkpoints["mean_NRMSE_H5_H10_H20"],
        color="#9C6ADE",
        linewidth=1.4,
        marker="s",
        markersize=4.0,
        label="mean(H5, H10, H20)",
    )
    axes[1].axvline(
        selected_epoch,
        color="#4C5964",
        linestyle="--",
        linewidth=1.0,
        label="架构内选定轮次",
    )
    axes[1].scatter(
        [selected_epoch, selected_epoch],
        [
            selected_checkpoint["one_step_NRMSE"],
            selected_checkpoint["mean_NRMSE_H5_H10_H20"],
        ],
        color=["#008E9B", "#9C6ADE"],
        edgecolor="white",
        linewidth=0.6,
        s=34,
        zorder=4,
    )
    axes[1].set_xticks([5, 10, 20, 30, 40, 50])
    axes[1].set_xlabel("候选 checkpoint 轮次")
    axes[1].set_ylabel("NRMSE")
    axes[1].set_title("统一验证指标", fontweight="bold", loc="left")
    axes[1].grid(axis="y", alpha=0.22)
    axes[1].legend(ncol=1, loc="upper right")
    panel_label(axes[1], "b")
    save_figure(fig, "fig9_training_sufficiency")


def latex_num(value: float, decimals: int = 3) -> str:
    return f"{value:.{decimals}f}"


def latex_return(mean: float, std: float) -> str:
    return f"{mean:,.0f} $\\pm$ {std:,.0f}"


def generate_tables(wm: pd.DataFrame) -> None:
    architecture_rows = []
    config = {
        "MLP": ("无显式时序递归", "2$\\times$256，dropout 0.1"),
        "GRU": ("门控递归状态", "2层，隐状态128，动作分支64"),
        "LSTM": ("带记忆单元的递归状态", "2层，隐状态128，动作分支64"),
        "Transformer-2L": ("自注意力历史编码", "$d=64$，4头，FFN 128，2层"),
        "Transformer-4L": ("自注意力历史编码", "$d=64$，4头，FFN 128，4层"),
    }
    for architecture in ARCHITECTURES:
        row = wm[wm["architecture"] == architecture].iloc[0]
        architecture_rows.append(
            f"{architecture} & {config[architecture][0]} & {config[architecture][1]} & {int(row['parameter_count']):,} \\\\"
        )
    table1 = r"""
\begin{table}[H]
\centering
\caption{世界模型架构。参数量不随数据规模改变。}
\label{tab:architectures}
\small
\begin{tabularx}{\textwidth}{lXXr}
\toprule
模型架构 & 时间信息建模方式 & 主要配置 & 参数量 \\
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
\caption{统一验证选出的世界模型，同时列出 H50 以展示更长时域的递归预测表现。}
\label{tab:selected_wm}
\small
\begin{tabular}{lllrrrrr}
\toprule
数据规模 & 模型架构 & 选定轮次 & 单步 & H5 & H10 & H20 & H50 \\
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
\caption{Transformer 深度消融。单元格为单步 NRMSE / H5、H10、H20 平均 NRMSE；差值为4L减2L，负值表示4L更优。}
\label{tab:depth}
\small
\begin{tabular}{lccc}
\toprule
数据规模 & Transformer-2L & Transformer-4L & 差值 \\
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
\caption{不同数据规模下的策略累计奖励。每种组合包含5次独立仿真，数值为均值 $\pm$ 标准差，越高越好。}
\label{tab:strategy_matrix}
\scriptsize
\resizebox{\textwidth}{!}{%
\begin{tabular}{lrrrr}
\toprule
数据规模 & CEM & iCEM & MPPI & MB-PPO \\
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
        f"原始行为（数据集历史轨迹） & NeoRL 数据集历史轨迹 & {latex_return(behavior.mean(), behavior.std())} & -- \\\\ ",
        f"BC & NeoRL 仿真环境 & {latex_return(bc.mean(), bc.std())} & 0 \\\\ ",
    ]
    names = {
        "M-100": "M100: GRU + CEM",
        "M-1000": "M1000: Transformer-2L + MPPI",
        "M-10000": "M10000: Transformer-2L + MPPI",
    }
    for _, row in final_summary.iterrows():
        final_rows.append(
            f"{names[row['dataset_scale']]} & NeoRL 仿真环境 & {latex_return(row['mean_return'], row['std_return'])} & "
            f"{row['mean_delta_vs_bc']:+,.0f} \\\\"
        )
    table5 = r"""
\begin{table}[H]
\centering
\caption{最终 NeoRL 控制结果。仿真方法在10组未参与模型和策略选择的独立随机初始条件下评价；原始行为来自数据集已有轨迹。}
\label{tab:final}
\scriptsize
\resizebox{\textwidth}{!}{%
\begin{tabular}{llrr}
\toprule
方法或系统 & 评价来源 & 1000步累计奖励 & 相对 BC 变化 \\
\midrule
""" + "\n".join(final_rows) + r"""
\bottomrule
\end{tabular}}
\end{table}
"""

    all_world_model_rows = []
    ordered_world_models = (
        wm.assign(
            scale_order=pd.Categorical(wm["scale"], categories=SCALES, ordered=True),
            architecture_order=pd.Categorical(
                wm["architecture"], categories=ARCHITECTURES, ordered=True
            ),
        )
        .sort_values(["scale_order", "architecture_order"])
    )
    for _, row in ordered_world_models.iterrows():
        all_world_model_rows.append(
            f"{row['scale']} & {row['architecture']} & {int(row['epoch'])} & "
            f"{row['one_step_NRMSE']:.3f} & {row['NRMSE_H5']:.3f} & "
            f"{row['NRMSE_H10']:.3f} & {row['NRMSE_H20']:.3f} & "
            f"{row['NRMSE_H50']:.3f} & {row['mean_NRMSE_H5_H10_H20']:.3f} \\\\"
        )
    all_world_models = r"""
\begingroup
\scriptsize
\setlength{\tabcolsep}{3.2pt}
\begin{longtable}{llrrrrrrr}
\caption{五种架构在三个数据规模上的完整预测结果。所有数值均来自统一验证，平均值为 H5、H10 和 H20 的算术平均。架构内选定轮次表示该模型架构所有候选 checkpoint 中最终采用的参数位置。}
\label{tab:all_world_models}\\
\toprule
数据规模 & 模型架构 & 架构内选定轮次 & 单步 & H5 & H10 & H20 & H50 & H5--H20平均 \\
\midrule
\endfirsthead
\multicolumn{9}{c}{\tablename\ \thetable\ （续）}\\
\toprule
数据规模 & 模型架构 & 架构内选定轮次 & 单步 & H5 & H10 & H20 & H50 & H5--H20平均 \\
\midrule
\endhead
""" + "\n".join(all_world_model_rows) + r"""
\bottomrule
\end{longtable}
\endgroup
"""

    t2 = wm[(wm["scale"] == "M1000") & (wm["architecture"] == "Transformer-2L")].iloc[0]
    variable_names = ["setpoint", "velocity", "gain", "shift", "fatigue", "consumption"]
    variable_labels = {
        "setpoint": "设定值",
        "velocity": "速度",
        "gain": "增益",
        "shift": "偏移",
        "fatigue": "疲劳度",
        "consumption": "能耗",
    }
    variable_rows = []
    for variable in variable_names:
        variable_rows.append(
            f"{variable_labels[variable]} & {t2[f'one_step_NRMSE_{variable}']:.3f} & "
            f"{t2[f'NRMSE_H20_{variable}']:.3f} & {t2[f'NRMSE_H50_{variable}']:.3f} \\\\"
        )
    per_variable = r"""
\begin{table}[H]
\centering
\caption{M1000 Transformer-2L 的变量级 NRMSE。不同物理变量的误差累积速度明显不同。}
\label{tab:per_variable}
\small
\begin{tabular}{lrrr}
\toprule
物理变量 & 单步 & H20 & H50 \\
\midrule
""" + "\n".join(variable_rows) + r"""
\bottomrule
\end{tabular}
\end{table}
"""

    scale_audit = pd.read_csv(SOURCE_DATA / "dataset_scale_audit.csv").set_index("scale").loc[SCALES]
    scale_rows = []
    for scale, row in scale_audit.iterrows():
        scale_rows.append(
            f"{scale} & {int(row['training_trajectories']):,} & "
            f"{int(row['transitions_per_trajectory']):,} & "
            f"{int(row['total_usable_transitions']):,} & "
            f"{int(row['formal_training_epochs'])} \\\\"
        )
    data_scales = r"""
\begin{table}[H]
\centering
\caption{数据规模与统一架构比较的训练设置。每条轨迹均包含1000个连续控制时间步。}
\label{tab:data_scales}
\small
\begin{tabular}{lrrrr}
\toprule
数据规模 & 训练轨迹数 & 每轨迹转移数 & 可用状态转移数 & 正式训练轮数 \\
\midrule
""" + "\n".join(scale_rows) + r"""
\bottomrule
\end{tabular}
\end{table}
"""

    horizon = horizon_ablation_data()
    horizon_rows = []
    for _, row in horizon.iterrows():
        relative = (
            "最终采用"
            if bool(row["selected"])
            else f"低于 H=10 {abs(row['delta_vs_h10']):,.0f}"
        )
        horizon_rows.append(
            f"H={int(row['horizon'])} & "
            f"{latex_return(row['mean_return'], row['std_return'])} & "
            f"{row['median_return']:,.0f} & {relative} \\\\"
        )
    horizon_table = r"""
\begin{table}[H]
\centering
\caption{CEM 规划长度消融。固定 M1000 GRU 世界模型及其余规划参数，每个规划长度在5个独立仿真初始条件下运行1000步；累计奖励越高越好。}
\label{tab:horizon_ablation}
\small
\begin{tabular}{lrrl}
\toprule
规划长度 & 1000步累计奖励（均值 $\pm$ 标准差） & 中位数 & 相对表现 \\
\midrule
""" + "\n".join(horizon_rows) + r"""
\bottomrule
\end{tabular}
\end{table}
"""

    sensitivity = horizon_sensitivity_data(wm)
    sensitivity_rows = []
    for _, row in sensitivity.iterrows():
        sensitivity_rows.append(
            f"{row['scale']} & {row['best_h5_h10_h20_architecture']} & "
            f"{row['best_h50_architecture']} & "
            f"{'是' if bool(row['consistent']) else '否'} \\\\"
        )
    sensitivity_table = r"""
\begin{table}[H]
\centering
\caption{预测时间尺度敏感性。每列均在同一数据规模的五种架构中直接比较 NRMSE，数值越低越优。}
\label{tab:horizon_sensitivity}
\small
\begin{tabular}{lccc}
\toprule
数据规模 & H5/H10/H20综合最优架构 & H50最优架构 & 是否一致 \\
\midrule
""" + "\n".join(sensitivity_rows) + r"""
\bottomrule
\end{tabular}
\end{table}
"""

    training = training_sufficiency_data()
    training_rows = []
    for _, row in training.iterrows():
        training_rows.append(
            f"{row['scale']} & {row['architecture']} & {int(row['total_epochs'])} & "
            f"{int(row['selected_epoch'])} & {row['selected_train_mse']:.4f} & "
            f"{row['selected_val_mse']:.4f} & {row['last_train_mse']:.4f} & "
            f"{row['last_val_mse']:.4f} \\\\"
        )
    training_table = r"""
\begin{table}[H]
\centering
\caption{三个最终世界模型的训练充分性与 checkpoint 位置。选定轮次由统一验证的单步与多步预测协议确定，训练/验证 MSE 用于观察优化过程。}
\label{tab:training_sufficiency}
\scriptsize
\resizebox{\textwidth}{!}{%
\begin{tabular}{llrrrrrr}
\toprule
数据规模 & 模型 & 总轮次 & 架构内选定轮次 & 选定训练MSE & 选定验证MSE & 最后训练MSE & 最后验证MSE \\
\midrule
""" + "\n".join(training_rows) + r"""
\bottomrule
\end{tabular}}
\end{table}
"""

    (GENERATED / "tables.tex").write_text(
        "\n".join(
            [
                table1,
                table2,
                table3,
                table4,
                table5,
                all_world_models,
                per_variable,
                data_scales,
                horizon_table,
                sensitivity_table,
                training_table,
            ]
        ),
        encoding="utf-8",
    )
    for name, content in (
        ("table1_architectures.tex", table1),
        ("table2_selected_world_models.tex", table2),
        ("table3_depth_ablation.tex", table3),
        ("table4_strategy_matrix.tex", table4),
        ("table5_final_evaluation.tex", table5),
        ("table_all_world_models.tex", all_world_models),
        ("table_per_variable.tex", per_variable),
        ("table_data_scales.tex", data_scales),
        ("table_horizon_ablation.tex", horizon_table),
        ("table_horizon_sensitivity.tex", sensitivity_table),
        ("table_training_sufficiency.tex", training_table),
    ):
        (GENERATED / name).write_text(content, encoding="utf-8")


def figure_notes(wm: pd.DataFrame, trace: pd.DataFrame) -> None:
    main = pd.read_csv(ROOT / "outputs/metrics/main_system_matrix_development_summary.csv")
    cross = pd.read_csv(ROOT / "outputs/metrics/world_model_control_cross_m1000_summary.csv")
    final = pd.read_csv(ROOT / "outputs/metrics/final_selected_systems_fresh_seeds_summary.csv")
    behavior = np.asarray(json.loads((ROOT / "outputs/metrics/strategy_ib_m1000.json").read_text(encoding="utf-8"))["original_behavior_reference"]["returns"])
    selected = wm[wm["selected_best_world_model"] == True].set_index("scale")  # noqa: E712
    notes = f"""# 图表解释草稿

## 图1：项目整体技术路线

Question: 历史工业轨迹如何经过世界模型、策略优化和仿真评价形成完整流程？

Observation: 历史状态与动作先用于训练世界模型；模型冻结后分别支持规划方法和MB-PPO，并在NeoRL仿真环境中评价累计奖励。

Interpretation: 该图帮助读者快速理解项目做了什么，以及预测模型如何服务于控制策略。

Limitation: 流程图只描述技术路线，不提供性能证据。

## 图2：模型架构与数据规模

Question: 模型架构和数据规模如何共同影响单步与多步预测？

Observation: M100由GRU取得最低综合递归误差；M1000和M10000由Transformer-2L胜出。M10000的Transformer-2L在H5/H10/H20上的平均NRMSE为{selected.loc['M10000','mean_NRMSE_H5_H10_H20']:.3f}。

Interpretation: 模型架构与数据规模存在交互，单步预测排序与多步递归预测排序也不完全相同。

Limitation: 每格来自一次正式训练结果，不能替代多次独立训练的鲁棒性研究。

## 图3：递归预测误差随预测步长的变化

Question: 单步NRMSE为何不足以描述需要递归使用的世界模型？

Observation: M1000中五个模型的单步误差接近，但H20和H50明显分离；Transformer-2L在H5--H20综合最低，Transformer-4L在H50更低。

Interpretation: 小的单步偏差会在递归更新中传播，短中期稳定性需要直接评价。

Limitation: 图中固定 M1000，其他数据规模可能呈现不同的误差传播形态。

## 图4：代表性50步轨迹

Question: 代表性连续轨迹上的预测偏差如何随时间演化？

Observation: 速度、疲劳度和能耗呈现不同的偏差累积方式，所示样本的预测误差接近候选验证轨迹的中位数。

Interpretation: 整体NRMSE背后可能包含不同的变量级误差，工业控制需要同时监控关键变量。

Limitation: 一条代表轨迹不能替代全部验证轨迹的统计结果。

## 图5：不同数据规模下的策略比较

Question: 每个数据规模固定世界模型后，哪种策略取得更高累计奖励？

Observation: M100的CEM最高（{main[(main.dataset_scale=='M-100') & (main.strategy=='CEM')].mean_return.iloc[0]:,.0f}）；M1000和M10000的MPPI最高。M100的iCEM约为{main[(main.dataset_scale=='M-100') & (main.strategy=='iCEM')].mean_return.iloc[0]:,.0f}，明显超出主图范围。

Interpretation: 规划器能力与世界模型质量共同影响真实控制效果，优化器可能放大低质量模型的误差。

Limitation: iCEM 的表现还可能受到模型误差、采样预算和代价尺度共同影响。

## 图6：世界模型与后续策略

Question: 预测误差排序是否完全决定后续控制排序？

Observation: M1000中MPPI在Transformer-4L上最高（{cross[(cross.architecture=='Transformer-4L') & (cross.strategy=='MPPI')].mean_return.iloc[0]:,.0f}），MB-PPO在LSTM上最高（{cross[(cross.architecture=='LSTM') & (cross.strategy=='MB-PPO+KL')].mean_return.iloc[0]:,.0f}），统一预测评价则选择Transformer-2L。

Interpretation: 不同策略可能对世界模型的误差结构具有不同敏感性。

Limitation: 当前结果覆盖五种模型、两种策略和一个数据规模，其他设置仍需独立研究。

## 图7：最终NeoRL控制结果

Question: 最终系统能否提高1000步累计奖励？

Observation: 原始行为数据轨迹均值为{behavior.mean():,.0f}；三个最终系统相对BC的改善分别为{final.iloc[0].mean_delta_vs_bc:,.0f}、{final.iloc[1].mean_delta_vs_bc:,.0f}和{final.iloc[2].mean_delta_vs_bc:,.0f}。

Interpretation: 基于世界模型的策略优化在独立仿真条件下保持了长期回报提升。

Limitation: 原始行为来自数据集已有轨迹，与仿真策略的评价来源不同；结果也只覆盖当前NeoRL工业控制基准。

## 图8：MB-PPO行为KL消融

Question: 行为KL约束是否限制MB-PPO偏离历史行为分布？

Observation: 不加KL的平均累计奖励约为-883,709，加KL后为-284,715；训练记录显示无KL策略与行为分布的偏离显著扩大。

Interpretation: 在当前消融设置中，行为KL约束显著降低了模型利用风险。

Limitation: 消融覆盖一个世界模型和一次策略训练，最合适的KL系数及跨场景稳定性仍需进一步评价。

## 图9：代表性正式训练过程与候选checkpoint评价

Question: 正式训练是否完成，以及为什么不机械使用最后一轮参数？

Observation: M1000 Transformer-2L 的训练 MSE 在50轮内持续下降；验证 MSE 在前期进入较低区间后未继续同步改善，并在后期波动和部分回升。实际候选 checkpoint 的统一验证指标显示，第20轮取得最低的 H5/H10/H20 平均 NRMSE。

Interpretation: 继续降低训练误差没有带来稳定的验证收益，保存并比较中间 checkpoint 有助于依据实际单步和递归预测表现确定参数。

Limitation: 右图仅连接11个实际保存并评价的候选 checkpoint，连线用于辅助观察，不代表插值后的连续评价结果。
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
    horizon_ablation_data().to_csv(
        SOURCE_DATA / "table_horizon_ablation.csv", index=False
    )
    horizon_sensitivity_data(wm).to_csv(
        SOURCE_DATA / "table_horizon_sensitivity.csv", index=False
    )
    training_sufficiency_data().to_csv(
        SOURCE_DATA / "table_training_sufficiency.csv", index=False
    )
    pd.read_csv(
        ROOT
        / "outputs/metrics/world_model_ib_m1000_transformer2_fair_training.csv"
    ).to_csv(SOURCE_DATA / "fig9_training_history.csv", index=False)
    checkpoint_metrics = pd.read_csv(
        ROOT / "outputs/metrics/world_model_5x3_common_validation_checkpoints.csv"
    )
    checkpoint_metrics = checkpoint_metrics[
        (checkpoint_metrics["data_scale"] == 1000)
        & (checkpoint_metrics["architecture"] == "Transformer-2L")
    ].sort_values("epoch")
    checkpoint_metrics[
        [
            "epoch",
            "one_step_NRMSE",
            "NRMSE_H5",
            "NRMSE_H10",
            "NRMSE_H20",
            "mean_NRMSE_H5_H10_H20",
            "NRMSE_H50",
            "selected_within_architecture",
            "selected_status",
        ]
    ].to_csv(SOURCE_DATA / "fig9_checkpoint_metrics.csv", index=False)

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
    figure_9_training_sufficiency()
    generate_tables(wm)
    figure_notes(wm, trace)
    export_source_data(wm, trace)
    normalize_generated_text()
    print(f"generated figures={len(list(FIGURES.glob('*.pdf')))} tables=11")


if __name__ == "__main__":
    main()
