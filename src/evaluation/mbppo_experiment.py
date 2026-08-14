from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any, Dict

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.ticker import FuncFormatter

from src.evaluation.simulator import _summarize, evaluate_online_policy
from src.strategy.mbppo import MBPPOPolicy, train_mbppo_variant
from src.utils.config import load_config, resolve_path


def _read_json(path: Path) -> Dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def _read_training_history(path: Path) -> list[Dict[str, float]]:
    with path.open("r", encoding="utf-8") as handle:
        return [
            {key: float(value) for key, value in row.items()}
            for row in csv.DictReader(handle)
        ]


def _reference_returns(config: Dict[str, Any], root: Path) -> Dict[str, list[float]]:
    evaluation = config["evaluation"]
    seeds = [int(seed) for seed in evaluation["seeds"]]
    cem_payload = _read_json(resolve_path(root, evaluation["reuse_cem_metrics"]))
    cem_returns = [float(cem_payload["methods"]["cem"]["episodes"][str(seed)]["return"])
                   for seed in seeds]
    bc_payload = _read_json(resolve_path(root, evaluation["reuse_bc_metrics"]))
    source_seeds = [int(seed) for seed in bc_payload["metadata"]["seeds"]]
    bc_by_seed = dict(zip(source_seeds, bc_payload["official_bc"]["returns"]))
    return {
        "official_bc": [float(bc_by_seed[seed]) for seed in seeds],
        "cem": cem_returns,
    }


def _write_episode_csv(path: Path, seeds: list[int], methods: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=["method", "seed", "return"])
        writer.writeheader()
        for method, section in methods.items():
            for seed, episode_return in zip(seeds, section["returns"]):
                writer.writerow({"method": method, "seed": seed, "return": episode_return})


def _plot_results(
    results: Dict[str, Any], histories: Dict[str, list[Dict[str, float]]], path: Path
) -> None:
    mpl.rcParams.update(
        {
            "font.family": "sans-serif",
            "font.sans-serif": ["Arial", "DejaVu Sans"],
            "font.size": 9,
            "svg.fonttype": "none",
            "axes.spines.top": False,
            "axes.spines.right": False,
        }
    )
    colors = {
        "official_bc": "#7A7A7A",
        "cem": "#0072B2",
        "mbppo_with_kl": "#009E73",
        "mbppo_without_kl": "#D55E00",
    }
    labels = {
        "official_bc": "Official BC",
        "cem": "CEM H=10",
        "mbppo_with_kl": "MB-PPO + KL",
        "mbppo_without_kl": "MB-PPO no KL",
    }
    order = list(labels)
    figure, axes = plt.subplots(1, 3, figsize=(12.2, 3.8))
    figure.suptitle(
        "Behavior KL tests whether MB-PPO limits exploitation of the frozen GRU world model",
        fontsize=12,
        fontweight="bold",
        y=1.02,
    )

    for position, method in enumerate(order):
        values = np.asarray(results["methods"][method]["returns"], dtype=np.float64)
        mean = values.mean()
        sem = values.std(ddof=1) / np.sqrt(len(values))
        if method == "mbppo_without_kl":
            axes[0].scatter(position, -308500, marker="v", color=colors[method], s=42,
                            clip_on=False, zorder=4)
            axes[0].annotate(
                f"off-scale mean\n{mean / 1000:.1f}k",
                xy=(position, -308500),
                xytext=(position - 0.15, -305500),
                ha="right",
                va="bottom",
                fontsize=8,
                color=colors[method],
                arrowprops={"arrowstyle": "->", "color": colors[method], "lw": 0.8},
            )
        else:
            axes[0].scatter(
                np.full(len(values), position) + np.linspace(-0.08, 0.08, len(values)),
                values,
                color=colors[method],
                s=24,
                alpha=0.85,
                zorder=3,
            )
            axes[0].errorbar(position, mean, yerr=sem, fmt="_", color="black", capsize=4,
                             markersize=14, linewidth=1.2, zorder=4)
    axes[0].set_xticks(range(len(order)), [labels[key] for key in order], rotation=18, ha="right")
    axes[0].set_ylabel("Simulator episode return (higher is better)")
    axes[0].set_title("a  Development seeds (n=5)", loc="left", fontweight="bold")
    axes[0].set_ylim(-310000, -260000)
    axes[0].yaxis.set_major_formatter(FuncFormatter(lambda value, _pos: f"{value / 1000:.0f}k"))
    axes[0].grid(axis="y", alpha=0.22)

    for variant, method in (("with_kl", "mbppo_with_kl"), ("without_kl", "mbppo_without_kl")):
        rows = histories[variant]
        axes[1].plot(
            [row["gradient_step"] for row in rows],
            [row["behavior_kl"] for row in rows],
            color=colors[method],
            label=labels[method],
            linewidth=1.6,
        )
        axes[2].plot(
            [row["gradient_step"] for row in rows],
            [row["model_reward_mean"] for row in rows],
            color=colors[method],
            label=labels[method],
            linewidth=1.6,
        )
    axes[1].set(
        xlabel="PPO gradient step",
        ylabel=r"$D_{KL}(\hat{\pi}_b\Vert\pi)$ (symlog)",
        title="b  Distance from behavior (symlog)",
    )
    axes[1].set_yscale("symlog", linthresh=0.1, linscale=0.7)
    axes[2].set(
        xlabel="PPO gradient step",
        ylabel="Mean predicted one-step reward",
        title="c  Frozen-model training signal",
    )
    for axis in axes[1:]:
        axis.title.set_fontweight("bold")
        axis.title.set_ha("left")
        axis.title.set_position((0, 1.0))
        axis.grid(alpha=0.22)
    axes[1].legend(frameon=False, fontsize=8)
    figure.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    for suffix in (".png", ".pdf", ".svg"):
        output = path.with_suffix(suffix)
        figure.savefig(output, dpi=300 if suffix == ".png" else None, bbox_inches="tight")
    plt.close(figure)


def run_mbppo_experiment(config_path: str | Path) -> Dict[str, Any]:
    config, root = load_config(config_path)
    metrics_path = resolve_path(root, config["outputs"]["metrics_json"])
    if metrics_path.exists():
        print(f"reuse completed MB-PPO experiment: {metrics_path}")
        return _read_json(metrics_path)

    checkpoints = {
        "with_kl": train_mbppo_variant(config, root, use_behavior_kl=True),
        "without_kl": train_mbppo_variant(config, root, use_behavior_kl=False),
    }
    evaluation = config["evaluation"]
    seeds = [int(seed) for seed in evaluation["seeds"]]
    episode_horizon = int(evaluation["episode_horizon"])
    device = str(config["policies"]["device"])
    references = _reference_returns(config, root)
    methods: Dict[str, Any] = {
        key: {"returns": values, "summary": _summarize(values)}
        for key, values in references.items()
    }
    for variant, method_name in (
        ("with_kl", "mbppo_with_kl"),
        ("without_kl", "mbppo_without_kl"),
    ):
        policy = MBPPOPolicy.from_checkpoint(checkpoints[variant], device=device)
        section = evaluate_online_policy(
            method_name,
            lambda _seed, policy=policy: policy,
            seeds,
            episode_horizon,
            root,
        )
        section.pop("reward_traces", None)
        methods[method_name] = section

    results: Dict[str, Any] = {
        "metadata": {
            "role": evaluation["role"],
            "development_seeds": seeds,
            "held_out_seeds_reserved": evaluation["held_out_seeds"],
            "episode_horizon": episode_horizon,
            "world_model_checkpoint": str(
                resolve_path(root, config["policies"]["world_model_checkpoint"])
            ),
            "world_model_frozen": True,
            "official_bc_checkpoint": str(
                resolve_path(root, config["policies"]["official_bc_checkpoint"])
            ),
            "neorl_protocol": {
                "rollout_horizon": config["training"]["rollout_horizon"],
                "gradient_steps": config["training"]["gradient_steps"],
                "learning_rate": config["training"]["learning_rate"],
                "policy_and_value_hidden_layers": config["model"],
                "behavior_kl_direction": "D_KL(pi_behavior || pi)",
                "behavior_kl_coefficient": config["training"]["behavior_kl_coefficient"],
                "coefficient_source": "explicit project setting; NeoRL v2 paper does not report the coefficient",
            },
            "sources": config["sources"],
        },
        "methods": methods,
    }
    metrics_path.parent.mkdir(parents=True, exist_ok=True)
    with metrics_path.open("w", encoding="utf-8") as handle:
        json.dump(results, handle, indent=2)
    _write_episode_csv(
        resolve_path(root, config["outputs"]["episode_csv"]), seeds, methods
    )
    histories = {
        variant: _read_training_history(
            resolve_path(root, config["outputs"][variant]["training_history_csv"])
        )
        for variant in ("with_kl", "without_kl")
    }
    _plot_results(
        results,
        histories,
        resolve_path(root, config["outputs"]["comparison_figure"]),
    )
    return results
