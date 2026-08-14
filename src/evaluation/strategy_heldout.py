from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any, Dict

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.ticker import FuncFormatter

from src.evaluation.planner_comparison import _episode, _finalize_section, _write
from src.evaluation.simulator import _summarize
from src.strategy.cem_mpc import CEMMPCPolicy
from src.strategy.icem_mpc import ICEMMPCPolicy
from src.strategy.mbppo import MBPPOPolicy
from src.strategy.mppi_mpc import MPPIMPCPolicy
from src.strategy.official_bc import OfficialBCPolicy
from src.utils.config import load_config, resolve_path
from src.world_model.interface import FrozenWorldModel


def _write_csv(path: Path, methods: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=[
                "method",
                "seed",
                "return",
                "elapsed_seconds",
                "clipped_action_fraction",
                "final_effective_sample_size",
            ],
        )
        writer.writeheader()
        for method, section in methods.items():
            for seed, episode in section["episodes"].items():
                writer.writerow({"method": method, "seed": seed, **episode})


def _reuse_bc_returns(path: Path, seeds: list[int]) -> Dict[str, Any]:
    source = json.loads(path.read_text(encoding="utf-8"))
    source_seeds = [int(seed) for seed in source["metadata"]["seeds"]]
    returns_by_seed = dict(zip(source_seeds, source["official_bc"]["returns"]))
    episodes = {
        str(seed): {
            "return": float(returns_by_seed[seed]),
            "elapsed_seconds": 0.0,
            "clipped_action_fraction": None,
        }
        for seed in seeds
    }
    returns = [episodes[str(seed)]["return"] for seed in seeds]
    return {
        "episodes": episodes,
        "returns": returns,
        "summary": _summarize(returns),
        "mean_episode_seconds": 0.0,
        "reused_without_rerun": True,
    }


def _plot(path: Path, methods: Dict[str, Any]) -> None:
    mpl.rcParams.update(
        {
            "font.family": "sans-serif",
            "font.sans-serif": ["Arial", "DejaVu Sans"],
            "font.size": 9,
            "svg.fonttype": "none",
            "pdf.fonttype": 42,
            "axes.spines.top": False,
            "axes.spines.right": False,
        }
    )
    order = ["official_bc", "cem", "icem", "mppi", "mbppo_with_kl", "mbppo_without_kl"]
    labels = ["BC", "CEM", "iCEM", "MPPI", "MB-PPO\n+ KL", "MB-PPO\nno KL"]
    colors = ["#7A7A7A", "#0072B2", "#56B4E9", "#CC79A7", "#009E73", "#D55E00"]
    figure, axes = plt.subplots(1, 2, figsize=(9.2, 3.8), gridspec_kw={"width_ratios": [1.55, 1]})
    figure.suptitle(
        "Held-out seeds confirm CEM H=10 as the fixed M-1000 strategy",
        fontsize=12,
        fontweight="bold",
        y=1.02,
    )
    visible_floor = -380000.0
    for index, (method, color) in enumerate(zip(order, colors)):
        values = np.asarray(methods[method]["returns"], dtype=np.float64)
        mean = values.mean()
        if mean < visible_floor:
            axes[0].scatter(index, visible_floor + 2500, marker="v", color=color, s=45, zorder=4)
            axes[0].annotate(
                f"{mean / 1000:.0f}k",
                xy=(index, visible_floor + 2500),
                xytext=(index, visible_floor + 10500),
                ha="center",
                fontsize=7.5,
                color=color,
                arrowprops={"arrowstyle": "->", "color": color, "lw": 0.7},
            )
        else:
            x = np.full(len(values), index) + np.linspace(-0.08, 0.08, len(values))
            axes[0].scatter(x, values, color=color, s=24, alpha=0.82, zorder=3)
            sem = values.std(ddof=1) / np.sqrt(len(values))
            axes[0].errorbar(index, mean, yerr=sem, fmt="_", markersize=14,
                             color="black", capsize=4, linewidth=1.1, zorder=4)
    axes[0].set_xticks(range(len(order)), labels)
    axes[0].set_ylim(visible_floor, -250000)
    axes[0].yaxis.set_major_formatter(FuncFormatter(lambda value, _pos: f"{value / 1000:.0f}k"))
    axes[0].set_ylabel("Simulator episode return (higher is better)")
    axes[0].set_title("a  Final held-out comparison (n=5)", loc="left", fontweight="bold")
    axes[0].grid(axis="y", alpha=0.22)

    development = {
        "CEM": -269373.94535004953,
        "iCEM": -337220.1720391664,
        "MPPI": -1002582.6735,
        "MB-PPO + KL": -284714.8205153441,
        "MB-PPO no KL": -883708.645071265,
    }
    heldout = {
        "CEM": methods["cem"]["summary"]["mean"],
        "iCEM": methods["icem"]["summary"]["mean"],
        "MPPI": methods["mppi"]["summary"]["mean"],
        "MB-PPO + KL": methods["mbppo_with_kl"]["summary"]["mean"],
        "MB-PPO no KL": methods["mbppo_without_kl"]["summary"]["mean"],
    }
    planner_order = list(development)
    dev_ranks = {key: rank + 1 for rank, key in enumerate(sorted(planner_order, key=development.get, reverse=True))}
    heldout_ranks = {key: rank + 1 for rank, key in enumerate(sorted(planner_order, key=heldout.get, reverse=True))}
    for key, color in zip(planner_order, colors[1:]):
        axes[1].plot([0, 1], [dev_ranks[key], heldout_ranks[key]], marker="o", color=color,
                     linewidth=1.5, label=key)
    axes[1].set_xticks([0, 1], ["Development", "Held-out"])
    axes[1].set_yticks(range(1, len(planner_order) + 1))
    axes[1].invert_yaxis()
    axes[1].set_ylabel("Strategy rank (1 is best)")
    axes[1].set_title("b  Ranking consistency", loc="left", fontweight="bold")
    axes[1].grid(alpha=0.22)
    axes[1].legend(frameon=False, fontsize=7.5, loc="center left", bbox_to_anchor=(1.0, 0.5))
    figure.tight_layout()
    base = path.with_suffix("")
    base.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(base.with_suffix(".png"), dpi=300, bbox_inches="tight")
    figure.savefig(base.with_suffix(".pdf"), bbox_inches="tight")
    figure.savefig(base.with_suffix(".svg"), bbox_inches="tight")
    plt.close(figure)


def evaluate_heldout_strategies(config_path: str | Path) -> Dict[str, Any]:
    config, root = load_config(config_path)
    policy_config = config["policies"]
    evaluation = config["evaluation"]
    outputs = config["outputs"]
    seeds = [int(seed) for seed in evaluation["seeds"]]
    episode_horizon = int(evaluation["episode_horizon"])
    metrics_path = resolve_path(root, outputs["metrics_json"])
    wm_path = resolve_path(root, policy_config["world_model_checkpoint"])
    bc_path = resolve_path(root, policy_config["official_bc_checkpoint"])
    metadata = {
        "role": evaluation["role"],
        "selection_locked_before_held_out": evaluation["selection_locked_before_held_out"],
        "held_out_seeds": seeds,
        "episode_horizon": episode_horizon,
        "world_model_checkpoint": str(wm_path),
        "world_model_frozen": True,
        "common_horizon": config["common"]["horizon"],
        "mppi_variant": "original reference-form configuration; stability diagnostic excluded after development",
    }
    progress: Dict[str, Any] = {"metadata": metadata, "methods": {}}
    if metrics_path.exists():
        progress = json.loads(metrics_path.read_text(encoding="utf-8"))
        if progress.get("metadata") != metadata:
            raise ValueError("Existing held-out metadata does not match the locked protocol")

    progress["methods"]["official_bc"] = _reuse_bc_returns(
        resolve_path(root, evaluation["reuse_bc_metrics"]), seeds
    )
    device = str(policy_config["device"])
    behavior = OfficialBCPolicy.from_checkpoint(bc_path, device=device)
    world_model = FrozenWorldModel(wm_path, device=device)
    if any(parameter.requires_grad for parameter in world_model.model.parameters()):
        raise RuntimeError("World Model must remain frozen during held-out reporting")
    common = config["common"]
    fixed_mbppo = {
        "mbppo_with_kl": MBPPOPolicy.from_checkpoint(
            resolve_path(root, policy_config["mbppo_with_kl_checkpoint"]), device=device
        ),
        "mbppo_without_kl": MBPPOPolicy.from_checkpoint(
            resolve_path(root, policy_config["mbppo_without_kl_checkpoint"]), device=device
        ),
    }
    factories = {
        "cem": lambda seed: CEMMPCPolicy(world_model, behavior, {**common, **config["cem"]}, seed=seed),
        "icem": lambda seed: ICEMMPCPolicy(world_model, behavior, {**common, **config["icem"]}, seed=seed),
        "mppi": lambda seed: MPPIMPCPolicy(world_model, behavior, {**common, **config["mppi"]}, seed=seed),
        "mbppo_with_kl": lambda _seed: fixed_mbppo["mbppo_with_kl"],
        "mbppo_without_kl": lambda _seed: fixed_mbppo["mbppo_without_kl"],
    }
    for method, factory in factories.items():
        section = progress["methods"].setdefault(method, {"episodes": {}})
        for seed in seeds:
            if str(seed) in section["episodes"]:
                print(f"reuse heldout method={method} seed={seed}")
                continue
            result = _episode(root, lambda seed=seed, factory=factory: factory(seed), seed, episode_horizon)
            section["episodes"][str(seed)] = result
            print(f"heldout method={method} seed={seed} return={result['return']:.3f}")
            _write(metrics_path, progress)
            _write_csv(resolve_path(root, outputs["episode_csv"]), progress["methods"])
        _finalize_section(section, seeds)
        _write(metrics_path, progress)

    _write_csv(resolve_path(root, outputs["episode_csv"]), progress["methods"])
    _plot(resolve_path(root, outputs["comparison_figure"]), progress["methods"])
    return progress
