from __future__ import annotations

import csv
import json
import time
from pathlib import Path
from typing import Any, Callable, Dict

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from src.evaluation.simulator import _make_official_ib, _summarize
from src.strategy.icem_mpc import ICEMMPCPolicy
from src.strategy.mppi_mpc import MPPIMPCPolicy
from src.strategy.official_bc import OfficialBCPolicy
from src.utils.config import load_config, resolve_path
from src.world_model.interface import FrozenWorldModel


def _write(path: Path, value: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(f"{path.suffix}.tmp")
    temporary.write_text(json.dumps(value, indent=2), encoding="utf-8")
    temporary.replace(path)


def _write_csv(path: Path, methods: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=(
                "method",
                "seed",
                "return",
                "elapsed_seconds",
                "clipped_action_fraction",
                "final_effective_sample_size",
            ),
        )
        writer.writeheader()
        for method, section in methods.items():
            for seed, episode in sorted(
                section.get("episodes", {}).items(), key=lambda item: int(item[0])
            ):
                writer.writerow({"method": method, "seed": seed, **episode})


def _episode(
    root: Path,
    policy_factory: Callable[[], Any],
    seed: int,
    episode_horizon: int,
) -> Dict[str, float]:
    env = _make_official_ib(seed, root)
    observation = np.asarray(env.reset(), dtype=np.float32)
    policy = policy_factory()
    rewards = []
    clipped_scalars = 0
    action_scalars = 0
    started = time.perf_counter()
    for _ in range(episode_horizon):
        raw_action = np.asarray(policy.act(observation), dtype=np.float32)
        action = np.clip(raw_action, -1.0, 1.0)
        clipped_scalars += int(np.count_nonzero(action != raw_action))
        action_scalars += int(action.size)
        observation, reward, done, _ = env.step(action)
        observation = np.asarray(observation, dtype=np.float32)
        rewards.append(float(reward))
        if done:
            break
    result = {
        "return": float(np.sum(rewards)),
        "elapsed_seconds": time.perf_counter() - started,
        "clipped_action_fraction": clipped_scalars / max(action_scalars, 1),
    }
    if hasattr(policy, "last_effective_sample_size"):
        result["final_effective_sample_size"] = float(
            policy.last_effective_sample_size
        )
    return result


def _finalize_section(section: Dict[str, Any], seeds: list[int]) -> None:
    returns = [section["episodes"][str(seed)]["return"] for seed in seeds]
    section["returns"] = returns
    section["summary"] = _summarize(returns)
    section["mean_episode_seconds"] = float(
        np.mean([section["episodes"][str(seed)]["elapsed_seconds"] for seed in seeds])
    )


def _plot(path: Path, methods: Dict[str, Any]) -> None:
    plt.rcParams.update(
        {
            "font.family": "sans-serif",
            "font.sans-serif": ["Arial", "DejaVu Sans", "Liberation Sans"],
            "svg.fonttype": "none",
            "pdf.fonttype": 42,
            "font.size": 8,
            "axes.spines.right": False,
            "axes.spines.top": False,
            "legend.frameon": False,
        }
    )
    labels = ["CEM", "iCEM", "MPPI"]
    keys = ["cem", "icem", "mppi"]
    colors = ["#7884B4", "#0F4D92", "#B64342"]
    returns = [np.asarray(methods[key]["returns"]) for key in keys]
    runtimes = [np.asarray([episode["elapsed_seconds"] for episode in methods[key]["episodes"].values()]) for key in keys]
    figure, axes = plt.subplots(1, 2, figsize=(7.2, 3.3))
    for index, (label, color, values) in enumerate(zip(labels, colors, returns)):
        axes[0].scatter(np.full(len(values), index), values, color=color, alpha=0.65, s=23)
        axes[0].errorbar(index, values.mean(), yerr=values.std(ddof=1) / np.sqrt(len(values)), color=color, marker="o", capsize=4)
    axes[0].set(
        title="Development-seed performance",
        ylabel="Simulator episode return (higher is better)",
        xticks=range(len(labels)),
        xticklabels=labels,
    )
    axes[1].bar(labels, [values.mean() for values in runtimes], color=colors)
    axes[1].set(title="Online planning cost", ylabel="Seconds per 1,000-step episode")
    for label, axis in zip(("a", "b"), axes):
        axis.grid(axis="y", alpha=0.25)
        axis.text(-0.12, 1.04, label, transform=axis.transAxes, fontsize=8, fontweight="bold")
    figure.tight_layout()
    base = path.with_suffix("")
    base.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(base.with_suffix(".svg"), bbox_inches="tight")
    figure.savefig(base.with_suffix(".pdf"), bbox_inches="tight")
    figure.savefig(base.with_suffix(".png"), dpi=300, bbox_inches="tight")
    plt.close(figure)


def evaluate_planners(config_path: str | Path) -> Dict[str, Any]:
    config, root = load_config(config_path)
    policy_cfg = config["policies"]
    eval_cfg = config["evaluation"]
    output_cfg = config["outputs"]
    seeds = [int(value) for value in eval_cfg["seeds"]]
    episode_horizon = int(eval_cfg["episode_horizon"])
    wm_path = resolve_path(root, policy_cfg["world_model_checkpoint"])
    bc_path = resolve_path(root, policy_cfg["official_bc_checkpoint"])
    metrics_path = resolve_path(root, output_cfg["metrics_json"])
    metadata = {
        "role": eval_cfg["role"],
        "world_model_checkpoint": str(wm_path),
        "world_model_frozen": True,
        "official_bc_checkpoint": str(bc_path),
        "development_seeds": seeds,
        "held_out_seeds_reserved": [int(value) for value in eval_cfg["held_out_seeds"]],
        "episode_horizon": episode_horizon,
        "horizon": int(config["common"]["horizon"]),
        "icem": config["icem"],
        "mppi": config["mppi"],
        "sources": {
            "icem_paper": "https://proceedings.mlr.press/v155/pinneri21a.html",
            "icem_official_code": "https://github.com/martius-lab/iCEM",
            "mppi_paper": "https://doi.org/10.1109/ICRA.2017.7989202",
            "mppi_reference_code": "https://github.com/UM-ARM-Lab/pytorch_mppi",
        },
    }
    progress: Dict[str, Any] = {"metadata": metadata, "methods": {}}
    if metrics_path.exists():
        existing = json.loads(metrics_path.read_text(encoding="utf-8"))
        if existing.get("metadata") != metadata:
            raise ValueError("Existing planner metrics metadata does not match")
        progress = existing

    cem_source = json.loads(
        resolve_path(root, eval_cfg["reuse_cem_metrics"]).read_text(encoding="utf-8")
    )
    cem_section = cem_source["horizons"][str(config["common"]["horizon"])]
    progress["methods"]["cem"] = cem_section

    device = str(policy_cfg["device"])
    bc_policy = OfficialBCPolicy.from_checkpoint(bc_path, device=device)
    world_model = FrozenWorldModel(wm_path, device=device)
    if any(parameter.requires_grad for parameter in world_model.model.parameters()):
        raise RuntimeError("World Model must remain frozen for every planner")

    common = config["common"]
    policies = {
        "icem": lambda seed: ICEMMPCPolicy(
            world_model, bc_policy, {**common, **config["icem"]}, seed=seed
        ),
        "mppi": lambda seed: MPPIMPCPolicy(
            world_model, bc_policy, {**common, **config["mppi"]}, seed=seed
        ),
    }
    for method, factory in policies.items():
        section = progress["methods"].setdefault(method, {"episodes": {}})
        for seed in seeds:
            if str(seed) in section["episodes"]:
                print(f"reuse method={method} seed={seed}")
                continue
            result = _episode(
                root, lambda seed=seed, factory=factory: factory(seed), seed, episode_horizon
            )
            section["episodes"][str(seed)] = result
            print(
                f"method={method} seed={seed} return={result['return']:.3f} "
                f"seconds={result['elapsed_seconds']:.2f}"
            )
            _write(metrics_path, progress)
            _write_csv(resolve_path(root, output_cfg["episode_csv"]), progress["methods"])
        _finalize_section(section, seeds)
        _write(metrics_path, progress)

    _plot(resolve_path(root, output_cfg["comparison_figure"]), progress["methods"])
    _write_csv(resolve_path(root, output_cfg["episode_csv"]), progress["methods"])
    return progress
