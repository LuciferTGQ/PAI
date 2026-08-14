from __future__ import annotations

import csv
import json
import time
from pathlib import Path
from typing import Any, Dict

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from src.evaluation.simulator import _make_official_ib, _summarize
from src.strategy.cem_mpc import CEMMPCPolicy
from src.strategy.official_bc import OfficialBCPolicy
from src.utils.config import load_config, resolve_path
from src.world_model.interface import FrozenWorldModel


def select_horizon_one_standard_error(
    returns_by_horizon: Dict[int, list[float]],
) -> Dict[str, Any]:
    if not returns_by_horizon or any(not values for values in returns_by_horizon.values()):
        raise ValueError("Every horizon must have at least one return")
    means = {horizon: float(np.mean(values)) for horizon, values in returns_by_horizon.items()}
    best_horizon = max(means, key=means.get)
    best_values = np.asarray(returns_by_horizon[best_horizon], dtype=np.float64)
    best_sem = (
        float(best_values.std(ddof=1) / np.sqrt(len(best_values)))
        if len(best_values) > 1
        else 0.0
    )
    cutoff = means[best_horizon] - best_sem
    eligible = sorted(horizon for horizon, mean in means.items() if mean >= cutoff)
    return {
        "best_mean_horizon": int(best_horizon),
        "best_mean_return": means[best_horizon],
        "best_standard_error": best_sem,
        "one_standard_error_cutoff": cutoff,
        "eligible_horizons": eligible,
        "selected_horizon": int(eligible[0]),
    }


def _write_progress(path: Path, progress: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(progress, indent=2), encoding="utf-8")


def _write_episode_csv(path: Path, results: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=(
                "horizon",
                "seed",
                "return",
                "elapsed_seconds",
                "clipped_action_fraction",
            ),
        )
        writer.writeheader()
        for horizon, values in results.items():
            for seed, episode in sorted(
                values["episodes"].items(), key=lambda item: int(item[0])
            ):
                writer.writerow({"horizon": horizon, "seed": seed, **episode})


def _evaluate_seed(
    root: Path,
    world_model: FrozenWorldModel,
    bc_policy: OfficialBCPolicy,
    cem_config: Dict[str, Any],
    seed: int,
    episode_horizon: int,
) -> Dict[str, float]:
    env = _make_official_ib(seed, root)
    observation = np.asarray(env.reset(), dtype=np.float32)
    policy = CEMMPCPolicy(world_model, bc_policy, cem_config, seed=seed)
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
    return {
        "return": float(np.sum(rewards)),
        "elapsed_seconds": time.perf_counter() - started,
        "clipped_action_fraction": clipped_scalars / max(action_scalars, 1),
    }


def _plot(path: Path, completed: Dict[str, Any], selection: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
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
    horizons = sorted(int(value) for value in completed)
    returns = [
        np.asarray(
            [episode["return"] for episode in completed[str(horizon)]["episodes"].values()]
        )
        for horizon in horizons
    ]
    means = np.asarray([values.mean() for values in returns])
    sems = np.asarray(
        [
            values.std(ddof=1) / np.sqrt(len(values)) if len(values) > 1 else 0.0
            for values in returns
        ]
    )
    figure, axis = plt.subplots(figsize=(6.2, 4.2))
    axis.errorbar(
        horizons, means, yerr=sems, color="#0F4D92", marker="o", capsize=4
    )
    for horizon, values in zip(horizons, returns):
        axis.scatter(
            np.full(len(values), horizon), values, color="#7884B4", alpha=0.65, s=22
        )
    selected = int(selection["selected_horizon"])
    axis.scatter(
        selected,
        means[horizons.index(selected)],
        marker="*",
        s=130,
        color="#B64342",
        zorder=5,
    )
    axis.set(
        title="CEM planning-horizon selection on development seeds",
        xlabel="Planning horizon",
        ylabel="Simulator episode return (higher is better)",
        xticks=horizons,
    )
    axis.grid(alpha=0.25)
    figure.tight_layout()
    output_base = path.with_suffix("")
    figure.savefig(output_base.with_suffix(".svg"), bbox_inches="tight")
    figure.savefig(output_base.with_suffix(".pdf"), bbox_inches="tight")
    figure.savefig(output_base.with_suffix(".png"), dpi=300, bbox_inches="tight")
    plt.close(figure)


def evaluate_cem_horizons(config_path: str | Path) -> Dict[str, Any]:
    config, root = load_config(config_path)
    policy_config = config["policies"]
    evaluation_config = config["evaluation"]
    output_config = config["outputs"]
    metrics_path = resolve_path(root, output_config["metrics_json"])
    episode_csv_path = resolve_path(root, output_config["episode_csv"])
    world_model_path = resolve_path(root, policy_config["world_model_checkpoint"])
    bc_path = resolve_path(root, policy_config["official_bc_checkpoint"])
    device = str(policy_config["device"])
    horizons = [int(value) for value in config["cem"]["horizons"]]
    seeds = [int(value) for value in evaluation_config["seeds"]]
    episode_horizon = int(evaluation_config["episode_horizon"])
    metadata = {
        "role": evaluation_config["role"],
        "world_model_checkpoint": str(world_model_path),
        "official_bc_checkpoint": str(bc_path),
        "world_model_frozen": True,
        "development_seeds": seeds,
        "held_out_seeds_reserved": [
            int(value) for value in evaluation_config["held_out_seeds"]
        ],
        "episode_horizon": episode_horizon,
        "selection_rule": evaluation_config["selection_rule"],
        "cem_base": {
            key: value for key, value in config["cem"].items() if key != "horizons"
        },
    }
    progress: Dict[str, Any] = {"metadata": metadata, "horizons": {}}
    if metrics_path.exists():
        existing = json.loads(metrics_path.read_text(encoding="utf-8"))
        if existing.get("metadata") != metadata:
            raise ValueError("Existing CEM horizon metrics metadata does not match this run")
        progress = existing

    bc_policy = OfficialBCPolicy.from_checkpoint(bc_path, device=device)
    world_model = FrozenWorldModel(world_model_path, device=device)
    if any(parameter.requires_grad for parameter in world_model.model.parameters()):
        raise RuntimeError("Selected deployment World Model must remain frozen")

    for horizon in horizons:
        horizon_key = str(horizon)
        section = progress["horizons"].setdefault(horizon_key, {"episodes": {}})
        cem_config = {**config["cem"], "horizon": horizon}
        cem_config.pop("horizons", None)
        for seed in seeds:
            seed_key = str(seed)
            if seed_key in section["episodes"]:
                print(f"reuse horizon={horizon} seed={seed}")
                continue
            episode = _evaluate_seed(
                root, world_model, bc_policy, cem_config, seed, episode_horizon
            )
            section["episodes"][seed_key] = episode
            print(
                f"horizon={horizon} seed={seed} return={episode['return']:.3f} "
                f"seconds={episode['elapsed_seconds']:.2f}"
            )
            _write_progress(metrics_path, progress)
            _write_episode_csv(episode_csv_path, progress["horizons"])
        ordered_returns = [
            section["episodes"][str(seed)]["return"] for seed in seeds
        ]
        section["returns"] = ordered_returns
        section["summary"] = _summarize(ordered_returns)
        section["mean_episode_seconds"] = float(
            np.mean(
                [section["episodes"][str(seed)]["elapsed_seconds"] for seed in seeds]
            )
        )
        _write_progress(metrics_path, progress)

    selection = select_horizon_one_standard_error(
        {
            horizon: progress["horizons"][str(horizon)]["returns"]
            for horizon in horizons
        }
    )
    progress["selection"] = selection
    _write_progress(metrics_path, progress)
    _plot(
        resolve_path(root, output_config["comparison_figure"]),
        progress["horizons"],
        selection,
    )
    print(json.dumps(selection, indent=2))
    return progress
