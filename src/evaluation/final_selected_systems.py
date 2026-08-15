from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any, Dict

import numpy as np

from src.evaluation.main_system_matrix import _policy_episode, _write_csv, _write_json
from src.evaluation.simulator import _summarize
from src.strategy.cem_mpc import CEMMPCPolicy
from src.strategy.official_bc import OfficialBCPolicy
from src.strategy.reference_mppi import ReferenceMPPIPolicy
from src.utils.config import load_config, resolve_path
from src.world_model.interface import FrozenWorldModel


def run_final_selected_systems(config_path: str | Path) -> Dict[str, Any]:
    config, root = load_config(config_path)
    policies = config["policies"]
    evaluation = config["evaluation"]
    outputs = config["outputs"]
    seeds = [int(seed) for seed in evaluation["seeds"]]
    device = str(policies["device"])
    bc_checkpoint = resolve_path(root, policies["official_bc_checkpoint"])
    metadata = {
        "role": evaluation["role"],
        "fresh_final_seeds": seeds,
        "selection_or_tuning_on_final_seeds": False,
        "episode_horizon": evaluation["episode_horizon"],
        "development_selection_source": evaluation["development_selection_source"],
        "selected_systems": policies["selected_systems"],
        "official_bc_checkpoint": str(bc_checkpoint),
        "world_model_selection_uses_simulator_reward": False,
        "cem": config["cem"],
        "mppi": config["mppi"],
    }
    metrics_path = resolve_path(root, outputs["metrics_json"])
    timestep_path = resolve_path(root, outputs["timesteps_csv"])
    if metrics_path.exists():
        progress = json.loads(metrics_path.read_text(encoding="utf-8"))
        if progress["metadata"] != metadata:
            raise ValueError("Existing final-evaluation metadata does not match")
    else:
        progress = {"metadata": metadata, "episodes": [], "summaries": []}
        _write_json(metrics_path, progress)
    if timestep_path.exists():
        with timestep_path.open("r", encoding="utf-8") as handle:
            timesteps = list(csv.DictReader(handle))
    else:
        timesteps = []
    timestep_fields = [
        "dataset_scale", "world_model_architecture", "world_model_checkpoint", "strategy",
        "seed", "step", "reward", "cumulative_reward", "action_0", "action_1", "action_2",
        "action_clipped", "planner_latency_seconds", "model_evaluations", "diagnostics_json",
        "source",
    ]
    timestep_keys = {
        (row["dataset_scale"], row["strategy"], int(row["seed"])) for row in timesteps
    }
    incomplete = {
        (row["dataset_scale"], row["strategy"], int(row["seed"]))
        for row in progress["episodes"]
        if (row["dataset_scale"], row["strategy"], int(row["seed"]))
        not in timestep_keys
    }
    if incomplete:
        progress["episodes"] = [
            row for row in progress["episodes"]
            if (row["dataset_scale"], row["strategy"], int(row["seed"])) not in incomplete
        ]
        _write_json(metrics_path, progress)
    completed = {
        (row["dataset_scale"], row["strategy"], int(row["seed"]))
        for row in progress["episodes"]
    }
    bc = OfficialBCPolicy.from_checkpoint(bc_checkpoint, device=device)
    for seed in seeds:
        key = ("BC-fixed", "BC", seed)
        if key in completed:
            continue
        episode, rows = _policy_episode(
            root, bc, "BC-fixed", "OfficialBC", str(bc_checkpoint), "BC", seed,
            int(evaluation["episode_horizon"]),
        )
        progress["episodes"].append(episode)
        timesteps.extend(rows)
        completed.add(key)
        _write_json(metrics_path, progress)
        _write_csv(timestep_path, timesteps, timestep_fields)
        print(f"final strategy=BC seed={seed} return={episode['episode_return']:.3f}")

    for scale, selected in policies["selected_systems"].items():
        checkpoint = resolve_path(root, selected["checkpoint"])
        world_model = FrozenWorldModel(checkpoint, device=device)
        strategy = selected["strategy"]
        for seed in seeds:
            key = (scale, strategy, seed)
            if key in completed:
                continue
            if strategy == "CEM":
                policy = CEMMPCPolicy(world_model, bc, config["cem"], seed=seed)
            elif strategy == "MPPI":
                policy = ReferenceMPPIPolicy(world_model, config["mppi"], seed=seed)
            else:
                raise ValueError(f"Unsupported frozen final strategy: {strategy}")
            episode, rows = _policy_episode(
                root, policy, scale, selected["architecture"], str(checkpoint), strategy,
                seed, int(evaluation["episode_horizon"]),
            )
            progress["episodes"].append(episode)
            timesteps.extend(rows)
            completed.add(key)
            _write_json(metrics_path, progress)
            _write_csv(timestep_path, timesteps, timestep_fields)
            print(
                f"final dataset={scale} strategy={strategy} seed={seed} "
                f"return={episode['episode_return']:.3f}"
            )

    bc_by_seed = {
        int(row["seed"]): float(row["episode_return"])
        for row in progress["episodes"] if row["strategy"] == "BC"
    }
    summaries: list[Dict[str, Any]] = []
    for scale, selected in policies["selected_systems"].items():
        rows = sorted(
            (row for row in progress["episodes"] if row["dataset_scale"] == scale),
            key=lambda row: int(row["seed"]),
        )
        returns = [float(row["episode_return"]) for row in rows]
        deltas = [value - bc_by_seed[seed] for value, seed in zip(returns, seeds)]
        summary = _summarize(returns)
        summaries.append(
            {
                "dataset_scale": scale,
                "world_model_architecture": selected["architecture"],
                "strategy": selected["strategy"],
                "mean_return": summary["mean"],
                "std_return": summary["std"],
                "median_return": summary["median"],
                "mean_delta_vs_bc": float(np.mean(deltas)),
                "win_rate_vs_bc": float(np.mean(np.asarray(deltas) > 0.0)),
                "mean_runtime_seconds": float(np.mean([row["runtime_seconds"] for row in rows])),
                "mean_model_evaluations": float(np.mean([row["model_evaluations"] for row in rows])),
            }
        )
    progress["summaries"] = summaries
    _write_json(metrics_path, progress)
    episode_fields = [
        "dataset_scale", "world_model_architecture", "world_model_checkpoint", "strategy",
        "seed", "episode_return", "episode_length", "action_clipped_fraction",
        "runtime_seconds", "planning_or_inference_seconds",
        "mean_planner_or_inference_latency_seconds",
        "median_planner_or_inference_latency_seconds", "model_evaluations", "source",
    ]
    _write_csv(resolve_path(root, outputs["episodes_csv"]), progress["episodes"], episode_fields)
    _write_csv(resolve_path(root, outputs["summary_csv"]), summaries, list(summaries[0]))
    return progress
