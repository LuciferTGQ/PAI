from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any, Dict

import numpy as np

from src.evaluation.icem_diagnostics import _write_csv, _write_json
from src.evaluation.simulator import _make_official_ib, _summarize
from src.strategy.reference_mppi import ReferenceMPPIPolicy
from src.utils.config import load_config, resolve_path
from src.world_model.interface import FrozenWorldModel


def _episode(
    root: Path,
    policy: ReferenceMPPIPolicy,
    dataset_scale: str,
    architecture: str,
    checkpoint: Path,
    seed: int,
    episode_horizon: int,
) -> tuple[Dict[str, Any], list[Dict[str, Any]]]:
    env = _make_official_ib(seed, root)
    observation = np.asarray(env.reset(), dtype=np.float32)
    policy.reset(seed)
    cumulative_reward = 0.0
    clipped_scalars = 0
    action_scalars = 0
    rows = []
    episode_started = time.perf_counter()
    for step in range(episode_horizon):
        planning_started = time.perf_counter()
        raw_action = np.asarray(policy.act(observation), dtype=np.float32)
        planner_latency = time.perf_counter() - planning_started
        action = np.clip(raw_action, -1.0, 1.0)
        clipped_scalars += int(np.count_nonzero(action != raw_action))
        action_scalars += int(action.size)
        observation, reward, done, _ = env.step(action)
        observation = np.asarray(observation, dtype=np.float32)
        reward = float(reward)
        cumulative_reward += reward
        diagnostics = policy.last_diagnostics
        rows.append(
            {
                "dataset_scale": dataset_scale,
                "world_model_architecture": architecture,
                "world_model_checkpoint": checkpoint.as_posix(),
                "strategy": "reference_MPPI",
                "seed": seed,
                "step": step,
                "reward": reward,
                "cumulative_reward": cumulative_reward,
                "action_0": float(action[0]),
                "action_1": float(action[1]),
                "action_2": float(action[2]),
                "action_clipped": bool(np.any(action != raw_action)),
                "planner_latency_seconds": planner_latency,
                **diagnostics,
            }
        )
        if done:
            break
    runtime = time.perf_counter() - episode_started
    latencies = [row["planner_latency_seconds"] for row in rows]
    return (
        {
            "dataset_scale": dataset_scale,
            "world_model_architecture": architecture,
            "world_model_checkpoint": checkpoint.as_posix(),
            "strategy": "reference_MPPI",
            "seed": seed,
            "episode_return": cumulative_reward,
            "episode_length": len(rows),
            "action_clipped_fraction": clipped_scalars / max(action_scalars, 1),
            "runtime_seconds": runtime,
            "planning_seconds": float(np.sum(latencies)),
            "mean_planner_latency_seconds": float(np.mean(latencies)),
            "median_planner_latency_seconds": float(np.median(latencies)),
            "model_evaluations": len(rows) * policy.num_samples * policy.horizon,
            "mean_effective_sample_size": float(
                np.mean([row["effective_sample_size"] for row in rows])
            ),
        },
        rows,
    )


def evaluate_mppi_diagnostics(config_path: str | Path) -> Dict[str, Any]:
    config, root = load_config(config_path)
    policies = config["policies"]
    evaluation = config["evaluation"]
    outputs = config["outputs"]
    seeds = [int(seed) for seed in evaluation["seeds"]]
    metadata = {
        "role": evaluation["role"],
        "development_seeds": seeds,
        "reserved_final_seeds": [int(seed) for seed in evaluation["reserved_final_seeds"]],
        "episode_horizon": int(evaluation["episode_horizon"]),
        "world_models": policies["world_models"],
        "mppi": config["mppi"],
        "source": config["source"],
        "toy_test": "outputs/metrics/mppi_reference_audit.json",
        "custom_implementation_preserved": "src/strategy/mppi_mpc.py",
        "world_model_selection_uses_simulator_reward": False,
    }
    metrics_path = resolve_path(root, outputs["metrics_json"])
    progress: Dict[str, Any] = {"metadata": metadata, "episodes": [], "timesteps": [], "summaries": {}}
    if metrics_path.exists():
        existing = json.loads(metrics_path.read_text(encoding="utf-8"))
        if existing.get("metadata") != metadata:
            raise ValueError("Existing reference MPPI metadata does not match")
        progress = existing
    completed = {
        (row["dataset_scale"], int(row["seed"])) for row in progress["episodes"]
    }
    device = str(policies["device"])
    for dataset_scale, model_config in policies["world_models"].items():
        checkpoint = resolve_path(root, model_config["checkpoint"])
        world_model = FrozenWorldModel(checkpoint, device=device)
        if any(parameter.requires_grad for parameter in world_model.model.parameters()):
            raise RuntimeError("Reference MPPI requires a Frozen World Model")
        for seed in seeds:
            if (dataset_scale, seed) in completed:
                print(f"reuse dataset={dataset_scale} seed={seed}")
                continue
            policy = ReferenceMPPIPolicy(world_model, config["mppi"], seed=seed)
            episode, timesteps = _episode(
                root,
                policy,
                dataset_scale,
                model_config["architecture"],
                checkpoint,
                seed,
                int(evaluation["episode_horizon"]),
            )
            progress["episodes"].append(episode)
            progress["timesteps"].extend(timesteps)
            print(
                f"dataset={dataset_scale} seed={seed} "
                f"return={episode['episode_return']:.3f} "
                f"ess={episode['mean_effective_sample_size']:.2f} "
                f"seconds={episode['runtime_seconds']:.2f}"
            )
            _write_json(metrics_path, progress)
            _write_csv(resolve_path(root, outputs["episodes_csv"]), progress["episodes"])
            _write_csv(resolve_path(root, outputs["timesteps_csv"]), progress["timesteps"])
        returns = [
            row["episode_return"] for row in progress["episodes"]
            if row["dataset_scale"] == dataset_scale
        ]
        progress["summaries"][dataset_scale] = _summarize(returns)
        _write_json(metrics_path, progress)
    return progress
