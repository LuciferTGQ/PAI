from __future__ import annotations

import csv
import json
import time
from pathlib import Path
from typing import Any, Dict

import numpy as np

from src.evaluation.simulator import _make_official_ib, _summarize
from src.strategy.icem_mpc import ICEMMPCPolicy
from src.strategy.official_bc import OfficialBCPolicy
from src.utils.config import load_config, resolve_path
from src.world_model.interface import FrozenWorldModel


def _write_json(path: Path, payload: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    for attempt in range(8):
        try:
            temporary.replace(path)
            return
        except PermissionError:
            if attempt == 7:
                raise
            time.sleep(0.25 * (attempt + 1))


def _write_csv(path: Path, rows: list[Dict[str, Any]]) -> None:
    if not rows:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def _episode(
    root: Path,
    policy: ICEMMPCPolicy,
    dataset_scale: str,
    architecture: str,
    checkpoint: Path,
    seed: int,
    episode_horizon: int,
    strategy_name: str = "reference_iCEM",
    config_variant: str = "reference_beta2_budget40",
) -> tuple[Dict[str, Any], list[Dict[str, Any]]]:
    env = _make_official_ib(seed, root)
    observation = np.asarray(env.reset(), dtype=np.float32)
    policy.reset(seed)
    cumulative_reward = 0.0
    clipped_scalars = 0
    action_scalars = 0
    total_model_evaluations = 0
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
        total_model_evaluations += int(diagnostics["model_evaluations"])
        rows.append(
            {
                "dataset_scale": dataset_scale,
                "world_model_architecture": architecture,
                "world_model_checkpoint": checkpoint.as_posix(),
                "strategy": strategy_name,
                "config_variant": config_variant,
                "seed": seed,
                "step": step,
                "reward": reward,
                "cumulative_reward": cumulative_reward,
                "action_0": float(action[0]),
                "action_1": float(action[1]),
                "action_2": float(action[2]),
                "action_clipped": bool(np.any(action != raw_action)),
                "planner_latency_seconds": planner_latency,
                "predicted_best_return": diagnostics["best_score"],
                "elite_score_mean": diagnostics["elite_score_mean"],
                "elite_score_std": diagnostics["elite_score_std"],
                "sample_clipped_fraction": diagnostics["action_clipped_fraction"],
                "population_schedule": ";".join(
                    str(value) for value in diagnostics["population_schedule"]
                ),
                "model_evaluations": diagnostics["model_evaluations"],
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
            "strategy": strategy_name,
            "config_variant": config_variant,
            "seed": seed,
            "episode_return": cumulative_reward,
            "episode_length": len(rows),
            "action_clipped_fraction": clipped_scalars / max(action_scalars, 1),
            "runtime_seconds": runtime,
            "planning_seconds": float(np.sum(latencies)),
            "mean_planner_latency_seconds": float(np.mean(latencies)),
            "median_planner_latency_seconds": float(np.median(latencies)),
            "model_evaluations": total_model_evaluations,
        },
        rows,
    )


def evaluate_icem_diagnostics(config_path: str | Path) -> Dict[str, Any]:
    config, root = load_config(config_path)
    policy_config = config["policies"]
    eval_config = config["evaluation"]
    output_config = config["outputs"]
    seeds = [int(seed) for seed in eval_config["seeds"]]
    metadata = {
        "role": eval_config["role"],
        "development_seeds": seeds,
        "reserved_final_seeds": [int(seed) for seed in eval_config["reserved_final_seeds"]],
        "episode_horizon": int(eval_config["episode_horizon"]),
        "world_models": policy_config["world_models"],
        "official_bc_checkpoint": policy_config["official_bc_checkpoint"],
        "common": config["common"],
        "icem": config["icem"],
        "source": config["source"],
        "toy_test": "outputs/metrics/icem_official_audit.json",
        "world_model_selection_uses_simulator_reward": False,
    }
    metrics_path = resolve_path(root, output_config["metrics_json"])
    progress: Dict[str, Any] = {"metadata": metadata, "episodes": [], "timesteps": [], "summaries": {}}
    if metrics_path.exists():
        existing = json.loads(metrics_path.read_text(encoding="utf-8"))
        if existing.get("metadata") != metadata:
            raise ValueError("Existing iCEM diagnostic metadata does not match")
        progress = existing
    completed = {
        (row["dataset_scale"], int(row["seed"])) for row in progress["episodes"]
    }
    device = str(policy_config["device"])
    bc = OfficialBCPolicy.from_checkpoint(
        resolve_path(root, policy_config["official_bc_checkpoint"]), device=device
    )
    for dataset_scale, model_config in policy_config["world_models"].items():
        checkpoint = resolve_path(root, model_config["checkpoint"])
        world_model = FrozenWorldModel(checkpoint, device=device)
        if any(parameter.requires_grad for parameter in world_model.model.parameters()):
            raise RuntimeError("iCEM diagnostics require a Frozen World Model")
        planner_config = {**config["common"], **config["icem"]}
        for seed in seeds:
            if (dataset_scale, seed) in completed:
                print(f"reuse dataset={dataset_scale} seed={seed}")
                continue
            policy = ICEMMPCPolicy(world_model, bc, planner_config, seed=seed)
            episode, timesteps = _episode(
                root,
                policy,
                dataset_scale,
                model_config["architecture"],
                checkpoint,
                seed,
                int(eval_config["episode_horizon"]),
            )
            progress["episodes"].append(episode)
            progress["timesteps"].extend(timesteps)
            print(
                f"dataset={dataset_scale} seed={seed} "
                f"return={episode['episode_return']:.3f} "
                f"seconds={episode['runtime_seconds']:.2f}"
            )
            _write_json(metrics_path, progress)
            _write_csv(resolve_path(root, output_config["episodes_csv"]), progress["episodes"])
            _write_csv(resolve_path(root, output_config["timesteps_csv"]), progress["timesteps"])
        returns = [
            row["episode_return"]
            for row in progress["episodes"]
            if row["dataset_scale"] == dataset_scale
        ]
        progress["summaries"][dataset_scale] = {
            **_summarize(returns),
            "mean_runtime_seconds": float(
                np.mean(
                    [
                        row["runtime_seconds"]
                        for row in progress["episodes"]
                        if row["dataset_scale"] == dataset_scale
                    ]
                )
            ),
        }
        _write_json(metrics_path, progress)
    return progress
