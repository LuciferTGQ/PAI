from __future__ import annotations

import csv
import json
import time
from pathlib import Path
from typing import Any, Dict

import numpy as np

from src.evaluation.simulator import _make_official_ib, _summarize
from src.strategy.cem_mpc import CEMMPCPolicy
from src.strategy.icem_mpc import ICEMMPCPolicy
from src.strategy.mbppo import MBPPOPolicy, train_mbppo_variant
from src.strategy.official_bc import OfficialBCPolicy
from src.strategy.reference_mppi import ReferenceMPPIPolicy
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


def _write_csv(path: Path, rows: list[Dict[str, Any]], fields: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)
    for attempt in range(8):
        try:
            temporary.replace(path)
            return
        except (OSError, PermissionError):
            if attempt == 7:
                raise
            time.sleep(0.25 * (attempt + 1))


def _policy_episode(
    root: Path,
    policy: Any,
    dataset_scale: str,
    architecture: str,
    checkpoint: str,
    method: str,
    seed: int,
    horizon: int,
) -> tuple[Dict[str, Any], list[Dict[str, Any]]]:
    env = _make_official_ib(seed, root)
    observation = np.asarray(env.reset(), dtype=np.float32)
    if hasattr(policy, "reset"):
        policy.reset(seed)
    rows: list[Dict[str, Any]] = []
    cumulative_reward = 0.0
    clipped_scalars = 0
    action_scalars = 0
    model_evaluations = 0
    episode_started = time.perf_counter()
    for step in range(horizon):
        inference_started = time.perf_counter()
        raw_action = np.asarray(policy.act(observation), dtype=np.float32)
        latency = time.perf_counter() - inference_started
        action = np.clip(raw_action, -1.0, 1.0)
        clipped = action != raw_action
        clipped_scalars += int(np.count_nonzero(clipped))
        action_scalars += int(action.size)
        observation, reward, done, _ = env.step(action)
        observation = np.asarray(observation, dtype=np.float32)
        reward = float(reward)
        cumulative_reward += reward
        diagnostics = dict(getattr(policy, "last_diagnostics", {}))
        step_model_evaluations = int(diagnostics.get("model_evaluations", 0))
        model_evaluations += step_model_evaluations
        rows.append(
            {
                "dataset_scale": dataset_scale,
                "world_model_architecture": architecture,
                "world_model_checkpoint": checkpoint,
                "strategy": method,
                "seed": seed,
                "step": step,
                "reward": reward,
                "cumulative_reward": cumulative_reward,
                "action_0": float(action[0]),
                "action_1": float(action[1]),
                "action_2": float(action[2]),
                "action_clipped": bool(np.any(clipped)),
                "planner_latency_seconds": latency,
                "model_evaluations": step_model_evaluations,
                "diagnostics_json": json.dumps(diagnostics, separators=(",", ":")),
                "source": "main_system_matrix",
            }
        )
        if done:
            break
    runtime = time.perf_counter() - episode_started
    latencies = [float(row["planner_latency_seconds"]) for row in rows]
    episode = {
        "dataset_scale": dataset_scale,
        "world_model_architecture": architecture,
        "world_model_checkpoint": checkpoint,
        "strategy": method,
        "seed": seed,
        "episode_return": cumulative_reward,
        "episode_length": len(rows),
        "action_clipped_fraction": clipped_scalars / max(action_scalars, 1),
        "runtime_seconds": runtime,
        "planning_or_inference_seconds": float(np.sum(latencies)),
        "mean_planner_or_inference_latency_seconds": float(np.mean(latencies)),
        "median_planner_or_inference_latency_seconds": float(np.median(latencies)),
        "model_evaluations": model_evaluations,
        "source": "main_system_matrix",
    }
    return episode, rows


def _reused_episodes(root: Path, freeze_path: Path) -> list[Dict[str, Any]]:
    payload = json.loads(freeze_path.read_text(encoding="utf-8"))
    rows: list[Dict[str, Any]] = []
    for source in payload["episodes"]:
        scale = source["dataset_scale"]
        planner = source.get("planner")
        if scale not in {"M-1000", "M-10000"} or planner not in {"iCEM", "MPPI"}:
            continue
        row = dict(source)
        row["strategy"] = planner
        row["planning_or_inference_seconds"] = row.pop("planning_seconds")
        row["mean_planner_or_inference_latency_seconds"] = row.pop(
            "mean_planner_latency_seconds"
        )
        row["median_planner_or_inference_latency_seconds"] = row.pop(
            "median_planner_latency_seconds"
        )
        row["source"] = str(freeze_path.relative_to(root)).replace("\\", "/")
        rows.append(row)
    return rows


def _load_reused_timesteps(root: Path) -> list[Dict[str, Any]]:
    specifications = [
        ("mppi_reference_diagnostics_timesteps.csv", "MPPI", {42}, {"M-1000", "M-10000"}, None),
        ("mppi_reference_validation_timesteps.csv", "MPPI", {43, 44, 45, 46}, {"M-1000", "M-10000"}, None),
        ("icem_reference_diagnostics_timesteps.csv", "iCEM", {42, 43, 44, 45, 46}, {"M-1000"}, None),
        ("icem_reference_parameter_check_timesteps.csv", "iCEM", {42}, {"M-10000"}, "white_beta0_budget40"),
        ("icem_m10000_white_validation_timesteps.csv", "iCEM", {43, 44, 45, 46}, {"M-10000"}, None),
    ]
    output: list[Dict[str, Any]] = []
    for filename, strategy, seeds, scales, required_variant in specifications:
        path = root / "outputs" / "metrics" / filename
        with path.open("r", encoding="utf-8") as handle:
            for row in csv.DictReader(handle):
                if row["dataset_scale"] not in scales or int(row["seed"]) not in seeds:
                    continue
                if required_variant is not None and row.get("config_variant") != required_variant:
                    continue
                standard = {
                    key: row.get(key, "")
                    for key in (
                        "dataset_scale", "world_model_architecture",
                        "world_model_checkpoint", "seed", "step", "reward",
                        "cumulative_reward", "action_0", "action_1", "action_2",
                        "action_clipped", "planner_latency_seconds", "model_evaluations",
                    )
                }
                standard["strategy"] = strategy
                diagnostics = {
                    key: value
                    for key, value in row.items()
                    if key not in standard and key not in {"strategy", "config_variant"} and value != ""
                }
                standard["diagnostics_json"] = json.dumps(diagnostics, separators=(",", ":"))
                standard["source"] = f"outputs/metrics/{filename}"
                output.append(standard)
    return output


def _summaries(episodes: list[Dict[str, Any]], scales: list[str], seeds: list[int]) -> list[Dict[str, Any]]:
    bc_by_seed = {
        int(row["seed"]): float(row["episode_return"])
        for row in episodes
        if row["strategy"] == "BC"
    }
    rows: list[Dict[str, Any]] = []
    for scale in scales:
        for strategy in ("CEM", "iCEM", "MPPI", "MB-PPO+KL"):
            selected = sorted(
                (
                    row for row in episodes
                    if row["dataset_scale"] == scale and row["strategy"] == strategy
                ),
                key=lambda row: int(row["seed"]),
            )
            returns = [float(row["episode_return"]) for row in selected]
            if len(returns) != len(seeds):
                continue
            deltas = [value - bc_by_seed[seed] for value, seed in zip(returns, seeds)]
            summary = _summarize(returns)
            rows.append(
                {
                    "dataset_scale": scale,
                    "world_model_architecture": selected[0]["world_model_architecture"],
                    "strategy": strategy,
                    "mean_return": summary["mean"],
                    "std_return": summary["std"],
                    "median_return": summary["median"],
                    "mean_delta_vs_bc": float(np.mean(deltas)),
                    "win_rate_vs_bc": float(np.mean(np.asarray(deltas) > 0.0)),
                    "mean_runtime_seconds": float(np.mean([row["runtime_seconds"] for row in selected])),
                    "mean_planning_or_inference_seconds": float(
                        np.mean([row["planning_or_inference_seconds"] for row in selected])
                    ),
                    "mean_model_evaluations": float(
                        np.mean([row["model_evaluations"] for row in selected])
                    ),
                }
            )
    return rows


def run_main_system_matrix(config_path: str | Path) -> Dict[str, Any]:
    config, root = load_config(config_path)
    policy_config = config["policies"]
    evaluation = config["evaluation"]
    output = config["outputs"]
    seeds = [int(seed) for seed in evaluation["seeds"]]
    scales = list(policy_config["world_models"])
    device = str(policy_config["device"])
    bc_checkpoint = resolve_path(root, policy_config["official_bc_checkpoint"])

    mbppo_checkpoints: Dict[str, Path] = {}
    for scale, model_config in policy_config["world_models"].items():
        mbppo_config, mbppo_root = load_config(resolve_path(root, model_config["mbppo_config"]))
        mbppo_checkpoints[scale] = train_mbppo_variant(
            mbppo_config, mbppo_root, use_behavior_kl=True
        )

    metadata = {
        "role": evaluation["role"],
        "development_seeds": seeds,
        "reserved_final_seeds": evaluation["reserved_final_seeds"],
        "reserved_final_seeds_used": False,
        "episode_horizon": evaluation["episode_horizon"],
        "world_models": policy_config["world_models"],
        "world_model_selection_uses_simulator_reward": False,
        "official_bc_checkpoint_fixed_across_scales": str(bc_checkpoint),
        "cem": config["cem"],
        "icem": config["icem"],
        "mppi": config["mppi"],
        "mbppo_checkpoints": {key: str(value) for key, value in mbppo_checkpoints.items()},
        "reused_reference_results": config["reuse"],
    }
    metrics_path = resolve_path(root, output["metrics_json"])
    if metrics_path.exists():
        progress = json.loads(metrics_path.read_text(encoding="utf-8"))
        if progress["metadata"] != metadata:
            raise ValueError("Existing main-matrix metadata does not match")
    else:
        progress = {
            "metadata": metadata,
            "episodes": _reused_episodes(
                root, resolve_path(root, config["reuse"]["planner_freeze_json"])
            ),
            "summaries": [],
        }
        _write_json(metrics_path, progress)
    timestep_path = resolve_path(root, output["timesteps_csv"])
    timestep_fields = [
        "dataset_scale", "world_model_architecture", "world_model_checkpoint",
        "strategy", "seed", "step", "reward", "cumulative_reward", "action_0",
        "action_1", "action_2", "action_clipped", "planner_latency_seconds",
        "model_evaluations", "diagnostics_json", "source",
    ]
    if timestep_path.exists():
        with timestep_path.open("r", encoding="utf-8") as handle:
            timesteps = list(csv.DictReader(handle))
    else:
        timesteps = _load_reused_timesteps(root)
    timestep_completed = {
        (row["dataset_scale"], row["strategy"], int(row["seed"])) for row in timesteps
    }
    incomplete_episode_keys = {
        (row["dataset_scale"], row["strategy"], int(row["seed"]))
        for row in progress["episodes"]
        if (row["dataset_scale"], row["strategy"], int(row["seed"]))
        not in timestep_completed
    }
    if incomplete_episode_keys:
        progress["episodes"] = [
            row for row in progress["episodes"]
            if (row["dataset_scale"], row["strategy"], int(row["seed"]))
            not in incomplete_episode_keys
        ]
        _write_json(metrics_path, progress)
        print(f"repair incomplete timestep episodes: {sorted(incomplete_episode_keys)}")
    completed = {
        (row["dataset_scale"], row["strategy"], int(row["seed"]))
        for row in progress["episodes"]
    }
    bc = OfficialBCPolicy.from_checkpoint(bc_checkpoint, device=device)
    for seed in seeds:
        key = ("BC-fixed", "BC", seed)
        if key not in completed:
            episode, rows = _policy_episode(
                root, bc, "BC-fixed", "OfficialBC", str(bc_checkpoint), "BC", seed,
                int(evaluation["episode_horizon"]),
            )
            progress["episodes"].append(episode)
            timesteps.extend(rows)
            completed.add(key)
            _write_json(metrics_path, progress)
            _write_csv(timestep_path, timesteps, timestep_fields)
            print(f"strategy=BC seed={seed} return={episode['episode_return']:.3f}")

    for scale, model_config in policy_config["world_models"].items():
        checkpoint = resolve_path(root, model_config["checkpoint"])
        world_model = FrozenWorldModel(checkpoint, device=device)
        if any(parameter.requires_grad for parameter in world_model.model.parameters()):
            raise RuntimeError("Main matrix requires Frozen World Models")
        mbppo = MBPPOPolicy.from_checkpoint(mbppo_checkpoints[scale], device=device)
        planner_factories = {
            "CEM": lambda seed: CEMMPCPolicy(world_model, bc, config["cem"], seed=seed),
            "iCEM": lambda seed: ICEMMPCPolicy(
                world_model, bc,
                {**config["icem"], "noise_beta": config["icem"]["noise_beta"][scale]},
                seed=seed,
            ),
            "MPPI": lambda seed: ReferenceMPPIPolicy(world_model, config["mppi"], seed=seed),
            "MB-PPO+KL": lambda _seed: mbppo,
        }
        for strategy, factory in planner_factories.items():
            for seed in seeds:
                key = (scale, strategy, seed)
                if key in completed:
                    continue
                episode, rows = _policy_episode(
                    root, factory(seed), scale, model_config["architecture"],
                    str(checkpoint), strategy, seed, int(evaluation["episode_horizon"]),
                )
                progress["episodes"].append(episode)
                timesteps.extend(rows)
                completed.add(key)
                _write_json(metrics_path, progress)
                _write_csv(timestep_path, timesteps, timestep_fields)
                print(
                    f"dataset={scale} strategy={strategy} seed={seed} "
                    f"return={episode['episode_return']:.3f}"
                )
    progress["summaries"] = _summaries(progress["episodes"], scales, seeds)
    _write_json(metrics_path, progress)
    episode_fields = [
        "dataset_scale", "world_model_architecture", "world_model_checkpoint", "strategy",
        "seed", "episode_return", "episode_length", "action_clipped_fraction",
        "runtime_seconds", "planning_or_inference_seconds",
        "mean_planner_or_inference_latency_seconds",
        "median_planner_or_inference_latency_seconds", "model_evaluations", "source",
    ]
    _write_csv(resolve_path(root, output["episodes_csv"]), progress["episodes"], episode_fields)
    _write_csv(resolve_path(root, output["summary_csv"]), progress["summaries"], list(progress["summaries"][0]))
    return progress
