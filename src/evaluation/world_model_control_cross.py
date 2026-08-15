from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any, Dict

import numpy as np

from src.evaluation.main_system_matrix import _policy_episode, _write_csv, _write_json
from src.evaluation.simulator import _summarize
from src.strategy.mbppo import MBPPOPolicy, train_mbppo_variant
from src.strategy.official_bc import OfficialBCPolicy
from src.strategy.reference_mppi import ReferenceMPPIPolicy
from src.utils.config import load_config, resolve_path
from src.world_model.interface import FrozenWorldModel


TIMESTEP_FIELDS = [
    "dataset_scale", "world_model_architecture", "world_model_checkpoint", "strategy",
    "seed", "step", "reward", "cumulative_reward", "action_0", "action_1", "action_2",
    "action_clipped", "planner_latency_seconds", "model_evaluations", "diagnostics_json",
    "source",
]


def _validation_metrics(root: Path) -> Dict[str, Dict[str, float]]:
    path = root / "outputs/metrics/world_model_5x3_common_validation_models.csv"
    output: Dict[str, Dict[str, float]] = {}
    with path.open("r", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            if row["data_scale"] == "1000" and row["selected_within_architecture"] == "True":
                output[row["architecture"]] = {
                    "one_step_NRMSE": float(row["one_step_NRMSE"]),
                    "mean_NRMSE_H5_H10_H20": float(row["mean_NRMSE_H5_H10_H20"]),
                    "NRMSE_H50": float(row["NRMSE_H50"]),
                }
    return output


def run_world_model_control_cross(config_path: str | Path) -> Dict[str, Any]:
    config, root = load_config(config_path)
    policies = config["policies"]
    evaluation = config["evaluation"]
    outputs = config["outputs"]
    seeds = [int(seed) for seed in evaluation["seeds"]]
    device = str(policies["device"])
    bc_checkpoint = resolve_path(root, policies["official_bc_checkpoint"])

    mbppo_checkpoints: Dict[str, Path] = {}
    for architecture, model in policies["world_models"].items():
        if "mbppo_checkpoint" in model:
            mbppo_checkpoints[architecture] = resolve_path(root, model["mbppo_checkpoint"])
        else:
            training_config, training_root = load_config(resolve_path(root, model["mbppo_config"]))
            mbppo_checkpoints[architecture] = train_mbppo_variant(
                training_config, training_root, use_behavior_kl=True
            )

    metadata = {
        "role": evaluation["role"],
        "development_seeds": seeds,
        "reserved_final_seeds": evaluation["reserved_final_seeds"],
        "reserved_final_seeds_used": False,
        "episode_horizon": evaluation["episode_horizon"],
        "dataset_scale": "M-1000",
        "world_models": policies["world_models"],
        "world_model_selection_uses_simulator_reward": False,
        "planning_method_selected_from_main_matrix": "MPPI",
        "mppi": config["mppi"],
        "mbppo_checkpoints": {key: str(value) for key, value in mbppo_checkpoints.items()},
        "reuse": config["reuse"],
    }
    metrics_path = resolve_path(root, outputs["metrics_json"])
    timestep_path = resolve_path(root, outputs["timesteps_csv"])
    if metrics_path.exists():
        progress = json.loads(metrics_path.read_text(encoding="utf-8"))
        if progress["metadata"] != metadata:
            raise ValueError("Existing cross-matrix metadata does not match")
    else:
        main = json.loads(resolve_path(root, config["reuse"]["main_metrics"]).read_text(encoding="utf-8"))
        reused_episodes = [
            {**row, "source": config["reuse"]["main_metrics"]}
            for row in main["episodes"]
            if row["dataset_scale"] == "M-1000"
            and row["world_model_architecture"] == "Transformer-2L"
            and row["strategy"] in {"MPPI", "MB-PPO+KL"}
        ]
        progress = {"metadata": metadata, "episodes": reused_episodes, "summaries": []}
        _write_json(metrics_path, progress)
    if timestep_path.exists():
        with timestep_path.open("r", encoding="utf-8") as handle:
            timesteps = list(csv.DictReader(handle))
    else:
        with resolve_path(root, config["reuse"]["main_timesteps"]).open(
            "r", encoding="utf-8"
        ) as handle:
            timesteps = [
                {**row, "source": config["reuse"]["main_timesteps"]}
                for row in csv.DictReader(handle)
                if row["dataset_scale"] == "M-1000"
                and row["world_model_architecture"] == "Transformer-2L"
                and row["strategy"] in {"MPPI", "MB-PPO+KL"}
            ]
    timestep_keys = {
        (row["world_model_architecture"], row["strategy"], int(row["seed"]))
        for row in timesteps
    }
    incomplete = {
        (row["world_model_architecture"], row["strategy"], int(row["seed"]))
        for row in progress["episodes"]
        if (row["world_model_architecture"], row["strategy"], int(row["seed"]))
        not in timestep_keys
    }
    if incomplete:
        progress["episodes"] = [
            row for row in progress["episodes"]
            if (row["world_model_architecture"], row["strategy"], int(row["seed"]))
            not in incomplete
        ]
        _write_json(metrics_path, progress)
    completed = {
        (row["world_model_architecture"], row["strategy"], int(row["seed"]))
        for row in progress["episodes"]
    }
    bc = OfficialBCPolicy.from_checkpoint(bc_checkpoint, device=device)
    for architecture, model in policies["world_models"].items():
        checkpoint = resolve_path(root, model["checkpoint"])
        world_model = FrozenWorldModel(checkpoint, device=device)
        mbppo = MBPPOPolicy.from_checkpoint(mbppo_checkpoints[architecture], device=device)
        factories = {
            "MPPI": lambda seed: ReferenceMPPIPolicy(world_model, config["mppi"], seed=seed),
            "MB-PPO+KL": lambda _seed: mbppo,
        }
        for strategy, factory in factories.items():
            for seed in seeds:
                key = (architecture, strategy, seed)
                if key in completed:
                    continue
                episode, rows = _policy_episode(
                    root, factory(seed), "M-1000", architecture, str(checkpoint), strategy,
                    seed, int(evaluation["episode_horizon"]),
                )
                progress["episodes"].append(episode)
                timesteps.extend(rows)
                completed.add(key)
                _write_json(metrics_path, progress)
                _write_csv(timestep_path, timesteps, TIMESTEP_FIELDS)
                print(
                    f"architecture={architecture} strategy={strategy} seed={seed} "
                    f"return={episode['episode_return']:.3f}"
                )

    validation = _validation_metrics(root)
    summaries: list[Dict[str, Any]] = []
    for architecture in policies["world_models"]:
        for strategy in ("MPPI", "MB-PPO+KL"):
            selected = sorted(
                (row for row in progress["episodes"] if row["world_model_architecture"] == architecture and row["strategy"] == strategy),
                key=lambda row: int(row["seed"]),
            )
            values = [float(row["episode_return"]) for row in selected]
            summary = _summarize(values)
            summaries.append(
                {
                    "architecture": architecture,
                    "strategy": strategy,
                    "mean_return": summary["mean"],
                    "std_return": summary["std"],
                    "median_return": summary["median"],
                    "mean_runtime_seconds": float(np.mean([row["runtime_seconds"] for row in selected])),
                    "mean_model_evaluations": float(np.mean([row["model_evaluations"] for row in selected])),
                    **validation[architecture],
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
