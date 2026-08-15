from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict

from src.evaluation.icem_diagnostics import _episode, _write_csv, _write_json
from src.strategy.icem_mpc import ICEMMPCPolicy
from src.strategy.official_bc import OfficialBCPolicy
from src.utils.config import load_config, resolve_path
from src.world_model.interface import FrozenWorldModel


def evaluate_icem_parameter_check(config_path: str | Path) -> Dict[str, Any]:
    config, root = load_config(config_path)
    policies = config["policies"]
    evaluation = config["evaluation"]
    outputs = config["outputs"]
    seeds = [int(seed) for seed in evaluation["seeds"]]
    metadata = {
        "role": evaluation["role"],
        "world_models": policies["world_models"],
        "seeds": seeds,
        "episode_horizon": int(evaluation["episode_horizon"]),
        "common": config["common"],
        "variants": config["variants"],
        "source": config["source"],
        "scope": "minimal beta and sampling-budget check; not a grid search",
    }
    metrics_path = resolve_path(root, outputs["metrics_json"])
    progress: Dict[str, Any] = {"metadata": metadata, "episodes": [], "timesteps": []}
    if metrics_path.exists():
        existing = json.loads(metrics_path.read_text(encoding="utf-8"))
        if existing.get("metadata") != metadata:
            raise ValueError("Existing iCEM parameter-check metadata does not match")
        progress = existing
    completed = {
        (row["dataset_scale"], row["config_variant"], int(row["seed"]))
        for row in progress["episodes"]
    }
    device = str(policies["device"])
    bc = OfficialBCPolicy.from_checkpoint(
        resolve_path(root, policies["official_bc_checkpoint"]), device=device
    )
    for dataset_scale, model_config in policies["world_models"].items():
        checkpoint = resolve_path(root, model_config["checkpoint"])
        world_model = FrozenWorldModel(checkpoint, device=device)
        if any(parameter.requires_grad for parameter in world_model.model.parameters()):
            raise RuntimeError("iCEM parameter checks require a Frozen World Model")
        for variant_name, variant in config["variants"].items():
            planner_config = {**config["common"], **variant}
            for seed in seeds:
                key = (dataset_scale, variant_name, seed)
                if key in completed:
                    print(f"reuse dataset={dataset_scale} variant={variant_name} seed={seed}")
                    continue
                policy = ICEMMPCPolicy(world_model, bc, planner_config, seed=seed)
                episode, timesteps = _episode(
                    root,
                    policy,
                    dataset_scale,
                    model_config["architecture"],
                    checkpoint,
                    seed,
                    int(evaluation["episode_horizon"]),
                    strategy_name="reference_iCEM_parameter_check",
                    config_variant=variant_name,
                )
                progress["episodes"].append(episode)
                progress["timesteps"].extend(timesteps)
                print(
                    f"dataset={dataset_scale} variant={variant_name} "
                    f"return={episode['episode_return']:.3f} "
                    f"seconds={episode['runtime_seconds']:.2f}"
                )
                _write_json(metrics_path, progress)
                _write_csv(resolve_path(root, outputs["episodes_csv"]), progress["episodes"])
                _write_csv(resolve_path(root, outputs["timesteps_csv"]), progress["timesteps"])
    return progress
