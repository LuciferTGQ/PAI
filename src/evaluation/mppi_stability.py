from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any, Dict

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from src.evaluation.planner_comparison import _episode, _finalize_section
from src.strategy.mppi_mpc import MPPIMPCPolicy
from src.strategy.official_bc import OfficialBCPolicy
from src.utils.config import load_config, resolve_path
from src.world_model.interface import FrozenWorldModel


def _save(path: Path, values: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(values, indent=2), encoding="utf-8")


def _csv(path: Path, section: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=(
                "seed",
                "return",
                "elapsed_seconds",
                "clipped_action_fraction",
                "final_effective_sample_size",
            ),
        )
        writer.writeheader()
        for seed, episode in sorted(
            section["episodes"].items(), key=lambda item: int(item[0])
        ):
            writer.writerow({"seed": seed, **episode})


def _plot(path: Path, original: list[float], stable: list[float]) -> None:
    plt.rcParams.update(
        {
            "font.family": "sans-serif",
            "font.sans-serif": ["Arial", "DejaVu Sans", "Liberation Sans"],
            "svg.fonttype": "none",
            "pdf.fonttype": 42,
            "font.size": 8,
            "axes.spines.right": False,
            "axes.spines.top": False,
        }
    )
    values = [np.asarray(original), np.asarray(stable)]
    colors = ["#7884B4", "#B64342"]
    labels = ["Original correction\n(ESS 2–4)", "BC-prior correction\n(ESS target ≈18)"]
    figure, axis = plt.subplots(figsize=(5.5, 3.8))
    for index, (array, color) in enumerate(zip(values, colors)):
        axis.scatter(np.full(len(array), index), array, color=color, alpha=0.65)
        axis.errorbar(
            index,
            array.mean(),
            yerr=array.std(ddof=1) / np.sqrt(len(array)),
            color=color,
            marker="o",
            capsize=4,
        )
    axis.set(
        title="MPPI importance-weight stability correction",
        ylabel="Simulator episode return (higher is better)",
        xticks=(0, 1),
        xticklabels=labels,
    )
    axis.grid(axis="y", alpha=0.25)
    figure.tight_layout()
    base = path.with_suffix("")
    base.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(base.with_suffix(".svg"), bbox_inches="tight")
    figure.savefig(base.with_suffix(".pdf"), bbox_inches="tight")
    figure.savefig(base.with_suffix(".png"), dpi=300, bbox_inches="tight")
    plt.close(figure)


def evaluate_mppi_stability(config_path: str | Path) -> Dict[str, Any]:
    config, root = load_config(config_path)
    policy_cfg = config["policies"]
    eval_cfg = config["evaluation"]
    output_cfg = config["outputs"]
    seeds = [int(value) for value in eval_cfg["seeds"]]
    wm_path = resolve_path(root, policy_cfg["world_model_checkpoint"])
    bc_path = resolve_path(root, policy_cfg["official_bc_checkpoint"])
    metrics_path = resolve_path(root, output_cfg["metrics_json"])
    metadata = {
        "role": eval_cfg["role"],
        "world_model_checkpoint": str(wm_path),
        "world_model_frozen": True,
        "development_seeds": seeds,
        "held_out_seeds_reserved": [int(value) for value in eval_cfg["held_out_seeds"]],
        "episode_horizon": int(eval_cfg["episode_horizon"]),
        "diagnostic_basis": eval_cfg["diagnostic_basis"],
        "mppi": config["mppi"],
    }
    progress: Dict[str, Any] = {"metadata": metadata, "episodes": {}}
    if metrics_path.exists():
        existing = json.loads(metrics_path.read_text(encoding="utf-8"))
        if existing.get("metadata") != metadata:
            raise ValueError("Existing stable MPPI metrics metadata does not match")
        progress = existing

    device = str(policy_cfg["device"])
    bc_policy = OfficialBCPolicy.from_checkpoint(bc_path, device=device)
    world_model = FrozenWorldModel(wm_path, device=device)
    if any(parameter.requires_grad for parameter in world_model.model.parameters()):
        raise RuntimeError("World Model must remain frozen")
    for seed in seeds:
        if str(seed) in progress["episodes"]:
            print(f"reuse stable_mppi seed={seed}")
            continue
        result = _episode(
            root,
            lambda seed=seed: MPPIMPCPolicy(
                world_model, bc_policy, config["mppi"], seed=seed
            ),
            seed,
            int(eval_cfg["episode_horizon"]),
        )
        progress["episodes"][str(seed)] = result
        print(
            f"stable_mppi seed={seed} return={result['return']:.3f} "
            f"ess={result['final_effective_sample_size']:.2f}"
        )
        _save(metrics_path, progress)
        _csv(resolve_path(root, output_cfg["episode_csv"]), progress)
    _finalize_section(progress, seeds)
    _save(metrics_path, progress)
    parent = json.loads(
        resolve_path(root, eval_cfg["parent_metrics"]).read_text(encoding="utf-8")
    )
    _plot(
        resolve_path(root, output_cfg["comparison_figure"]),
        parent["methods"]["mppi"]["returns"],
        progress["returns"],
    )
    return progress
