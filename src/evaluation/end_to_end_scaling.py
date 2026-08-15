from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any, Dict, Iterable

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np

from src.evaluation.planner_comparison import _episode, _finalize_section, _write
from src.strategy.cem_mpc import CEMMPCPolicy
from src.strategy.official_bc import OfficialBCPolicy
from src.utils.config import load_config, resolve_path
from src.world_model.interface import FrozenWorldModel


def _write_csv(path: Path, methods: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=["data_scale", "seed", "return", "elapsed_seconds", "clipped_action_fraction"],
        )
        writer.writeheader()
        for scale, section in methods.items():
            for seed, episode in section["episodes"].items():
                writer.writerow({"data_scale": scale, "seed": seed, **episode})


def _plot(path: Path, methods: Dict[str, Any], common_rows: list[Dict[str, Any]]) -> None:
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
    scales = [100, 1000, 10000]
    colors = ["#7884B4", "#B64342", "#0F4D92"]
    figure, axes = plt.subplots(1, 2, figsize=(8.2, 3.6))
    figure.suptitle(
        "Fixed GRU architecture and CEM H=10 isolate World Model data-scale effects",
        fontsize=11,
        fontweight="bold",
        y=1.02,
    )
    for index, (scale, color) in enumerate(zip(scales, colors)):
        values = np.asarray(methods[str(scale)]["returns"], dtype=np.float64)
        x = np.full(len(values), scale) * np.exp(np.linspace(-0.025, 0.025, len(values)))
        axes[0].scatter(x, values, color=color, s=24, alpha=0.8)
        axes[0].errorbar(
            scale,
            values.mean(),
            yerr=values.std(ddof=1) / np.sqrt(len(values)),
            fmt="o",
            color=color,
            capsize=4,
        )
    axes[0].set_xscale("log")
    axes[0].set_xticks(scales, ["100", "1,000", "10,000"])
    axes[0].set(
        xlabel="World Model training trajectories",
        ylabel="Held-out simulator return (higher is better)",
        title="a  End-to-end control",
    )
    axes[0].title.set_fontweight("bold")
    axes[0].title.set_ha("left")
    axes[0].title.set_position((0, 1.0))
    axes[0].grid(alpha=0.22)

    gru_rows = {
        int(row["data_scale"]): row
        for row in common_rows
        if row["architecture"] == "GRU"
    }
    for scale, color in zip(scales, colors):
        x_value = gru_rows[scale]["mean_NRMSE_H5_H10_H20"]
        y_value = methods[str(scale)]["summary"]["mean"]
        axes[1].scatter(x_value, y_value, color=color, s=45)
        axes[1].annotate(f"M-{scale}", (x_value, y_value), xytext=(5, 4),
                         textcoords="offset points", color=color)
    axes[1].set(
        xlabel="Common-validation mean NRMSE (H5/H10/H20)",
        ylabel="Held-out simulator return",
        title="b  Dynamics-control relationship",
    )
    axes[1].title.set_fontweight("bold")
    axes[1].title.set_ha("left")
    axes[1].title.set_position((0, 1.0))
    axes[1].grid(alpha=0.22)
    figure.tight_layout()
    base = path.with_suffix("")
    base.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(base.with_suffix(".png"), dpi=300, bbox_inches="tight")
    figure.savefig(base.with_suffix(".pdf"), bbox_inches="tight")
    figure.savefig(base.with_suffix(".svg"), bbox_inches="tight")
    plt.close(figure)


def evaluate_end_to_end_scaling(
    config_path: str | Path, scales: Iterable[int] = (100, 1000, 10000)
) -> Dict[str, Any]:
    config, root = load_config(config_path)
    evaluation = config["evaluation"]
    outputs = config["outputs"]
    scales = [int(scale) for scale in scales]
    seeds = [int(seed) for seed in evaluation["seeds"]]
    metrics_path = resolve_path(root, outputs["metrics_json"])
    selection_path = resolve_path(root, evaluation["world_model_selection"])
    selection = json.loads(selection_path.read_text(encoding="utf-8"))
    metadata = {
        "role": evaluation["role"],
        "seeds": seeds,
        "episode_horizon": int(evaluation["episode_horizon"]),
        "architecture_locked_from_m1000": evaluation["architecture_locked_from_m1000"],
        "strategy_locked_from_m1000": evaluation["strategy_locked_from_m1000"],
        "official_bc_checkpoint_fixed_across_scales": str(
            resolve_path(root, config["policies"]["official_bc_checkpoint"])
        ),
        "cem": config["cem"],
        "selection_or_tuning_on_these_seeds": False,
    }
    progress: Dict[str, Any] = {"metadata": metadata, "data_scales": {}}
    if metrics_path.exists():
        progress = json.loads(metrics_path.read_text(encoding="utf-8"))
        if progress.get("metadata") != metadata:
            raise ValueError("Existing end-to-end scaling metadata does not match")

    device = str(config["policies"]["device"])
    behavior = OfficialBCPolicy.from_checkpoint(
        resolve_path(root, config["policies"]["official_bc_checkpoint"]), device=device
    )
    reused_m1000 = json.loads(
        resolve_path(root, evaluation["reuse_m1000_heldout"]).read_text(encoding="utf-8")
    )["methods"]["cem"]
    for scale in scales:
        key = str(scale)
        selected = selection["selections"].get(key, {}).get("GRU", {}).get("selected")
        if selected is None:
            raise ValueError(f"M-{scale} GRU checkpoint has not been selected yet")
        if scale == 1000:
            section = dict(reused_m1000)
            section["reused_without_rerun"] = True
            section["world_model_checkpoint"] = selected["checkpoint"]
            progress["data_scales"][key] = section
            _write(metrics_path, progress)
            continue
        section = progress["data_scales"].setdefault(
            key,
            {"episodes": {}, "world_model_checkpoint": selected["checkpoint"]},
        )
        checkpoint = root / selected["checkpoint"]
        world_model = FrozenWorldModel(checkpoint, device=device)
        if any(parameter.requires_grad for parameter in world_model.model.parameters()):
            raise RuntimeError("Scaling World Model must remain frozen")
        for seed in seeds:
            if str(seed) in section["episodes"]:
                print(f"reuse end-to-end scale={scale} seed={seed}")
                continue
            result = _episode(
                root,
                lambda seed=seed: CEMMPCPolicy(
                    world_model, behavior, config["cem"], seed=seed
                ),
                seed,
                int(evaluation["episode_horizon"]),
            )
            section["episodes"][str(seed)] = result
            print(f"end-to-end scale={scale} seed={seed} return={result['return']:.3f}")
            _write(metrics_path, progress)
            _write_csv(resolve_path(root, outputs["episode_csv"]), progress["data_scales"])
        _finalize_section(section, seeds)
        _write(metrics_path, progress)
        del world_model

    _write_csv(resolve_path(root, outputs["episode_csv"]), progress["data_scales"])
    if set(progress["data_scales"]) == {"100", "1000", "10000"}:
        common_rows = selection.get("fixed_common_validation_metrics", [])
        _plot(resolve_path(root, outputs["comparison_figure"]), progress["data_scales"], common_rows)
    return progress
