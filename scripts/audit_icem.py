from __future__ import annotations

import csv
import json
import sys
from pathlib import Path

import colorednoise
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.strategy.icem_mpc import ICEMMPCPolicy, powerlaw_noise


class _UnusedBehaviorPolicy:
    def act(self, observation):
        return np.zeros(1, dtype=np.float32)


class _ToyICEM(ICEMMPCPolicy):
    def __init__(self, target: np.ndarray, config, seed: int):
        self.target = target
        super().__init__(None, _UnusedBehaviorPolicy(), config, seed=seed)

    def _score_sequences(self, observation, sequences):
        return -np.mean(np.square(sequences - self.target[None]), axis=(1, 2))


def _write_json(path: Path, payload) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    temporary.replace(path)


def main() -> None:
    noise_config = {"seed": 17, "samples": 7, "horizon": 11, "action_dim": 3, "beta": 2.0}
    actual = powerlaw_noise(
        np.random.default_rng(noise_config["seed"]),
        noise_config["samples"],
        noise_config["horizon"],
        noise_config["action_dim"],
        noise_config["beta"],
    )
    expected = colorednoise.powerlaw_psd_gaussian(
        noise_config["beta"],
        size=(noise_config["samples"], noise_config["action_dim"], noise_config["horizon"]),
        random_state=np.random.default_rng(noise_config["seed"]),
    ).transpose(0, 2, 1).astype(np.float32)

    target = np.full((8, 1), 0.65, dtype=np.float32)
    toy_config = {
        "horizon": 8,
        "population": 256,
        "elites": 16,
        "iterations": 4,
        "initial_std": 1.0,
        "min_std": 0.0,
        "action_low": [-1.0],
        "action_high": [1.0],
        "momentum": 0.1,
        "population_decay": 1.25,
        "reuse_fraction": 0.3,
        "noise_beta": 2.0,
    }
    policy = _ToyICEM(target, toy_config, seed=23)
    action = policy.act(np.zeros(1, dtype=np.float32))
    best_sequence = np.asarray(policy.last_diagnostics["best_sequence"])
    baseline_mse = float(np.mean(np.square(target)))
    optimized_mse = float(np.mean(np.square(best_sequence - target)))

    audit = {
        "source_of_truth": {
            "repository": "https://github.com/martius-lab/iCEM",
            "commit": "98c1c1fe2bfc94a87e4658b6c793f4e81bc30203",
            "controller": "icem/controllers/icem.py",
            "paper": "https://proceedings.mlr.press/v155/pinneri21a.html",
            "colorednoise_dependency": "colorednoise==2.2.0",
        },
        "feature_audit": [
            {"feature": "colored_noise", "previous": "custom FFT approximation", "corrected": "direct colorednoise.powerlaw_psd_gaussian call", "official_equivalent": True},
            {"feature": "elite_reuse", "previous": "implemented", "corrected": "retained without reevaluation within an optimization call", "official_equivalent": True},
            {"feature": "previous_solution_shift", "previous": "implemented with an extra shift at next call", "corrected": "shift smoothed mean once after planning; shift final elites and sample tail next call", "official_equivalent": True},
            {"feature": "population_decay", "previous": "base / factor**iteration", "corrected": "iterative integer decay with lower bound 2x elites", "official_equivalent": True},
            {"feature": "clipping", "previous": "implemented", "corrected": "sample unconstrained then clip to bounds", "official_equivalent": True},
            {"feature": "best_sequence_preservation", "previous": "implemented", "corrected": "reuse ordered top elites and execute best evaluated final-pool sequence", "official_equivalent": True},
            {"feature": "final_mean_sample", "previous": "implemented", "corrected": "replace sample zero by distribution mean only in final iteration", "official_equivalent": True},
            {"feature": "population_sample_schedule", "previous": "not logged", "corrected": "logged per iteration with exact model-evaluation count", "official_equivalent": True},
            {"feature": "initial_mean", "previous": "BC action", "corrected": "action-bound midpoint", "official_equivalent": True},
            {"feature": "std_floor", "previous": "0.05 in historical config", "corrected": "0.0 in reference-validation config", "official_equivalent": True},
        ],
        "colored_noise_equivalence": {
            "config": noise_config,
            "array_equal": bool(np.array_equal(actual, expected)),
            "max_abs_difference": float(np.max(np.abs(actual - expected))),
        },
        "toy_objective": {
            "seed": 23,
            "target": target.tolist(),
            "config": toy_config,
            "baseline_mse": baseline_mse,
            "optimized_mse": optimized_mse,
            "mse_ratio": optimized_mse / baseline_mse,
            "first_action": action.tolist(),
            "passed": bool(optimized_mse < 0.02 and optimized_mse < 0.1 * baseline_mse),
            "diagnostics": policy.last_diagnostics,
        },
    }
    metrics_dir = ROOT / "outputs" / "metrics"
    _write_json(metrics_dir / "icem_official_audit.json", audit)
    with (metrics_dir / "icem_toy_objective.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=(
                "seed",
                "baseline_mse",
                "optimized_mse",
                "mse_ratio",
                "first_action",
                "population_schedule",
                "model_evaluations",
                "passed",
            ),
        )
        writer.writeheader()
        writer.writerow(
            {
                "seed": 23,
                "baseline_mse": baseline_mse,
                "optimized_mse": optimized_mse,
                "mse_ratio": optimized_mse / baseline_mse,
                "first_action": action.tolist(),
                "population_schedule": policy.last_diagnostics["population_schedule"],
                "model_evaluations": policy.last_diagnostics["model_evaluations"],
                "passed": audit["toy_objective"]["passed"],
            }
        )
    print(json.dumps(audit["colored_noise_equivalence"], indent=2))
    print(json.dumps(audit["toy_objective"], indent=2))


if __name__ == "__main__":
    main()
