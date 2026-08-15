from __future__ import annotations

import csv
import json
from collections import Counter
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
METRICS = ROOT / "outputs/metrics"


def _rows(name: str) -> list[dict[str, str]]:
    with (METRICS / name).open("r", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def _check_episode_timesteps(
    prefix: str, key_fields: tuple[str, str], expected_groups: int,
    expected_seeds: int,
) -> dict[str, object]:
    episodes = _rows(f"{prefix}_episodes.csv")
    timesteps = _rows(f"{prefix}_timesteps.csv")
    episode_counts = Counter((row[key_fields[0]], row[key_fields[1]]) for row in episodes)
    timestep_counts = Counter((row[key_fields[0]], row[key_fields[1]]) for row in timesteps)
    assert len(episode_counts) == expected_groups
    assert set(episode_counts.values()) == {expected_seeds}
    assert set(timestep_counts.values()) == {expected_seeds * 1000}
    assert set(episode_counts) == set(timestep_counts)
    return {
        "episode_rows": len(episodes),
        "timestep_rows": len(timesteps),
        "groups": {" | ".join(key): episode_counts[key] for key in sorted(episode_counts)},
    }


def _check_summary(
    prefix: str,
    episode_group_fields: tuple[str, str],
    summary_group_fields: tuple[str, str] | None = None,
) -> int:
    episodes = _rows(f"{prefix}_episodes.csv")
    summaries = _rows(f"{prefix}_summary.csv")
    summary_group_fields = summary_group_fields or episode_group_fields
    for summary in summaries:
        key = tuple(summary[field] for field in summary_group_fields)
        values = np.asarray(
            [
                float(row["episode_return"])
                for row in episodes
                if tuple(row[field] for field in episode_group_fields) == key
            ],
            dtype=np.float64,
        )
        assert len(values) > 0
        assert np.isclose(values.mean(), float(summary["mean_return"]), atol=1e-8)
        assert np.isclose(values.std(), float(summary["std_return"]), atol=1e-8)
        assert np.isclose(np.median(values), float(summary["median_return"]), atol=1e-8)
    return len(summaries)


def main() -> None:
    world_models = _rows("world_model_5x3_common_validation_models.csv")
    assert len(world_models) == 15
    assert sum(row["selected_best_world_model"] == "True" for row in world_models) == 3
    report = {
        "world_model_matrix": {"model_rows": 15, "selected_best_world_models": 3},
        "development_main_matrix": _check_episode_timesteps(
            "main_system_matrix_development", ("dataset_scale", "strategy"), 13, 5
        ),
        "world_model_control_cross": _check_episode_timesteps(
            "world_model_control_cross_m1000", ("world_model_architecture", "strategy"), 10, 5
        ),
        "final_fresh_seed_evaluation": _check_episode_timesteps(
            "final_selected_systems_fresh_seeds", ("dataset_scale", "strategy"), 4, 10
        ),
    }
    report["development_main_matrix"]["summary_rows"] = _check_summary(
        "main_system_matrix_development", ("dataset_scale", "strategy")
    )
    report["world_model_control_cross"]["summary_rows"] = _check_summary(
        "world_model_control_cross_m1000",
        ("world_model_architecture", "strategy"),
        ("architecture", "strategy"),
    )
    report["final_fresh_seed_evaluation"]["summary_rows"] = _check_summary(
        "final_selected_systems_fresh_seeds", ("dataset_scale", "strategy")
    )
    development_seeds = {42, 43, 44, 45, 46}
    final_seeds = {100, 101, 102, 103, 104, 105, 106, 107, 108, 109}
    assert development_seeds.isdisjoint(final_seeds)
    histories = sorted(METRICS.glob("mbppo_main_*_with_kl_training.csv")) + sorted(
        METRICS.glob("mbppo_cross_*_with_kl_training.csv")
    ) + [METRICS / "mbppo_ib_m1000_gru_with_kl_training.csv"]
    for path in histories:
        rows = _rows(path.name)
        assert int(float(rows[-1]["gradient_step"])) == 5000
        assert "behavior_kl" in rows[-1] and "model_reward_mean" in rows[-1]
    report["mbppo_training_histories"] = {
        "files": [path.relative_to(ROOT).as_posix() for path in histories],
        "all_reach_gradient_step": 5000,
    }
    report["seed_protocol"] = {
        "development": sorted(development_seeds),
        "final_fresh": sorted(final_seeds),
        "disjoint": True,
    }
    output = METRICS / "experiment_record_audit.json"
    output.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
