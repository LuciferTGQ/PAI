from __future__ import annotations

import csv
import json
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
METRICS = ROOT / "outputs" / "metrics"


def _load(name: str):
    return json.loads((METRICS / name).read_text(encoding="utf-8"))


def _summary(rows):
    returns = np.asarray([row["episode_return"] for row in rows], dtype=np.float64)
    return {
        "episodes": len(rows),
        "mean": float(returns.mean()),
        "std": float(returns.std()),
        "median": float(np.median(returns)),
        "min": float(returns.min()),
        "max": float(returns.max()),
        "mean_runtime_seconds": float(np.mean([row["runtime_seconds"] for row in rows])),
        "mean_planning_seconds": float(np.mean([row["planning_seconds"] for row in rows])),
        "mean_model_evaluations": float(np.mean([row["model_evaluations"] for row in rows])),
    }


def main() -> None:
    icem_base = _load("icem_reference_diagnostics.json")
    icem_check = _load("icem_reference_parameter_check.json")
    icem_white = _load("icem_m10000_white_validation.json")
    mppi_base = _load("mppi_reference_diagnostics.json")
    mppi_validation = _load("mppi_reference_validation.json")

    rows = []
    for row in icem_base["episodes"]:
        if row["dataset_scale"] == "M-1000":
            rows.append({**row, "planner": "iCEM", "frozen_variant": "beta2_budget40"})
    for row in icem_check["episodes"]:
        if row["dataset_scale"] == "M-10000" and row["config_variant"] == "white_beta0_budget40":
            rows.append({**row, "planner": "iCEM", "frozen_variant": "beta0_budget40"})
    for row in icem_white["episodes"]:
        rows.append({**row, "planner": "iCEM", "frozen_variant": "beta0_budget40"})
    for source in (mppi_base, mppi_validation):
        for row in source["episodes"]:
            rows.append({**row, "planner": "MPPI", "frozen_variant": "reference_512"})

    rows = sorted(rows, key=lambda row: (row["planner"], row["dataset_scale"], int(row["seed"])))
    summaries = {
        f"{planner}@{scale}": _summary(
            [row for row in rows if row["planner"] == planner and row["dataset_scale"] == scale]
        )
        for planner, scale in (
            ("iCEM", "M-1000"),
            ("iCEM", "M-10000"),
            ("MPPI", "M-1000"),
            ("MPPI", "M-10000"),
        )
    }
    result = {
        "status": "frozen_for_main_system_matrix",
        "development_seeds": [42, 43, 44, 45, 46],
        "reserved_final_seeds": list(range(100, 110)),
        "frozen_configs": {
            "iCEM": {
                "common": {
                    "horizon": 10,
                    "population": 40,
                    "elites": 10,
                    "iterations": 3,
                    "initial_std": 0.5,
                    "min_std": 0.0,
                    "momentum": 0.1,
                    "population_decay": 1.25,
                    "reuse_fraction": 0.3,
                },
                "noise_beta": {"M-100": 2.0, "M-1000": 2.0, "M-10000": 0.0},
            },
            "MPPI": {
                "horizon": 10,
                "num_samples": 512,
                "noise_std": [0.15, 0.15, 0.15],
                "temperature": 100.0,
            },
        },
        "sources": {
            "iCEM": {
                "repository": "https://github.com/martius-lab/iCEM",
                "commit": "98c1c1fe2bfc94a87e4658b6c793f4e81bc30203",
            },
            "MPPI": {
                "repository": "https://github.com/UM-ARM-Lab/pytorch_mppi",
                "tag": "v0.9.1",
                "commit": "e04a569cd4215e705f9013145c496fc59cb25ed6",
            },
        },
        "diagnosis": {
            "iCEM": "toy passes; M-10000 beta>=2 causes model-exploitation interaction; beta0 restores control",
            "MPPI": "custom implementation poor, reference implementation strong and high-ESS on both World Models",
            "not_claimed": "iCEM/MPPI are unsuitable for IB",
        },
        "summaries": summaries,
        "episodes": rows,
    }
    json_path = METRICS / "reference_planner_freeze.json"
    temporary = json_path.with_suffix(json_path.suffix + ".tmp")
    temporary.write_text(json.dumps(result, indent=2), encoding="utf-8")
    temporary.replace(json_path)
    with (METRICS / "reference_planner_freeze_episodes.csv").open(
        "w", newline="", encoding="utf-8"
    ) as handle:
        fieldnames = list(
            dict.fromkeys(key for row in rows for key in row)
        )
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    print(json.dumps(summaries, indent=2))


if __name__ == "__main__":
    main()
