from __future__ import annotations

import csv
import importlib.metadata
import json
from pathlib import Path

import torch
from pytorch_mppi import MPPI


ROOT = Path(__file__).resolve().parents[1]


def main() -> None:
    torch.manual_seed(19)
    goal = 3.0

    def dynamics(state, action):
        return state + action

    def running_cost(state, action):
        del action
        return torch.square(state[:, 0] - goal)

    controller = MPPI(
        dynamics,
        running_cost,
        nx=1,
        noise_sigma=torch.tensor([[0.5**2]], dtype=torch.float32),
        num_samples=2048,
        horizon=8,
        lambda_=0.1,
        u_min=torch.tensor([-1.0]),
        u_max=torch.tensor([1.0]),
        u_init=torch.tensor([0.0]),
        device="cpu",
    )
    action = None
    for _ in range(3):
        action = controller.command(
            torch.tensor([0.0]), shift_nominal_trajectory=False
        )
    planned = controller.U.detach().clone()
    state = torch.tensor([[0.0]])
    planned_cost = 0.0
    zero_cost = 0.0
    for step in range(8):
        state = dynamics(state, planned[step : step + 1])
        planned_cost += float(running_cost(state, planned[step : step + 1]).item())
        zero_cost += goal**2
    ess = float((1.0 / torch.sum(controller.omega.square())).item())
    result = {
        "source": {
            "repository": "https://github.com/UM-ARM-Lab/pytorch_mppi",
            "tag": "v0.9.1",
            "commit": "e04a569cd4215e705f9013145c496fc59cb25ed6",
            "installed_version": importlib.metadata.version("pytorch-mppi"),
            "core_algorithm_modified": False,
        },
        "toy_objective": {
            "seed": 19,
            "dynamics": "x_next = x + u",
            "goal": goal,
            "horizon": 8,
            "num_samples": 2048,
            "refinement_steps": 3,
            "baseline_zero_action_cost": zero_cost,
            "planned_cost": planned_cost,
            "cost_ratio": planned_cost / zero_cost,
            "first_action": float(action[0]),
            "effective_sample_size": ess,
            "importance_weight_sum": float(controller.omega.sum().item()),
            "passed": bool(float(action[0]) > 0.3 and planned_cost < 0.25 * zero_cost),
        },
        "ib_adapter_scope": [
            "FrozenWorldModel tensor dynamics",
            "IB running cost 3*fatigue + consumption",
            "action bounds and numpy policy interface",
        ],
    }
    metrics = ROOT / "outputs" / "metrics"
    metrics.mkdir(parents=True, exist_ok=True)
    path = metrics / "mppi_reference_audit.json"
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(result, indent=2), encoding="utf-8")
    temporary.replace(path)
    with (metrics / "mppi_reference_toy_objective.csv").open(
        "w", newline="", encoding="utf-8"
    ) as handle:
        writer = csv.DictWriter(handle, fieldnames=list(result["toy_objective"]))
        writer.writeheader()
        writer.writerow(result["toy_objective"])
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
