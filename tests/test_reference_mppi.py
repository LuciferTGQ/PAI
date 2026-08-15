import numpy as np
import torch

from pytorch_mppi import MPPI


def test_reference_mppi_optimizes_deterministic_integrator() -> None:
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
    assert action is not None
    planned = controller.U.detach().clone()
    state = torch.tensor([[0.0]])
    planned_cost = 0.0
    zero_cost = 0.0
    for step in range(8):
        state = dynamics(state, planned[step : step + 1])
        planned_cost += float(running_cost(state, planned[step : step + 1]).item())
        zero_cost += goal**2

    assert float(action[0]) > 0.3
    assert planned_cost < 0.25 * zero_cost
    assert torch.isfinite(controller.omega).all()
    assert abs(float(controller.omega.sum()) - 1.0) < 1e-5
