from __future__ import annotations

from typing import Any, Dict

import numpy as np
import torch

try:
    from pytorch_mppi import MPPI
except ImportError as error:  # pragma: no cover
    raise ImportError(
        "Reference MPPI requires pytorch-mppi==0.9.1; "
        "install requirements-planners.txt"
    ) from error

from src.world_model.interface import FrozenWorldModel


class ReferenceMPPIPolicy:
    """Thin IB adapter around UM-ARM-Lab's PyTorch MPPI implementation."""

    def __init__(
        self,
        world_model: FrozenWorldModel,
        config: Dict[str, Any],
        seed: int = 42,
    ) -> None:
        self.world_model = world_model
        self.device = world_model.device
        self.dtype = torch.float32
        self.horizon = int(config["horizon"])
        self.num_samples = int(config["num_samples"])
        self.action_low = torch.as_tensor(
            config["action_low"], dtype=self.dtype, device=self.device
        )
        self.action_high = torch.as_tensor(
            config["action_high"], dtype=self.dtype, device=self.device
        )
        noise_std = torch.as_tensor(
            config["noise_std"], dtype=self.dtype, device=self.device
        )
        noise_sigma = torch.diag(noise_std.square())
        self.temperature = float(config["temperature"])
        self._seed = int(seed)
        torch.manual_seed(self._seed)
        if self.device.type == "cuda":
            torch.cuda.manual_seed_all(self._seed)
        self.controller = MPPI(
            self._dynamics,
            self._running_cost,
            nx=world_model.obs_dim,
            noise_sigma=noise_sigma,
            num_samples=self.num_samples,
            horizon=self.horizon,
            device=self.device,
            lambda_=self.temperature,
            u_min=self.action_low,
            u_max=self.action_high,
            u_init=torch.zeros_like(self.action_low),
            rollout_samples=1,
        )
        self.last_effective_sample_size = float("nan")
        self.last_diagnostics: Dict[str, Any] = {}

    def _dynamics(self, state: torch.Tensor, action: torch.Tensor) -> torch.Tensor:
        next_frame = self.world_model.predict_next_frame_tensor(state, action)
        return self.world_model.update_history_tensor(state, next_frame)

    @staticmethod
    def _running_cost(state: torch.Tensor, action: torch.Tensor) -> torch.Tensor:
        del action
        return 3.0 * state[:, 4] + state[:, 5]

    def reset(self, seed: int) -> None:
        self._seed = int(seed)
        torch.manual_seed(self._seed)
        if self.device.type == "cuda":
            torch.cuda.manual_seed_all(self._seed)
        self.controller.reset()
        self.last_effective_sample_size = float("nan")
        self.last_diagnostics = {}

    @torch.inference_mode()
    def act(self, observation: np.ndarray) -> np.ndarray:
        state = torch.as_tensor(
            observation, dtype=self.dtype, device=self.device
        )
        action = self.controller.command(state)
        omega = self.controller.omega
        if omega is None:
            raise RuntimeError("Reference MPPI returned no importance weights")
        ess = float((1.0 / torch.sum(omega.square())).item())
        self.last_effective_sample_size = ess
        total_cost = self.controller.cost_total
        perturbed_action = self.controller.perturbed_action
        at_bounds = (
            torch.isclose(perturbed_action, self.action_low)
            | torch.isclose(perturbed_action, self.action_high)
        )
        self.last_diagnostics = {
            "effective_sample_size": ess,
            "effective_sample_fraction": ess / self.num_samples,
            "predicted_best_total_cost": float(torch.min(total_cost).item()),
            "predicted_mean_total_cost": float(torch.mean(total_cost).item()),
            "importance_weight_max": float(torch.max(omega).item()),
            "sample_at_bound_fraction": float(at_bounds.float().mean().item()),
            "model_evaluations": self.num_samples * self.horizon,
        }
        return action.detach().cpu().numpy().astype(np.float32)
