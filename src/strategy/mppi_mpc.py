from __future__ import annotations

from typing import Any, Dict

import numpy as np

from src.strategy.cem_mpc import CEMMPCPolicy


class MPPIMPCPolicy(CEMMPCPolicy):
    """Information-theoretic MPPI with a behavior-initialized nominal sequence."""

    def __init__(self, world_model, behavior_policy, config: Dict[str, Any], seed: int = 42):
        self.temperature = float(config["temperature"])
        self.control_cost_weight = float(config.get("control_cost_weight", 1.0))
        self.noise_std = np.asarray(config["noise_std"], dtype=np.float32)
        if self.noise_std.ndim == 0:
            self.noise_std = np.repeat(self.noise_std, 3)
        self._nominal = None
        super().__init__(world_model, behavior_policy, config, seed)

    def reset(self, seed: int) -> None:
        super().reset(seed)
        self._nominal = None

    def act(self, observation: np.ndarray) -> np.ndarray:
        baseline_action = np.clip(
            self.behavior_policy.act(observation), self.action_low, self.action_high
        )
        if self._nominal is None:
            nominal = np.repeat(baseline_action[None, :], self.horizon, axis=0)
        else:
            nominal = np.concatenate((self._nominal[1:], baseline_action[None, :]), axis=0)

        noise = self.rng.standard_normal(
            (self.population, self.horizon, len(self.action_low))
        ).astype(np.float32) * self.noise_std
        sequences = np.clip(
            nominal[None, :, :] + noise, self.action_low, self.action_high
        ).astype(np.float32)
        actual_noise = sequences - nominal[None, :, :]
        rollout_cost = -self._score_sequences(observation, sequences)
        perturbation_cost = self.temperature * self.control_cost_weight * np.sum(
            nominal[None, :, :] * actual_noise / np.square(self.noise_std),
            axis=(1, 2),
        )
        total_cost = rollout_cost + perturbation_cost
        stabilized = -(total_cost - total_cost.min()) / self.temperature
        weights = np.exp(np.clip(stabilized, -80.0, 0.0))
        weights /= np.maximum(weights.sum(), 1e-12)
        self.last_effective_sample_size = float(1.0 / np.sum(np.square(weights)))
        nominal = np.clip(
            nominal + np.tensordot(weights, actual_noise, axes=(0, 0)),
            self.action_low,
            self.action_high,
        )
        self._nominal = nominal.astype(np.float32)
        return self._nominal[0].copy()
