from __future__ import annotations

from typing import Any, Dict

import numpy as np

from src.strategy.official_bc import OfficialBCPolicy
from src.world_model.interface import FrozenWorldModel


def ib_reward_from_frame(next_frame: np.ndarray) -> np.ndarray:
    """Official IB classic reward from the predicted next visible frame."""
    frames = np.asarray(next_frame, dtype=np.float32)
    return -(3.0 * frames[..., 4] + frames[..., 5])


class CEMMPCPolicy:
    """Receding-horizon CEM planning against an immutable world model."""

    def __init__(
        self,
        world_model: FrozenWorldModel,
        behavior_policy: OfficialBCPolicy,
        config: Dict[str, Any],
        seed: int = 42,
    ) -> None:
        self.world_model = world_model
        self.behavior_policy = behavior_policy
        self.horizon = int(config["horizon"])
        self.population = int(config["population"])
        self.elites = int(config["elites"])
        self.iterations = int(config["iterations"])
        self.initial_std = float(config["initial_std"])
        self.min_std = float(config["min_std"])
        self.action_low = np.asarray(config["action_low"], dtype=np.float32)
        self.action_high = np.asarray(config["action_high"], dtype=np.float32)
        if not 0 < self.elites <= self.population:
            raise ValueError("CEM elites must be in [1, population]")
        self.last_diagnostics: Dict[str, Any] = {}
        self.reset(seed)

    def reset(self, seed: int) -> None:
        self.rng = np.random.default_rng(seed)
        self.last_diagnostics = {}

    def _score_sequences(self, observation: np.ndarray, sequences: np.ndarray) -> np.ndarray:
        states = np.repeat(
            np.asarray(observation, dtype=np.float32).reshape(1, -1),
            len(sequences),
            axis=0,
        )
        returns = np.zeros(len(sequences), dtype=np.float32)
        for step in range(self.horizon):
            next_frames = self.world_model.predict_next_frame(states, sequences[:, step])
            returns += ib_reward_from_frame(next_frames)
            states = np.concatenate((next_frames, states[:, :-6]), axis=1)
        return returns

    def act(self, observation: np.ndarray) -> np.ndarray:
        baseline_action = np.clip(
            self.behavior_policy.act(observation), self.action_low, self.action_high
        )
        mean = np.repeat(baseline_action[None, :], self.horizon, axis=0)
        std = np.full_like(mean, self.initial_std)
        best_score = -np.inf
        sampled_at_bounds = 0
        sampled_scalars = 0
        for _ in range(self.iterations):
            noise = self.rng.standard_normal(
                (self.population, self.horizon, len(self.action_low))
            ).astype(np.float32)
            sequences = np.clip(mean[None, :, :] + std[None, :, :] * noise,
                                self.action_low, self.action_high)
            sampled_at_bounds += int(
                np.count_nonzero(
                    np.isclose(sequences, self.action_low)
                    | np.isclose(sequences, self.action_high)
                )
            )
            sampled_scalars += int(sequences.size)
            scores = self._score_sequences(observation, sequences)
            best_score = max(best_score, float(np.max(scores)))
            elite_indices = np.argpartition(scores, -self.elites)[-self.elites:]
            elite_sequences = sequences[elite_indices]
            mean = elite_sequences.mean(axis=0)
            std = np.maximum(elite_sequences.std(axis=0), self.min_std)
        self.last_diagnostics = {
            "predicted_best_return": best_score,
            "sample_at_bound_fraction": sampled_at_bounds / max(sampled_scalars, 1),
            "model_evaluations": self.population * self.iterations * self.horizon,
        }
        return np.clip(mean[0], self.action_low, self.action_high).astype(np.float32)
