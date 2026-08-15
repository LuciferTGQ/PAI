from __future__ import annotations

from typing import Any, Dict

import numpy as np

try:
    import colorednoise
except ImportError as error:  # pragma: no cover - exercised by deployment environments
    raise ImportError(
        "Official iCEM noise requires colorednoise==2.2.0; "
        "install requirements-planners.txt"
    ) from error

from src.strategy.cem_mpc import CEMMPCPolicy


def powerlaw_noise(
    rng: np.random.Generator,
    samples: int,
    horizon: int,
    action_dim: int,
    beta: float,
) -> np.ndarray:
    """Official iCEM colored-noise dependency, correlated along the horizon."""
    if beta <= 0:
        colored = rng.standard_normal((samples, action_dim, horizon))
    else:
        colored = colorednoise.powerlaw_psd_gaussian(
            beta,
            size=(samples, action_dim, horizon),
            random_state=rng,
        )
    return colored.transpose(0, 2, 1).astype(np.float32)


class ICEMMPCPolicy(CEMMPCPolicy):
    """iCEM adaptation with colored noise, memory, decay, and best-action execution."""

    def __init__(self, world_model, behavior_policy, config: Dict[str, Any], seed: int = 42):
        self.momentum = float(config["momentum"])
        self.population_decay = float(config["population_decay"])
        self.reuse_fraction = float(config["reuse_fraction"])
        self.noise_beta = float(config["noise_beta"])
        self._previous_mean = None
        self._previous_elites = None
        self.last_diagnostics: Dict[str, Any] = {}
        super().__init__(world_model, behavior_policy, config, seed)

    def reset(self, seed: int) -> None:
        super().reset(seed)
        self._previous_mean = None
        self._previous_elites = None
        self.last_diagnostics = {}

    def _sample(self, mean: np.ndarray, std: np.ndarray, population: int) -> np.ndarray:
        noise = powerlaw_noise(
            self.rng, population, self.horizon, len(self.action_low), self.noise_beta
        )
        return np.clip(
            mean[None, :, :] + std[None, :, :] * noise,
            self.action_low,
            self.action_high,
        ).astype(np.float32)

    def act(self, observation: np.ndarray) -> np.ndarray:
        if self._previous_mean is None:
            midpoint = (self.action_high + self.action_low) / 2.0
            mean = np.repeat(midpoint[None, :], self.horizon, axis=0)
        else:
            mean = self._previous_mean.copy()
        initial_std = (
            (self.action_high - self.action_low) / 2.0 * self.initial_std
        )
        std = np.broadcast_to(initial_std, mean.shape).copy()
        previous_iteration_elites = None
        previous_iteration_scores = None
        final_elites = None
        final_scores = None
        population = self.population
        population_schedule = []
        model_evaluations = 0
        best_score_by_iteration = []

        for iteration in range(self.iterations):
            if iteration > 0:
                population = max(
                    2 * self.elites,
                    int(population / self.population_decay),
                )
            population_schedule.append(population)
            sequences = self._sample(mean, std, population)
            if iteration == self.iterations - 1:
                sequences[0] = mean
            scores = self._score_sequences(observation, sequences)
            model_evaluations += population * self.horizon

            if iteration == 0 and self._previous_elites is not None:
                reuse_count = int(
                    len(self._previous_elites) * self.reuse_fraction
                )
                if reuse_count:
                    shifted = self._previous_elites[:reuse_count, 1:].copy()
                    tail = self._sample(mean, std, reuse_count)[:, -1:]
                    shifted = np.concatenate((shifted, tail), axis=1)
                    shifted_scores = self._score_sequences(observation, shifted)
                    model_evaluations += reuse_count * self.horizon
                    sequences = np.concatenate((sequences, shifted), axis=0)
                    scores = np.concatenate((scores, shifted_scores), axis=0)
            elif iteration > 0 and previous_iteration_elites is not None:
                reuse_count = int(
                    len(previous_iteration_elites) * self.reuse_fraction
                )
                if reuse_count:
                    sequences = np.concatenate(
                        (sequences, previous_iteration_elites[:reuse_count]), axis=0
                    )
                    scores = np.concatenate(
                        (scores, previous_iteration_scores[:reuse_count]), axis=0
                    )

            elite_indices = np.argpartition(scores, -self.elites)[-self.elites :]
            elite_indices = elite_indices[np.argsort(scores[elite_indices])[::-1]]
            final_elites = sequences[elite_indices]
            final_scores = scores[elite_indices]
            new_mean = final_elites.mean(axis=0)
            new_std = final_elites.std(axis=0)
            mean = (1.0 - self.momentum) * new_mean + self.momentum * mean
            std = np.maximum(
                (1.0 - self.momentum) * new_std + self.momentum * std,
                self.min_std,
            )
            previous_iteration_elites = final_elites
            previous_iteration_scores = final_scores
            best_score_by_iteration.append(float(final_scores[0]))

        if final_elites is None or final_scores is None:
            raise RuntimeError("iCEM produced no candidates")
        self._previous_mean = np.concatenate(
            (mean[1:], mean[-1:]), axis=0
        ).astype(np.float32)
        self._previous_elites = final_elites.astype(np.float32)
        best_index = int(np.argmax(final_scores))
        best_sequence = final_elites[best_index]
        self.last_diagnostics = {
            "population_schedule": population_schedule,
            "model_evaluations": model_evaluations,
            "best_score": float(final_scores[best_index]),
            "best_sequence": best_sequence.tolist(),
            "best_score_by_iteration": best_score_by_iteration,
            "elite_score_mean": float(np.mean(final_scores)),
            "elite_score_std": float(np.std(final_scores)),
            "action_clipped_fraction": float(
                np.mean(
                    (best_sequence <= self.action_low)
                    | (best_sequence >= self.action_high)
                )
            ),
        }
        return np.clip(
            best_sequence[0], self.action_low, self.action_high
        ).astype(np.float32)
