from __future__ import annotations

from typing import Any, Dict

import numpy as np

from src.strategy.cem_mpc import CEMMPCPolicy


def powerlaw_noise(
    rng: np.random.Generator,
    samples: int,
    horizon: int,
    action_dim: int,
    beta: float,
) -> np.ndarray:
    """FFT power-law noise with unit variance along the temporal dimension."""
    white = rng.standard_normal((samples, action_dim, horizon))
    if beta <= 0:
        return white.transpose(0, 2, 1).astype(np.float32)
    frequencies = np.fft.rfftfreq(horizon)
    scale = np.zeros_like(frequencies)
    scale[1:] = np.power(frequencies[1:], -beta / 2.0)
    spectrum = np.fft.rfft(white, axis=-1) * scale[None, None, :]
    colored = np.fft.irfft(spectrum, n=horizon, axis=-1)
    colored -= colored.mean(axis=-1, keepdims=True)
    colored /= np.maximum(colored.std(axis=-1, keepdims=True), 1e-6)
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
        super().__init__(world_model, behavior_policy, config, seed)

    def reset(self, seed: int) -> None:
        super().reset(seed)
        self._previous_mean = None
        self._previous_elites = None

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
        baseline_action = np.clip(
            self.behavior_policy.act(observation), self.action_low, self.action_high
        )
        if self._previous_mean is None:
            mean = np.repeat(baseline_action[None, :], self.horizon, axis=0)
        else:
            mean = np.concatenate(
                (self._previous_mean[1:], self._previous_mean[-1:]), axis=0
            )
        std = np.full_like(mean, self.initial_std)
        previous_iteration_elites = None
        previous_iteration_scores = None
        final_elites = None
        final_scores = None

        for iteration in range(self.iterations):
            population = max(
                2 * self.elites,
                int(self.population / (self.population_decay**iteration)),
            )
            sequences = self._sample(mean, std, population)
            if iteration == self.iterations - 1:
                sequences[0] = mean
            scores = self._score_sequences(observation, sequences)

            if iteration == 0 and self._previous_elites is not None:
                reuse_count = max(1, int(len(self._previous_elites) * self.reuse_fraction))
                shifted = self._previous_elites[:reuse_count, 1:].copy()
                tail = self._sample(mean, std, reuse_count)[:, -1:]
                shifted = np.concatenate((shifted, tail), axis=1)
                shifted_scores = self._score_sequences(observation, shifted)
                sequences = np.concatenate((sequences, shifted), axis=0)
                scores = np.concatenate((scores, shifted_scores), axis=0)
            elif iteration > 0 and previous_iteration_elites is not None:
                reuse_count = max(
                    1, int(len(previous_iteration_elites) * self.reuse_fraction)
                )
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

        if final_elites is None or final_scores is None:
            raise RuntimeError("iCEM produced no candidates")
        self._previous_mean = mean.astype(np.float32)
        self._previous_elites = final_elites.astype(np.float32)
        return np.clip(
            final_elites[int(np.argmax(final_scores)), 0], self.action_low, self.action_high
        ).astype(np.float32)
