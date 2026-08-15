import colorednoise
import numpy as np

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


def test_colored_noise_is_exact_official_dependency_call() -> None:
    actual = powerlaw_noise(
        np.random.default_rng(17),
        samples=7,
        horizon=11,
        action_dim=3,
        beta=2.0,
    )
    expected = colorednoise.powerlaw_psd_gaussian(
        2.0,
        size=(7, 3, 11),
        random_state=np.random.default_rng(17),
    ).transpose(0, 2, 1).astype(np.float32)
    assert np.array_equal(actual, expected)


def test_icem_optimizes_deterministic_toy_objective() -> None:
    target = np.full((8, 1), 0.65, dtype=np.float32)
    config = {
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
    policy = _ToyICEM(target, config, seed=23)
    action = policy.act(np.zeros(1, dtype=np.float32))
    best_sequence = np.asarray(policy.last_diagnostics["best_sequence"])
    baseline_mse = float(np.mean(np.square(target)))
    optimized_mse = float(np.mean(np.square(best_sequence - target)))

    assert policy.last_diagnostics["population_schedule"] == [256, 204, 163, 130]
    assert optimized_mse < 0.02
    assert optimized_mse < 0.1 * baseline_mse
    assert abs(float(action[0]) - 0.65) < 0.2
