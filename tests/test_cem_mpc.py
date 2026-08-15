import numpy as np

from src.evaluation.cem_horizon import select_horizon_one_standard_error
from src.strategy.cem_mpc import CEMMPCPolicy, ib_reward_from_frame
from src.strategy.icem_mpc import ICEMMPCPolicy, powerlaw_noise


class _FakeWorldModel:
    def predict_next_frame(self, histories, actions):
        histories = np.asarray(histories)
        actions = np.asarray(actions)
        frames = histories[:, :6].copy()
        frames[:, 4] = 10.0 + actions[:, 0]
        frames[:, 5] = 20.0 + actions[:, 1]
        return frames


class _ZeroPolicy:
    def act(self, observation):
        return np.zeros(3, dtype=np.float32)


def test_ib_reward_formula() -> None:
    frame = np.array([70, 50, 50, 50, 4, 5], dtype=np.float32)
    assert ib_reward_from_frame(frame) == -17.0


def test_cem_action_is_bounded_and_improves_fake_cost() -> None:
    config = {
        "horizon": 2,
        "population": 256,
        "elites": 16,
        "iterations": 3,
        "initial_std": 0.7,
        "min_std": 0.01,
        "action_low": [-1.0, -1.0, -1.0],
        "action_high": [1.0, 1.0, 1.0],
    }
    policy = CEMMPCPolicy(_FakeWorldModel(), _ZeroPolicy(), config, seed=1)
    action = policy.act(np.zeros(180, dtype=np.float32))
    assert np.all(action >= -1.0) and np.all(action <= 1.0)
    assert action[0] < -0.5
    assert action[1] < -0.5
    assert policy.last_diagnostics["model_evaluations"] == 256 * 3 * 2
    assert np.isfinite(policy.last_diagnostics["predicted_best_return"])


def test_one_standard_error_horizon_selection_prefers_shorter_candidate() -> None:
    selection = select_horizon_one_standard_error(
        {
            5: [-100.5, -98.5, -99.5],
            10: [-100.0, -98.0, -99.0],
            20: [-103.0, -101.0, -102.0],
        }
    )
    assert selection["best_mean_horizon"] == 10
    assert selection["eligible_horizons"] == [5, 10]
    assert selection["selected_horizon"] == 5


def test_powerlaw_noise_shape_and_scale() -> None:
    noise = powerlaw_noise(np.random.default_rng(3), 16, 10, 3, beta=2.0)
    assert noise.shape == (16, 10, 3)
    assert np.isfinite(noise).all()
    assert 0.2 < float(noise.std()) < 2.0


def test_icem_action_is_bounded() -> None:
    base = {
        "horizon": 3,
        "population": 32,
        "elites": 4,
        "iterations": 2,
        "initial_std": 0.5,
        "min_std": 0.01,
        "action_low": [-1.0, -1.0, -1.0],
        "action_high": [1.0, 1.0, 1.0],
    }
    icem = ICEMMPCPolicy(
        _FakeWorldModel(),
        _ZeroPolicy(),
        {
            **base,
            "momentum": 0.1,
            "population_decay": 1.25,
            "reuse_fraction": 0.3,
            "noise_beta": 2.0,
        },
        seed=5,
    )
    action = icem.act(np.zeros(180, dtype=np.float32))
    assert np.all(action >= -1.0) and np.all(action <= 1.0)
