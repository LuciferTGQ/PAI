import numpy as np

from src.strategy.cem_mpc import CEMMPCPolicy, ib_reward_from_frame


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
