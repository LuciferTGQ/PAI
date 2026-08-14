import copy

import numpy as np
import torch
from torch.distributions import kl_divergence

from src.strategy.mbppo import MBPPOPolicy, ValueNetwork, _collect_rollout
from src.strategy.official_bc import OfficialGaussianActor


class _TensorWorldModel:
    def predict_next_frame_tensor(self, histories, actions):
        frames = histories[:, :6].clone()
        frames[:, 4] = 10.0 + actions[:, 0]
        frames[:, 5] = 20.0 + actions[:, 1]
        return frames

    def update_history_tensor(self, histories, next_frames):
        return torch.cat((next_frames, histories[:, :-6]), dim=1)


def test_mbppo_policy_is_tanh_bounded() -> None:
    actor = OfficialGaussianActor(hidden_size=16, hidden_layers=1)
    policy = MBPPOPolicy(actor, device="cpu")
    actions = policy.act(np.zeros((5, 180), dtype=np.float32))
    assert actions.shape == (5, 3)
    assert np.all(actions >= -1.0) and np.all(actions <= 1.0)


def test_mbppo_rollout_shapes_and_behavior_kl_identity() -> None:
    actor = OfficialGaussianActor(hidden_size=16, hidden_layers=1)
    behavior = copy.deepcopy(actor)
    value = ValueNetwork(
        obs_dim=180,
        hidden_size=16,
        hidden_layers=1,
        obs_mean=np.zeros(180, dtype=np.float32),
        obs_std=np.ones(180, dtype=np.float32),
    )
    observations = torch.zeros(4, 180)
    rollout = _collect_rollout(
        actor,
        value,
        _TensorWorldModel(),
        observations,
        horizon=3,
        gamma=0.99,
        gae_lambda=0.95,
    )
    assert rollout["observations"].shape == (12, 180)
    assert rollout["latent_actions"].shape == (12, 3)
    assert rollout["advantages"].shape == (12,)
    assert rollout["returns"].shape == (12,)
    distribution = actor(observations)
    behavior_distribution = behavior(observations)
    behavior_kl = kl_divergence(behavior_distribution, distribution).sum(dim=-1)
    assert torch.allclose(behavior_kl, torch.zeros_like(behavior_kl), atol=1e-7)
