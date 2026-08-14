import numpy as np
import torch

from src.strategy.official_bc import OfficialBCPolicy, OfficialGaussianActor, soft_clamp


def test_official_actor_shapes_and_bounds() -> None:
    actor = OfficialGaussianActor()
    distribution = actor(torch.zeros(4, 180))
    assert distribution.mean.shape == (4, 3)
    assert distribution.stddev.shape == (4, 3)
    assert torch.all(distribution.stddev > 0)


def test_soft_clamp_retains_gradient() -> None:
    value = torch.tensor([100.0, -100.0], requires_grad=True)
    result = soft_clamp(value, torch.tensor(-10.0), torch.tensor(1.0))
    result.sum().backward()
    assert value.grad is not None
    assert torch.all(torch.isfinite(value.grad))


def test_policy_returns_unbounded_official_mean() -> None:
    actor = OfficialGaussianActor()
    with torch.no_grad():
        actor.backbone[-2].weight.zero_()
        actor.backbone[-2].bias[:3].fill_(2.0)
    policy = OfficialBCPolicy(actor, device="cpu")
    action = policy.act(np.zeros(180, dtype=np.float32))
    np.testing.assert_allclose(action, np.full(3, 2.0, dtype=np.float32))
