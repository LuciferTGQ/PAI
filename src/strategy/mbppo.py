from __future__ import annotations

import copy
import csv
import time
from pathlib import Path
from typing import Any, Dict

import numpy as np
import torch
from torch import nn
from torch.distributions import kl_divergence

from src.data.ib_dataset import load_ib_npz
from src.strategy.official_bc import OfficialBCPolicy, OfficialGaussianActor
from src.utils.seed import seed_everything
from src.world_model.interface import FrozenWorldModel


NEORL_PAPER = "https://arxiv.org/abs/2102.00714"
PPO_PAPER = "https://arxiv.org/abs/1707.06347"


class ValueNetwork(nn.Module):
    """NeoRL Appendix-D 2x256 value MLP with fixed data normalization."""

    def __init__(
        self,
        obs_dim: int,
        hidden_size: int,
        hidden_layers: int,
        obs_mean: np.ndarray,
        obs_std: np.ndarray,
    ) -> None:
        super().__init__()
        self.register_buffer("obs_mean", torch.as_tensor(obs_mean, dtype=torch.float32))
        self.register_buffer("obs_std", torch.as_tensor(obs_std, dtype=torch.float32))
        layers: list[nn.Module] = []
        for layer_index in range(hidden_layers):
            layers.extend(
                [
                    nn.Linear(obs_dim if layer_index == 0 else hidden_size, hidden_size),
                    nn.Tanh(),
                ]
            )
        layers.append(nn.Linear(hidden_size, 1))
        self.network = nn.Sequential(*layers)

    def forward(self, observations: torch.Tensor) -> torch.Tensor:
        normalized = (observations - self.obs_mean) / self.obs_std
        return self.network(normalized).squeeze(-1)


class MBPPOPolicy:
    """Deterministic deployment wrapper for a tanh-Gaussian MB-PPO actor."""

    def __init__(self, actor: OfficialGaussianActor, device: str = "cuda") -> None:
        if device.startswith("cuda") and not torch.cuda.is_available():
            device = "cpu"
        self.device = torch.device(device)
        self.actor = actor.to(self.device).eval()

    @classmethod
    def from_checkpoint(
        cls, checkpoint_path: str | Path, device: str = "cuda"
    ) -> "MBPPOPolicy":
        checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
        actor = OfficialGaussianActor(**checkpoint["actor_config"])
        actor.load_state_dict(checkpoint["actor_state"])
        return cls(actor, device=device)

    @torch.inference_mode()
    def act(self, observations: np.ndarray) -> np.ndarray:
        array = np.asarray(observations, dtype=np.float32)
        single = array.ndim == 1
        if single:
            array = array[None, :]
        tensor = torch.as_tensor(array, device=self.device)
        actions = torch.tanh(self.actor(tensor).mean).cpu().numpy()
        return actions[0] if single else actions


def _mean_and_safe_std(
    values: np.ndarray, chunk_size: int = 65_536
) -> tuple[np.ndarray, np.ndarray]:
    """Compute fixed normalization without a dataset-sized float64 copy."""
    feature_sum = np.zeros(values.shape[1], dtype=np.float64)
    feature_square_sum = np.zeros(values.shape[1], dtype=np.float64)
    for start in range(0, len(values), chunk_size):
        chunk = values[start : start + chunk_size].astype(np.float64)
        feature_sum += chunk.sum(axis=0)
        feature_square_sum += np.square(chunk).sum(axis=0)
    mean = feature_sum / len(values)
    variance = np.maximum(feature_square_sum / len(values) - np.square(mean), 0.0)
    std = np.sqrt(variance)
    return (
        mean.astype(np.float32),
        np.where(std < 1e-6, 1.0, std).astype(np.float32),
    )


def _collect_rollout(
    actor: OfficialGaussianActor,
    value: ValueNetwork,
    world_model: FrozenWorldModel,
    initial_observations: torch.Tensor,
    horizon: int,
    gamma: float,
    gae_lambda: float,
) -> Dict[str, torch.Tensor]:
    observations: list[torch.Tensor] = []
    latent_actions: list[torch.Tensor] = []
    old_log_probs: list[torch.Tensor] = []
    rewards: list[torch.Tensor] = []
    values: list[torch.Tensor] = []
    next_values: list[torch.Tensor] = []
    current = initial_observations

    actor.eval()
    value.eval()
    with torch.inference_mode():
        for _ in range(horizon):
            distribution = actor(current)
            latent_action = distribution.sample()
            action = torch.tanh(latent_action)
            next_frame = world_model.predict_next_frame_tensor(current, action)
            reward = -(3.0 * next_frame[:, 4] + next_frame[:, 5])
            next_observation = world_model.update_history_tensor(current, next_frame)

            observations.append(current)
            latent_actions.append(latent_action)
            old_log_probs.append(distribution.log_prob(latent_action).sum(dim=-1))
            rewards.append(reward)
            values.append(value(current))
            next_values.append(value(next_observation))
            current = next_observation

    reward_tensor = torch.stack(rewards)
    value_tensor = torch.stack(values)
    next_value_tensor = torch.stack(next_values)
    advantages = torch.zeros_like(reward_tensor)
    running_advantage = torch.zeros_like(reward_tensor[0])
    for step in reversed(range(horizon)):
        delta = reward_tensor[step] + gamma * next_value_tensor[step] - value_tensor[step]
        running_advantage = delta + gamma * gae_lambda * running_advantage
        advantages[step] = running_advantage
    returns = advantages + value_tensor

    def flatten(items: list[torch.Tensor] | torch.Tensor) -> torch.Tensor:
        tensor = torch.stack(items) if isinstance(items, list) else items
        return tensor.flatten(0, 1)

    return {
        "observations": flatten(observations),
        "latent_actions": flatten(latent_actions),
        "old_log_probs": flatten(old_log_probs),
        "advantages": flatten(advantages),
        "returns": flatten(returns),
        "rewards": flatten(reward_tensor),
        "final_observations": current,
    }


def _write_history(path: Path, rows: list[Dict[str, float | int]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def train_mbppo_variant(
    config: Dict[str, Any], root: Path, use_behavior_kl: bool
) -> Path:
    """Train one MB-PPO variant; the supplied world model is never optimized."""
    seed = int(config["seed"])
    seed_everything(seed)
    device_name = str(config["policies"]["device"])
    if device_name.startswith("cuda") and not torch.cuda.is_available():
        device_name = "cpu"
    device = torch.device(device_name)
    variant = "with_kl" if use_behavior_kl else "without_kl"
    variant_outputs = config["outputs"][variant]
    final_checkpoint = root / variant_outputs["final_checkpoint"]
    if final_checkpoint.exists():
        print(f"reuse completed MB-PPO checkpoint: {final_checkpoint}")
        return final_checkpoint

    train_data = load_ib_npz(root / config["data"]["train_path"])
    train_observations = np.ascontiguousarray(train_data["obs"], dtype=np.float32)
    obs_mean, obs_std = _mean_and_safe_std(train_observations)

    bc_checkpoint = root / config["policies"]["official_bc_checkpoint"]
    behavior = OfficialBCPolicy.from_checkpoint(bc_checkpoint, device=device_name)
    bc_payload = torch.load(bc_checkpoint, map_location=device, weights_only=False)
    actor_config = dict(bc_payload["model_config"])
    actor = OfficialGaussianActor(**actor_config).to(device)
    actor.load_state_dict(copy.deepcopy(behavior.actor.state_dict()))
    actor.train()
    value_config = config["model"]
    value = ValueNetwork(
        obs_dim=int(actor_config["obs_dim"]),
        hidden_size=int(value_config["hidden_size"]),
        hidden_layers=int(value_config["hidden_layers"]),
        obs_mean=obs_mean,
        obs_std=obs_std,
    ).to(device)

    world_model_checkpoint = root / config["policies"]["world_model_checkpoint"]
    world_model = FrozenWorldModel(world_model_checkpoint, device=device_name)
    if any(parameter.requires_grad for parameter in world_model.model.parameters()):
        raise RuntimeError("Frozen World Model unexpectedly has trainable parameters")

    training = config["training"]
    learning_rate = float(training["learning_rate"])
    actor_optimizer = torch.optim.Adam(actor.parameters(), lr=learning_rate)
    value_optimizer = torch.optim.Adam(value.parameters(), lr=learning_rate)
    gradient_steps = int(training["gradient_steps"])
    rollout_horizon = int(training["rollout_horizon"])
    start_states = int(training["start_states_per_collection"])
    batch_size = int(training["batch_size"])
    ppo_epochs = int(training["ppo_epochs_per_collection"])
    gamma = float(training["gamma"])
    gae_lambda = float(training["gae_lambda"])
    clip_ratio = float(training["clip_ratio"])
    entropy_coefficient = float(training["entropy_coefficient"])
    kl_coefficient = float(training["behavior_kl_coefficient"]) if use_behavior_kl else 0.0
    max_grad_norm = float(training["max_grad_norm"])
    checkpoint_interval = int(training["checkpoint_interval"])
    log_interval = int(training["log_interval"])
    rng = np.random.default_rng(seed)
    history: list[Dict[str, float | int]] = []
    step = 0
    previous_elapsed = 0.0
    checkpoint_dir = root / variant_outputs["checkpoint_dir"]
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    resumable = sorted(checkpoint_dir.glob("step_*.pt"))
    if resumable:
        resume_path = resumable[-1]
        payload = torch.load(resume_path, map_location=device, weights_only=False)
        if bool(payload["use_behavior_kl"]) != use_behavior_kl:
            raise ValueError(f"MB-PPO resume variant mismatch: {resume_path}")
        if Path(payload["world_model_checkpoint"]).resolve() != world_model_checkpoint.resolve():
            raise ValueError(f"MB-PPO resume World Model mismatch: {resume_path}")
        actor.load_state_dict(payload["actor_state"])
        value.load_state_dict(payload["value_state"])
        actor_optimizer.load_state_dict(payload["actor_optimizer_state"])
        value_optimizer.load_state_dict(payload["value_optimizer_state"])
        step = int(payload["gradient_step"])
        history = list(payload.get("history", []))
        previous_elapsed = float(payload.get("elapsed_time_seconds", 0.0))
        if "numpy_rng_state" in payload:
            rng.bit_generator.state = payload["numpy_rng_state"]
        if "torch_rng_state" in payload:
            torch.set_rng_state(payload["torch_rng_state"].cpu())
        if device.type == "cuda" and payload.get("cuda_rng_state_all") is not None:
            torch.cuda.set_rng_state_all(
                [state.cpu() for state in payload["cuda_rng_state_all"]]
            )
        print(f"resume MB-PPO variant={variant} step={step} checkpoint={resume_path}")
    training_started = time.perf_counter()

    while step < gradient_steps:
        initial_indices = rng.integers(0, len(train_observations), size=start_states)
        initial_observations = torch.as_tensor(
            train_observations[initial_indices], device=device
        )
        rollout = _collect_rollout(
            actor,
            value,
            world_model,
            initial_observations,
            rollout_horizon,
            gamma,
            gae_lambda,
        )
        advantages = rollout["advantages"]
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
        sample_count = len(advantages)

        for _ in range(ppo_epochs):
            order = rng.permutation(sample_count)
            for start in range(0, sample_count, batch_size):
                if step >= gradient_steps:
                    break
                indices = torch.as_tensor(order[start : start + batch_size], device=device)
                observations = rollout["observations"][indices]
                latent_actions = rollout["latent_actions"][indices]
                old_log_probs = rollout["old_log_probs"][indices]
                batch_advantages = advantages[indices]
                batch_returns = rollout["returns"][indices]

                actor.train()
                distribution = actor(observations)
                log_probs = distribution.log_prob(latent_actions).sum(dim=-1)
                ratio = torch.exp(log_probs - old_log_probs)
                unclipped = ratio * batch_advantages
                clipped = torch.clamp(ratio, 1.0 - clip_ratio, 1.0 + clip_ratio)
                policy_loss = -torch.minimum(unclipped, clipped * batch_advantages).mean()
                entropy = distribution.entropy().sum(dim=-1).mean()
                with torch.no_grad():
                    behavior_distribution = behavior.actor(observations)
                behavior_kl = kl_divergence(behavior_distribution, distribution).sum(dim=-1).mean()
                actor_loss = (
                    policy_loss
                    + kl_coefficient * behavior_kl
                    - entropy_coefficient * entropy
                )

                actor_optimizer.zero_grad(set_to_none=True)
                actor_loss.backward()
                actor_grad_norm = torch.nn.utils.clip_grad_norm_(
                    actor.parameters(), max_grad_norm
                )
                actor_optimizer.step()

                value.train()
                value_loss = 0.5 * (value(observations) - batch_returns).square().mean()
                value_optimizer.zero_grad(set_to_none=True)
                value_loss.backward()
                value_grad_norm = torch.nn.utils.clip_grad_norm_(
                    value.parameters(), max_grad_norm
                )
                value_optimizer.step()

                step += 1
                if not torch.isfinite(actor_loss + value_loss):
                    raise FloatingPointError(f"non-finite MB-PPO loss at step {step}")
                if step == 1 or step % log_interval == 0 or step == gradient_steps:
                    row: Dict[str, float | int] = {
                        "gradient_step": step,
                        "elapsed_time_seconds": previous_elapsed
                        + time.perf_counter()
                        - training_started,
                        "model_reward_mean": float(rollout["rewards"].mean().item()),
                        "model_reward_min": float(rollout["rewards"].min().item()),
                        "model_state_abs_max": float(
                            rollout["final_observations"].abs().max().item()
                        ),
                        "policy_loss": float(policy_loss.item()),
                        "value_loss": float(value_loss.item()),
                        "behavior_kl": float(behavior_kl.item()),
                        "entropy": float(entropy.item()),
                        "approx_old_policy_kl": float((old_log_probs - log_probs).mean().item()),
                        "clip_fraction": float(
                            ((ratio - 1.0).abs() > clip_ratio).float().mean().item()
                        ),
                        "actor_grad_norm": float(actor_grad_norm.item()),
                        "value_grad_norm": float(value_grad_norm.item()),
                    }
                    history.append(row)
                    print(
                        f"variant={variant} step={step:04d} "
                        f"reward={row['model_reward_mean']:.3f} "
                        f"behavior_kl={row['behavior_kl']:.5f} "
                        f"policy_loss={row['policy_loss']:.5f}"
                    )

                if step % checkpoint_interval == 0 or step == gradient_steps:
                    checkpoint_path = checkpoint_dir / f"step_{step:04d}.pt"
                    payload = {
                        "actor_state": actor.state_dict(),
                        "actor_config": actor_config,
                        "value_state": value.state_dict(),
                        "value_config": {
                            "obs_dim": int(actor_config["obs_dim"]),
                            "hidden_size": int(value_config["hidden_size"]),
                            "hidden_layers": int(value_config["hidden_layers"]),
                        },
                        "actor_optimizer_state": actor_optimizer.state_dict(),
                        "value_optimizer_state": value_optimizer.state_dict(),
                        "gradient_step": step,
                        "use_behavior_kl": use_behavior_kl,
                        "behavior_kl_coefficient": kl_coefficient,
                        "history": history,
                        "numpy_rng_state": rng.bit_generator.state,
                        "torch_rng_state": torch.get_rng_state(),
                        "cuda_rng_state_all": (
                            torch.cuda.get_rng_state_all() if device.type == "cuda" else None
                        ),
                        "elapsed_time_seconds": previous_elapsed
                        + time.perf_counter()
                        - training_started,
                        "world_model_checkpoint": str(world_model_checkpoint),
                        "official_bc_checkpoint": str(bc_checkpoint),
                        "source": {
                            "neorl_paper": NEORL_PAPER,
                            "ppo_paper": PPO_PAPER,
                            "neorl_mbppo_official_code_in_fixed_offlinerl_commit": False,
                        },
                        "config": config,
                    }
                    torch.save(payload, checkpoint_path)
                    _write_history(root / variant_outputs["training_history_csv"], history)
                    if step == gradient_steps:
                        final_checkpoint.parent.mkdir(parents=True, exist_ok=True)
                        torch.save(payload, final_checkpoint)

    history_path = root / variant_outputs["training_history_csv"]
    _write_history(history_path, history)
    return final_checkpoint
