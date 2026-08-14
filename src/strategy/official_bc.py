from __future__ import annotations

import copy
import csv
from pathlib import Path
from typing import Dict, Iterable

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F
from torch import nn

from src.data.ib_dataset import load_ib_npz
from src.utils.seed import seed_everything


OFFLINERL_COMMIT = "807933a87f77529f17bd81ac64d717aad89f5cdf"


def soft_clamp(
    value: torch.Tensor,
    minimum: torch.Tensor | None = None,
    maximum: torch.Tensor | None = None,
) -> torch.Tensor:
    """Gradient-preserving clamp copied semantically from fixed OfflineRL."""
    if maximum is not None:
        value = maximum - F.softplus(maximum - value)
    if minimum is not None:
        value = minimum + F.softplus(value - minimum)
    return value


class OfficialGaussianActor(nn.Module):
    """Architecture-compatible implementation of OfflineRL GaussianActor."""

    def __init__(
        self,
        obs_dim: int = 180,
        action_dim: int = 3,
        hidden_size: int = 256,
        hidden_layers: int = 2,
    ) -> None:
        super().__init__()
        layers: list[nn.Module] = []
        for layer_index in range(hidden_layers):
            layers.append(
                nn.Linear(obs_dim if layer_index == 0 else hidden_size, hidden_size)
            )
            layers.append(nn.LeakyReLU(negative_slope=0.1, inplace=True))
        layers.append(nn.Linear(hidden_size, 2 * action_dim))
        layers.append(nn.Identity())
        self.backbone = nn.Sequential(*layers)
        self.max_logstd = nn.Parameter(torch.ones(action_dim))
        self.min_logstd = nn.Parameter(torch.ones(action_dim) * -10.0)

    def forward(self, observations: torch.Tensor) -> torch.distributions.Normal:
        mean, log_std = torch.chunk(self.backbone(observations), 2, dim=-1)
        log_std = soft_clamp(log_std, self.min_logstd, self.max_logstd)
        return torch.distributions.Normal(mean, torch.exp(log_std))

    def policy_infer(self, observations: torch.Tensor) -> torch.Tensor:
        return self(observations).mean


class OfficialBCPolicy:
    """Deterministic mean-action inference for a trained official BC actor."""

    def __init__(self, actor: OfficialGaussianActor, device: str = "cuda") -> None:
        if device.startswith("cuda") and not torch.cuda.is_available():
            device = "cpu"
        self.device = torch.device(device)
        self.actor = actor.to(self.device).eval()
        for parameter in self.actor.parameters():
            parameter.requires_grad_(False)

    @classmethod
    def from_checkpoint(
        cls, checkpoint_path: str | Path, device: str = "cuda"
    ) -> "OfficialBCPolicy":
        if device.startswith("cuda") and not torch.cuda.is_available():
            device = "cpu"
        checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
        actor = OfficialGaussianActor(**checkpoint["model_config"])
        actor.load_state_dict(checkpoint["actor_state"])
        return cls(actor, device=device)

    @torch.inference_mode()
    def act(self, observations: np.ndarray) -> np.ndarray:
        array = np.asarray(observations, dtype=np.float32)
        single = array.ndim == 1
        if single:
            array = array[None, :]
        tensor = torch.as_tensor(array, device=self.device)
        actions = self.actor.policy_infer(tensor).cpu().numpy()
        return actions[0] if single else actions


def _official_validation_loss(
    actor: OfficialGaussianActor,
    observations: torch.Tensor,
    actions: torch.Tensor,
    batch_size: int,
    device: torch.device,
) -> tuple[float, float]:
    """Return the official summed batch-MSE criterion and sample-weighted MSE."""
    official_sum = 0.0
    squared_error_sum = 0.0
    scalar_count = 0
    actor.eval()
    with torch.inference_mode():
        for start in range(0, len(observations), batch_size):
            stop = start + batch_size
            obs_batch = observations[start:stop].to(device)
            action_batch = actions[start:stop].to(device)
            prediction = actor(obs_batch).mean
            squared_error = (prediction - action_batch).square()
            official_sum += squared_error.mean().item()
            squared_error_sum += squared_error.sum().item()
            scalar_count += squared_error.numel()
    return official_sum, squared_error_sum / scalar_count


def _write_history(history: Iterable[Dict[str, float]], path: str | Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    rows = list(history)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)


def _plot_history(history: list[Dict[str, float]], path: str | Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    epochs = [row["epoch"] for row in history]
    figure, axes = plt.subplots(1, 2, figsize=(10, 4))
    axes[0].plot(epochs, [row["train_nll"] for row in history])
    axes[0].set(title="Official BC training", xlabel="Epoch", ylabel="Gaussian NLL")
    axes[0].set_yscale("symlog", linthresh=0.1)
    axes[1].plot(epochs, [row["val_mse"] for row in history])
    axes[1].set(title="Validation model selection", xlabel="Epoch", ylabel="Action MSE")
    axes[1].set_yscale("log")
    best_row = min(history, key=lambda row: row["val_official_sum_batch_mse"])
    for axis in axes:
        axis.axvline(best_row["epoch"], color="tab:green", linestyle="--", alpha=0.8)
        axis.grid(alpha=0.25)
    figure.tight_layout()
    figure.savefig(path, dpi=160)
    plt.close(figure)


def train_official_bc(config: Dict[str, object]) -> Path:
    """Train with the exact fixed-commit BC update and selection semantics."""
    seed_everything(int(config["seed"]))
    data_config = config["data"]
    model_config = config["model"]
    training_config = config["training"]
    output_config = config["outputs"]

    train_data = load_ib_npz(data_config["train_path"])
    val_data = load_ib_npz(data_config["val_path"])
    train_obs = torch.from_numpy(np.ascontiguousarray(train_data["obs"], dtype=np.float32))
    train_actions = torch.from_numpy(
        np.ascontiguousarray(train_data["action"], dtype=np.float32)
    )
    val_obs = torch.from_numpy(np.ascontiguousarray(val_data["obs"], dtype=np.float32))
    val_actions = torch.from_numpy(np.ascontiguousarray(val_data["action"], dtype=np.float32))

    device_name = str(training_config["device"])
    if device_name.startswith("cuda") and not torch.cuda.is_available():
        device_name = "cpu"
    device = torch.device(device_name)
    train_obs = train_obs.to(device)
    train_actions = train_actions.to(device)
    val_obs = val_obs.to(device)
    val_actions = val_actions.to(device)
    actor_kwargs = {
        "obs_dim": int(data_config["obs_dim"]),
        "action_dim": int(data_config["action_dim"]),
        "hidden_size": int(model_config["actor_features"]),
        "hidden_layers": int(model_config["actor_layers"]),
    }
    actor = OfficialGaussianActor(**actor_kwargs).to(device)
    optimizer = torch.optim.Adam(actor.parameters(), lr=float(training_config["actor_lr"]))
    batch_size = int(training_config["batch_size"])
    steps_per_epoch = int(training_config["steps_per_epoch"])
    max_epoch = int(training_config["max_epoch"])
    log_interval = int(training_config.get("log_interval", 1))

    best_actor_state = copy.deepcopy(actor.state_dict())
    best_official_loss = float("inf")
    best_epoch = 0
    history: list[Dict[str, float]] = []
    generator = torch.Generator().manual_seed(int(config["seed"]))

    for epoch_index in range(max_epoch):
        actor.train()
        nll_sum = torch.zeros((), device=device)
        for _ in range(steps_per_epoch):
            indices = torch.randint(
                len(train_obs), (batch_size,), generator=generator
            ).to(device)
            obs_batch = train_obs[indices]
            action_batch = train_actions[indices]
            loss = -actor(obs_batch).log_prob(action_batch).mean()
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
            nll_sum.add_(loss.detach())

        official_loss, val_mse = _official_validation_loss(
            actor, val_obs, val_actions, batch_size, device
        )
        if official_loss < best_official_loss:
            best_official_loss = official_loss
            best_actor_state = copy.deepcopy(actor.state_dict())
            best_epoch = epoch_index + 1
        row = {
            "epoch": epoch_index + 1,
            "train_nll": nll_sum.item() / steps_per_epoch,
            "val_official_sum_batch_mse": official_loss,
            "val_mse": val_mse,
        }
        history.append(row)
        if epoch_index == 0 or (epoch_index + 1) % log_interval == 0:
            print(
                f"epoch={epoch_index + 1:03d} train_nll={row['train_nll']:.6f} "
                f"val_mse={val_mse:.8f} best_epoch={best_epoch}"
            )

    checkpoint_path = Path(output_config["checkpoint"])
    checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "actor_state": best_actor_state,
            "model_config": actor_kwargs,
            "best_epoch": best_epoch,
            "best_official_validation_loss": best_official_loss,
            "seed": int(config["seed"]),
            "offlinerl_commit": OFFLINERL_COMMIT,
            "source_config": config,
        },
        checkpoint_path,
    )
    _write_history(history, output_config["training_history_csv"])
    _plot_history(history, output_config["training_curve"])
    print(f"saved={checkpoint_path} best_epoch={best_epoch}")
    return checkpoint_path
