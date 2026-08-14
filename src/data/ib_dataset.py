from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, Tuple

import numpy as np
import torch
from torch.utils.data import Dataset


REQUIRED_KEYS = ("obs", "next_obs", "action", "reward", "done", "index")


@dataclass(frozen=True)
class NormalizationStats:
    frame_mean: np.ndarray
    frame_std: np.ndarray
    action_mean: np.ndarray
    action_std: np.ndarray
    target_mean: np.ndarray
    target_std: np.ndarray

    def to_dict(self) -> Dict[str, list]:
        return {
            "frame_mean": self.frame_mean.tolist(),
            "frame_std": self.frame_std.tolist(),
            "action_mean": self.action_mean.tolist(),
            "action_std": self.action_std.tolist(),
            "target_mean": self.target_mean.tolist(),
            "target_std": self.target_std.tolist(),
        }

    @classmethod
    def from_dict(cls, values: Dict[str, Iterable[float]]) -> "NormalizationStats":
        return cls(**{key: np.asarray(value, dtype=np.float32) for key, value in values.items()})


def _safe_std(values: np.ndarray, axes: Tuple[int, ...]) -> np.ndarray:
    std = values.astype(np.float64).std(axis=axes)
    return np.where(std < 1e-6, 1.0, std).astype(np.float32)


def load_ib_npz(path: str | Path) -> Dict[str, np.ndarray]:
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"NeoRL dataset not found: {path}")
    with np.load(path) as archive:
        missing = set(REQUIRED_KEYS) - set(archive.files)
        if missing:
            raise ValueError(f"Dataset {path} is missing keys: {sorted(missing)}")
        return {key: np.asarray(archive[key]) for key in REQUIRED_KEYS}


def trajectory_spans(index: np.ndarray, transition_count: int) -> np.ndarray:
    """Return [start, end) spans for raw-file end offsets or API start offsets."""
    offsets = np.asarray(index).reshape(-1).astype(np.int64)
    if offsets.size == 0:
        raise ValueError("Trajectory index is empty")

    if offsets[-1] == transition_count:
        ends = offsets
        starts = np.concatenate(([0], ends[:-1]))
    else:
        starts = offsets
        if starts[0] != 0:
            starts = np.concatenate(([0], starts))
        ends = np.concatenate((starts[1:], [transition_count]))

    spans = np.stack((starts, ends), axis=1)
    if np.any(spans[:, 0] < 0) or np.any(spans[:, 1] > transition_count):
        raise ValueError("Trajectory offsets fall outside the transition array")
    if np.any(spans[:, 1] <= spans[:, 0]):
        raise ValueError("Trajectory spans must have positive length")
    return spans


def validate_ib_semantics(
    data: Dict[str, np.ndarray], history_len: int = 30, frame_dim: int = 6
) -> Dict[str, object]:
    expected_obs_dim = history_len * frame_dim
    n = len(data["obs"])
    if data["obs"].shape != (n, expected_obs_dim):
        raise ValueError(f"Expected obs shape ({n}, {expected_obs_dim}), got {data['obs'].shape}")
    if data["next_obs"].shape != data["obs"].shape:
        raise ValueError("obs and next_obs shapes differ")
    if data["action"].shape != (n, 3):
        raise ValueError(f"Expected action shape ({n}, 3), got {data['action'].shape}")
    if not np.array_equal(data["next_obs"][:, frame_dim:], data["obs"][:, :-frame_dim]):
        raise ValueError("Released data does not obey the verified sliding-window update")

    spans = trajectory_spans(data["index"], n)
    lengths = spans[:, 1] - spans[:, 0]
    reward_expected = -(3.0 * data["next_obs"][:, 4] + data["next_obs"][:, 5])
    reward_error = np.abs(reward_expected - data["reward"].reshape(-1))
    return {
        "transitions": n,
        "trajectories": len(spans),
        "trajectory_lengths": sorted(np.unique(lengths).tolist()),
        "max_reward_abs_error": float(reward_error.max()),
        "action_min": data["action"].min(axis=0).tolist(),
        "action_max": data["action"].max(axis=0).tolist(),
    }


def compute_normalization(
    train_data: Dict[str, np.ndarray], history_len: int = 30, frame_dim: int = 6
) -> NormalizationStats:
    frames = train_data["obs"].reshape(-1, history_len, frame_dim)
    actions = train_data["action"]
    targets = train_data["next_obs"][:, :frame_dim]
    return NormalizationStats(
        frame_mean=frames.astype(np.float64).mean(axis=(0, 1)).astype(np.float32),
        frame_std=_safe_std(frames, (0, 1)),
        action_mean=actions.astype(np.float64).mean(axis=0).astype(np.float32),
        action_std=_safe_std(actions, (0,)),
        target_mean=targets.astype(np.float64).mean(axis=0).astype(np.float32),
        target_std=_safe_std(targets, (0,)),
    )


class IBTransitionDataset(Dataset):
    """Normalized one-step samples with chronological Transformer tokens."""

    def __init__(
        self,
        data: Dict[str, np.ndarray],
        stats: NormalizationStats,
        history_len: int = 30,
        frame_dim: int = 6,
    ) -> None:
        self.obs = torch.from_numpy(np.ascontiguousarray(data["obs"], dtype=np.float32))
        self.actions = torch.from_numpy(np.ascontiguousarray(data["action"], dtype=np.float32))
        self.targets = torch.from_numpy(
            np.ascontiguousarray(data["next_obs"][:, :frame_dim], dtype=np.float32)
        )
        self.history_len = history_len
        self.frame_dim = frame_dim
        self.frame_mean = torch.from_numpy(stats.frame_mean)
        self.frame_std = torch.from_numpy(stats.frame_std)
        self.action_mean = torch.from_numpy(stats.action_mean)
        self.action_std = torch.from_numpy(stats.action_std)
        self.target_mean = torch.from_numpy(stats.target_mean)
        self.target_std = torch.from_numpy(stats.target_std)

    def __len__(self) -> int:
        return len(self.obs)

    def __getitem__(self, index: int) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        latest_first = self.obs[index].reshape(self.history_len, self.frame_dim)
        chronological = torch.flip(latest_first, dims=(0,))
        history = (chronological - self.frame_mean) / self.frame_std
        action = (self.actions[index] - self.action_mean) / self.action_std
        target = (self.targets[index] - self.target_mean) / self.target_std
        return history, action, target

    def __getitems__(
        self, indices: list[int]
    ) -> list[Tuple[torch.Tensor, torch.Tensor, torch.Tensor]]:
        """Vectorized DataLoader fetch with the same per-sample transformations."""
        index_tensor = torch.as_tensor(indices, dtype=torch.long)
        latest_first = self.obs[index_tensor].reshape(
            -1, self.history_len, self.frame_dim
        )
        chronological = torch.flip(latest_first, dims=(1,))
        histories = (chronological - self.frame_mean) / self.frame_std
        actions = (self.actions[index_tensor] - self.action_mean) / self.action_std
        targets = (self.targets[index_tensor] - self.target_mean) / self.target_std
        return list(zip(histories.unbind(0), actions.unbind(0), targets.unbind(0)))
