from __future__ import annotations

from pathlib import Path

import numpy as np
import torch

from src.data.ib_dataset import NormalizationStats
from src.world_model.model import build_world_model


class FrozenWorldModel:
    """Stable inference interface for the frozen Module A checkpoint."""

    def __init__(self, checkpoint_path: str | Path, device: str = "cuda") -> None:
        if device.startswith("cuda") and not torch.cuda.is_available():
            device = "cpu"
        self.device = torch.device(device)
        checkpoint = torch.load(checkpoint_path, map_location=self.device, weights_only=False)
        self.model_config = checkpoint["model_config"]
        self.stats = NormalizationStats.from_dict(checkpoint["normalization"])
        self.model = build_world_model(self.model_config).to(self.device)
        self.model.load_state_dict(checkpoint["model_state"])
        self.model.eval()
        for parameter in self.model.parameters():
            parameter.requires_grad_(False)

        self.history_len = int(self.model_config["history_len"])
        self.frame_dim = int(self.model_config["frame_dim"])
        self.obs_dim = self.history_len * self.frame_dim
        self._frame_mean = torch.as_tensor(self.stats.frame_mean, device=self.device)
        self._frame_std = torch.as_tensor(self.stats.frame_std, device=self.device)
        self._action_mean = torch.as_tensor(self.stats.action_mean, device=self.device)
        self._action_std = torch.as_tensor(self.stats.action_std, device=self.device)
        self._target_mean = torch.as_tensor(self.stats.target_mean, device=self.device)
        self._target_std = torch.as_tensor(self.stats.target_std, device=self.device)

    @torch.inference_mode()
    def predict_next_frame(self, history: np.ndarray, action: np.ndarray) -> np.ndarray:
        histories = np.asarray(history, dtype=np.float32)
        actions = np.asarray(action, dtype=np.float32)
        single = histories.ndim in (1, 2) and not (
            histories.ndim == 2 and histories.shape == (histories.shape[0], self.obs_dim)
        )
        if histories.ndim == 1:
            histories = histories[None, :]
        elif histories.ndim == 2 and histories.shape == (self.history_len, self.frame_dim):
            histories = histories.reshape(1, -1)
        elif histories.ndim == 3:
            histories = histories.reshape(len(histories), -1)
        if histories.ndim != 2 or histories.shape[1] != self.obs_dim:
            raise ValueError(f"history must end in {self.obs_dim} values, got {histories.shape}")
        if actions.ndim == 1:
            actions = actions[None, :]
        if len(actions) != len(histories):
            raise ValueError("history and action batch sizes differ")

        latest_first = torch.as_tensor(histories, device=self.device).reshape(
            -1, self.history_len, self.frame_dim
        )
        chronological = torch.flip(latest_first, dims=(1,))
        normalized_history = (chronological - self._frame_mean) / self._frame_std
        action_tensor = torch.as_tensor(actions, device=self.device)
        normalized_action = (action_tensor - self._action_mean) / self._action_std
        normalized_prediction = self.model(normalized_history, normalized_action)
        prediction = normalized_prediction * self._target_std + self._target_mean
        result = prediction.cpu().numpy()
        return result[0] if single else result

    @torch.inference_mode()
    def predict_next_frame_tensor(
        self, history: torch.Tensor, action: torch.Tensor
    ) -> torch.Tensor:
        """Batched raw-space inference without CPU transfers for policy training.

        Histories use the released NeoRL convention: flattened, latest frame first.
        The returned frame is in raw simulator units. The world model stays frozen.
        """
        histories = history.to(device=self.device, dtype=torch.float32)
        actions = action.to(device=self.device, dtype=torch.float32)
        if histories.ndim == 3:
            histories = histories.reshape(len(histories), -1)
        if histories.ndim != 2 or histories.shape[1] != self.obs_dim:
            raise ValueError(
                f"history must have shape [B,{self.obs_dim}], got {tuple(histories.shape)}"
            )
        if actions.ndim != 2 or len(actions) != len(histories):
            raise ValueError("action must be a matching [B,action_dim] tensor")

        latest_first = histories.reshape(-1, self.history_len, self.frame_dim)
        chronological = torch.flip(latest_first, dims=(1,))
        normalized_history = (chronological - self._frame_mean) / self._frame_std
        normalized_action = (actions - self._action_mean) / self._action_std
        normalized_prediction = self.model(normalized_history, normalized_action)
        return normalized_prediction * self._target_std + self._target_mean

    def update_history_tensor(
        self, history: torch.Tensor, next_frame: torch.Tensor
    ) -> torch.Tensor:
        histories = history.to(device=self.device, dtype=torch.float32)
        frames = next_frame.to(device=self.device, dtype=torch.float32)
        if histories.ndim != 2 or histories.shape[1] != self.obs_dim:
            raise ValueError(f"history must have shape [B,{self.obs_dim}]")
        if frames.shape != (len(histories), self.frame_dim):
            raise ValueError(f"next_frame must have shape [B,{self.frame_dim}]")
        return torch.cat((frames, histories[:, :-self.frame_dim]), dim=1)

    def rollout(self, history: np.ndarray, action_sequence: np.ndarray) -> np.ndarray:
        current = np.asarray(history, dtype=np.float32).reshape(self.obs_dim).copy()
        actions = np.asarray(action_sequence, dtype=np.float32)
        predictions = []
        for action in actions:
            next_frame = self.predict_next_frame(current, action)
            predictions.append(next_frame)
            current = self.update_history(current, next_frame)
        return np.asarray(predictions, dtype=np.float32)

    def update_history(self, history: np.ndarray, next_frame: np.ndarray) -> np.ndarray:
        history = np.asarray(history, dtype=np.float32).reshape(self.obs_dim)
        next_frame = np.asarray(next_frame, dtype=np.float32).reshape(self.frame_dim)
        return np.concatenate((next_frame, history[:-self.frame_dim]))
