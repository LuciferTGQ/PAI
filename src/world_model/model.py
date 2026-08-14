from __future__ import annotations

from typing import Any, Dict

import torch
from torch import nn


def _validate_inputs(
    history: torch.Tensor,
    action: torch.Tensor,
    history_len: int,
    frame_dim: int,
    action_dim: int,
) -> None:
    if history.ndim != 3 or history.shape[1:] != (history_len, frame_dim):
        raise ValueError(
            f"history must have shape [B,{history_len},{frame_dim}], got {tuple(history.shape)}"
        )
    if action.ndim != 2 or action.shape[1] != action_dim:
        raise ValueError(f"action must have shape [B,{action_dim}], got {tuple(action.shape)}")


class MLPWorldModel(nn.Module):
    """Flattened-history MLP with the common world-model interface."""

    def __init__(
        self,
        frame_dim: int = 6,
        action_dim: int = 3,
        history_len: int = 30,
        hidden_dim: int = 256,
        num_layers: int = 2,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        self.frame_dim = frame_dim
        self.action_dim = action_dim
        self.history_len = history_len
        input_dim = history_len * frame_dim + action_dim
        layers: list[nn.Module] = []
        for layer_index in range(num_layers):
            layers.extend(
                [
                    nn.Linear(input_dim if layer_index == 0 else hidden_dim, hidden_dim),
                    nn.GELU(),
                    nn.Dropout(dropout),
                ]
            )
        layers.append(nn.Linear(hidden_dim, frame_dim))
        self.network = nn.Sequential(*layers)

    def forward(self, history: torch.Tensor, action: torch.Tensor) -> torch.Tensor:
        _validate_inputs(history, action, self.history_len, self.frame_dim, self.action_dim)
        return self.network(torch.cat((history.flatten(1), action), dim=-1))


class RecurrentWorldModel(nn.Module):
    """GRU/LSTM history encoder with action fusion and a shared prediction head."""

    def __init__(
        self,
        recurrent_type: str,
        frame_dim: int = 6,
        action_dim: int = 3,
        history_len: int = 30,
        hidden_size: int = 128,
        num_layers: int = 2,
        dropout: float = 0.1,
        action_hidden_dim: int = 64,
        head_hidden_dim: int = 128,
    ) -> None:
        super().__init__()
        self.frame_dim = frame_dim
        self.action_dim = action_dim
        self.history_len = history_len
        recurrent_type = recurrent_type.lower()
        recurrent_class = {"gru": nn.GRU, "lstm": nn.LSTM}.get(recurrent_type)
        if recurrent_class is None:
            raise ValueError(f"Unsupported recurrent type: {recurrent_type}")
        self.recurrent_type = recurrent_type
        self.recurrent = recurrent_class(
            input_size=frame_dim,
            hidden_size=hidden_size,
            num_layers=num_layers,
            dropout=dropout if num_layers > 1 else 0.0,
            batch_first=True,
        )
        self.action_embedding = nn.Sequential(
            nn.Linear(action_dim, action_hidden_dim), nn.GELU()
        )
        self.prediction_head = nn.Sequential(
            nn.Linear(hidden_size + action_hidden_dim, head_hidden_dim),
            nn.GELU(),
            nn.Linear(head_hidden_dim, frame_dim),
        )

    def forward(self, history: torch.Tensor, action: torch.Tensor) -> torch.Tensor:
        _validate_inputs(history, action, self.history_len, self.frame_dim, self.action_dim)
        _, hidden = self.recurrent(history)
        if self.recurrent_type == "lstm":
            hidden = hidden[0]
        temporal = hidden[-1]
        action_repr = self.action_embedding(action)
        return self.prediction_head(torch.cat((temporal, action_repr), dim=-1))


class TemporalTransformer(nn.Module):
    """Predict the next six-dimensional IB frame from history and action."""

    def __init__(
        self,
        frame_dim: int = 6,
        action_dim: int = 3,
        history_len: int = 30,
        d_model: int = 64,
        nhead: int = 4,
        num_layers: int = 2,
        dim_feedforward: int = 128,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        self.frame_dim = frame_dim
        self.action_dim = action_dim
        self.history_len = history_len
        self.frame_embedding = nn.Linear(frame_dim, d_model)
        self.position_embedding = nn.Parameter(torch.zeros(1, history_len, d_model))
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(
            encoder_layer, num_layers=num_layers, enable_nested_tensor=False
        )
        self.temporal_norm = nn.LayerNorm(d_model)
        self.action_embedding = nn.Sequential(
            nn.Linear(action_dim, d_model),
            nn.GELU(),
            nn.Linear(d_model, d_model),
        )
        self.prediction_head = nn.Sequential(
            nn.Linear(2 * d_model, dim_feedforward),
            nn.GELU(),
            nn.Linear(dim_feedforward, frame_dim),
        )
        nn.init.normal_(self.position_embedding, mean=0.0, std=0.02)

    def forward(self, history: torch.Tensor, action: torch.Tensor) -> torch.Tensor:
        _validate_inputs(history, action, self.history_len, self.frame_dim, self.action_dim)
        tokens = self.frame_embedding(history) + self.position_embedding
        encoded = self.encoder(tokens)
        temporal = self.temporal_norm(encoded[:, -1])
        action_repr = self.action_embedding(action)
        return self.prediction_head(torch.cat((temporal, action_repr), dim=-1))


def build_world_model(model_config: Dict[str, Any]) -> nn.Module:
    """Build any supported architecture from a config or checkpoint dictionary."""
    config = dict(model_config)
    model_type = str(config.pop("type", "transformer")).lower().replace("-", "_")
    if model_type in {"transformer", "transformer_2l", "transformer_4l"}:
        return TemporalTransformer(**config)
    if model_type == "mlp":
        return MLPWorldModel(**config)
    if model_type in {"gru", "lstm"}:
        return RecurrentWorldModel(recurrent_type=model_type, **config)
    raise ValueError(f"Unsupported world-model type: {model_type}")
