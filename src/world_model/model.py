from __future__ import annotations

import torch
from torch import nn


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
        if history.ndim != 3 or history.shape[1:] != (self.history_len, self.frame_dim):
            raise ValueError(
                f"history must have shape [B,{self.history_len},{self.frame_dim}], got {tuple(history.shape)}"
            )
        if action.ndim != 2 or action.shape[1] != self.action_dim:
            raise ValueError(f"action must have shape [B,{self.action_dim}], got {tuple(action.shape)}")
        tokens = self.frame_embedding(history) + self.position_embedding
        encoded = self.encoder(tokens)
        temporal = self.temporal_norm(encoded[:, -1])
        action_repr = self.action_embedding(action)
        return self.prediction_head(torch.cat((temporal, action_repr), dim=-1))
