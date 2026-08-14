import numpy as np
import torch

from src.world_model.model import TemporalTransformer


def test_temporal_transformer_shape() -> None:
    model = TemporalTransformer()
    output = model(torch.zeros(4, 30, 6), torch.zeros(4, 3))
    assert output.shape == (4, 6)


def test_latest_first_history_update() -> None:
    history = np.arange(180, dtype=np.float32)
    next_frame = np.arange(6, dtype=np.float32) + 1000
    updated = np.concatenate((next_frame, history[:-6]))
    assert np.array_equal(updated[:6], next_frame)
    assert np.array_equal(updated[6:], history[:-6])

