from pathlib import Path

import numpy as np
import torch

from src.data.ib_dataset import (
    IBTransitionDataset,
    compute_normalization,
    load_ib_npz,
    trajectory_spans,
    validate_ib_semantics,
)


def test_raw_end_offsets_are_parsed() -> None:
    spans = trajectory_spans(np.array([3, 6, 9]), 9)
    assert np.array_equal(spans, np.array([[0, 3], [3, 6], [6, 9]]))


def test_api_start_offsets_are_parsed() -> None:
    spans = trajectory_spans(np.array([0, 3, 6]), 9)
    assert np.array_equal(spans, np.array([[0, 3], [3, 6], [6, 9]]))


def test_released_ib_m100_semantics_when_cached() -> None:
    path = Path(__file__).resolve().parents[1] / "data/raw/ib-medium-100-train.npz"
    if not path.exists():
        return
    audit = validate_ib_semantics(load_ib_npz(path))
    assert audit["transitions"] == 100_000
    assert audit["trajectories"] == 100
    assert audit["trajectory_lengths"] == [1000]


def test_vectorized_fetch_matches_individual_samples() -> None:
    path = Path(__file__).resolve().parents[1] / "data/raw/ib-medium-10-val.npz"
    if not path.exists():
        return
    data = load_ib_npz(path)
    dataset = IBTransitionDataset(data, compute_normalization(data))
    indices = [0, 7, 999]
    individual = [dataset[index] for index in indices]
    vectorized = dataset.__getitems__(indices)
    for expected, actual in zip(individual, vectorized):
        for expected_tensor, actual_tensor in zip(expected, actual):
            torch.testing.assert_close(expected_tensor, actual_tensor, rtol=0, atol=0)
