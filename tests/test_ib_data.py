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


def test_streaming_normalization_matches_direct_numpy() -> None:
    path = Path(__file__).resolve().parents[1] / "data/raw/ib-medium-10-val.npz"
    if not path.exists():
        return
    data = load_ib_npz(path)
    stats = compute_normalization(data)
    frames = data["obs"].reshape(-1, 30, 6).astype(np.float64)
    actions = data["action"].astype(np.float64)
    targets = data["next_obs"][:, :6].astype(np.float64)
    np.testing.assert_allclose(stats.frame_mean, frames.mean(axis=(0, 1)), rtol=1e-6)
    frame_std = frames.std(axis=(0, 1))
    frame_std[frame_std < 1e-6] = 1.0
    np.testing.assert_allclose(stats.frame_std, frame_std, rtol=1e-6)
    np.testing.assert_allclose(stats.action_mean, actions.mean(axis=0), rtol=1e-6)
    np.testing.assert_allclose(stats.action_std, actions.std(axis=0), rtol=1e-6)
    np.testing.assert_allclose(stats.target_mean, targets.mean(axis=0), rtol=1e-6)
    target_std = targets.std(axis=0)
    target_std[target_std < 1e-6] = 1.0
    np.testing.assert_allclose(stats.target_std, target_std, rtol=1e-6)


def test_memory_mapped_directory_loader(tmp_path) -> None:
    source_path = Path(__file__).resolve().parents[1] / "data/raw/ib-medium-10-val.npz"
    if not source_path.exists():
        return
    source = load_ib_npz(source_path)
    for key, value in source.items():
        np.save(tmp_path / f"{key}.npy", value[:10] if key != "index" else np.array([10]))
    loaded = load_ib_npz(tmp_path)
    assert all(isinstance(value, np.memmap) for value in loaded.values())
    assert loaded["obs"].shape == (10, 180)
