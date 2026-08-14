from pathlib import Path

import numpy as np

from src.data.ib_dataset import load_ib_npz, trajectory_spans, validate_ib_semantics


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

