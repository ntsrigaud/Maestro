from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from gesture_control.training.dataset_builder import (
    LabeledLandmarkFrame,
    SequenceDatasetBuilder,
)


def _landmarks(frame_idx: int) -> list[list[float]]:
    points: list[list[float]] = []
    for i in range(21):
        points.append(
            [
                0.2 + (i * 0.01) + (frame_idx * 0.0005),
                0.3 + (i * 0.02) + (frame_idx * 0.0003),
                i * 0.005,
            ]
        )
    return points


def _frame(index: int, *, label: str = "swipe_right") -> LabeledLandmarkFrame:
    return LabeledLandmarkFrame(
        recording_id="rec-1",
        timestamp_ms=index * 33,
        label=label,
        landmarks=_landmarks(index),
        valid=True,
        handedness="Right",
        detect_conf=0.95,
        track_conf=1.0,
    )


def test_sequence_extraction_and_shape_consistency() -> None:
    builder = SequenceDatasetBuilder(
        vocabulary=["swipe_right", "unknown"],
        window_length=4,
        window_step=2,
        min_valid_ratio=0.75,
    )

    frames = [_frame(i) for i in range(8)]
    samples = builder.build_from_frames(frames)

    assert len(samples) == 3
    for sample in samples:
        assert len(sample.sequence) == 4
        assert len(sample.sequence[0]) == 147
        assert sample.label == "swipe_right"
        assert sample.label_id == 0


def test_label_alignment_majority_strategy() -> None:
    builder = SequenceDatasetBuilder(
        vocabulary=["swipe_right", "swipe_up", "unknown"],
        window_length=5,
        window_step=1,
        min_valid_ratio=0.8,
        label_strategy="majority",
    )

    labels = ["swipe_right", "swipe_right", "swipe_up", "swipe_right", "swipe_up"]
    for i, label in enumerate(labels):
        builder.add_frame(_frame(i, label=label))

    samples = builder.samples
    assert len(samples) == 1
    assert samples[0].label == "swipe_right"
    assert samples[0].label_id == 0


def test_unknown_label_raises_validation_error() -> None:
    builder = SequenceDatasetBuilder(
        vocabulary=["swipe_right", "unknown"], window_length=4
    )

    with pytest.raises(ValueError, match="Unknown label"):
        builder.add_frame(_frame(0, label="not_in_vocab"))


def test_dataset_serialization_roundtrip(tmp_path: Path) -> None:
    builder = SequenceDatasetBuilder(
        vocabulary=["swipe_right", "unknown"],
        window_length=4,
        window_step=2,
        min_valid_ratio=0.75,
    )
    for i in range(8):
        builder.add_frame(_frame(i))

    output = builder.save_npz(tmp_path / "train_dataset.npz")
    payload = SequenceDatasetBuilder.load_npz(output)

    assert payload["sequences"].shape == (3, 4, 147)
    assert payload["label_ids"].shape == (3,)
    assert payload["metadata"]["window_length"] == 4
    assert payload["metadata"]["feature_dim"] == 147
    np.testing.assert_array_equal(
        payload["labels"], np.asarray(["swipe_right"] * 3, dtype=object)
    )
