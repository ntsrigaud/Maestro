"""Unit tests for GraphSequenceDatasetBuilder (graph-schema-v1, Phase 1.5)."""

from __future__ import annotations

import json
import tempfile
from pathlib import Path

import numpy as np
import pytest

from maestro.domain.entities import LabeledGraphFrame
from maestro.pipeline import graph_schema
from maestro.training.dataset_builder import (
    GraphSequenceDatasetBuilder,
    GraphSequenceSample,
    SequenceDatasetBuilder,  # ensure old builder is still importable
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

VOCAB = ["swipe_left", "swipe_right", "open_palm_hold", "unknown"]


def _make_frame(
    label: str = "swipe_left",
    recording_id: str = "rec_01",
    ts: int = 0,
    has_right_hand: bool = True,
) -> LabeledGraphFrame:
    """Return a synthetic LabeledGraphFrame with plausible data."""
    kp = np.zeros((graph_schema.NUM_NODES, graph_schema.NUM_COORDS), dtype=np.float32)
    vm = np.zeros(graph_schema.NUM_NODES, dtype=bool)
    if has_right_hand:
        kp[0:21] = 0.5  # non-zero right hand
        vm[0:21] = True
        # Also set at least one pose node for scale reference.
        kp[42] = [0.3, 0.7, 0.0]
        vm[42] = True
    return LabeledGraphFrame(
        keypoints=kp,
        valid_mask=vm,
        label=label,
        recording_id=recording_id,
        timestamp_ms=ts,
    )


def _make_builder(
    window_length: int = 24,
    window_step: int = 3,
    min_valid_ratio: float = 0.8,
) -> GraphSequenceDatasetBuilder:
    return GraphSequenceDatasetBuilder(
        vocabulary=VOCAB,
        window_length=window_length,
        window_step=window_step,
        min_valid_ratio=min_valid_ratio,
        label_strategy="majority",
    )


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


class TestGraphSequenceDatasetBuilderBasics:
    def test_empty_builder_raises_on_save(self, tmp_path: Path) -> None:
        builder = _make_builder()
        with pytest.raises(RuntimeError, match="No sequence samples"):
            builder.save_npz(tmp_path / "out.npz")

    def test_unknown_label_raises(self) -> None:
        builder = _make_builder()
        with pytest.raises(ValueError, match="Unknown label"):
            builder.add_frame(_make_frame(label="not_in_vocab"))

    def test_no_window_before_fill(self) -> None:
        builder = _make_builder(window_length=24)
        for i in range(23):
            result = builder.add_frame(_make_frame(ts=i * 33))
            assert result == []

    def test_window_emitted_at_fill(self) -> None:
        builder = _make_builder(window_length=24, window_step=3, min_valid_ratio=0.8)
        emitted = []
        for i in range(24):
            emitted.extend(builder.add_frame(_make_frame(ts=i * 33)))

        assert len(emitted) == 1
        assert isinstance(emitted[0], GraphSequenceSample)

    def test_sequence_shape(self) -> None:
        builder = _make_builder(window_length=24, window_step=3, min_valid_ratio=0.0)
        for i in range(24):
            builder.add_frame(_make_frame(ts=i * 33))
        sample = builder.samples[0]
        assert sample.sequence.shape == (24, graph_schema.NUM_NODES, graph_schema.NUM_COORDS)
        assert sample.valid_masks.shape == (24, graph_schema.NUM_NODES)

    def test_recording_boundary_resets_buffer(self) -> None:
        """Switching recording_id must reset the sliding window."""
        builder = _make_builder(window_length=24, window_step=3, min_valid_ratio=0.0)
        # Push 20 frames from rec_01 — not enough to fill.
        for i in range(20):
            builder.add_frame(_make_frame(recording_id="rec_01", ts=i * 33))
        # Switch to rec_02 — should reset the buffer.
        for i in range(24):
            builder.add_frame(_make_frame(recording_id="rec_02", ts=i * 33))

        # All samples should belong to rec_02.
        assert all(s.recording_id == "rec_02" for s in builder.samples)


class TestGraphSequenceDatasetBuilderNPZ:
    def test_save_and_load_npz(self, tmp_path: Path) -> None:
        builder = _make_builder(window_length=10, window_step=1, min_valid_ratio=0.0)
        for i in range(15):
            builder.add_frame(_make_frame(ts=i * 33))

        out_path = tmp_path / "test.npz"
        builder.save_npz(out_path)

        assert out_path.exists()

        # Load via the universal load_npz helper.
        from maestro.training.dataset_builder import SequenceDatasetBuilder

        loaded = SequenceDatasetBuilder.load_npz(out_path)

        seqs = loaded["sequences"]
        assert seqs.ndim == 4, "Graph NPZ sequences must be 4-D (N, T, N_nodes, C)"
        n_samples = len(builder.samples)
        assert seqs.shape == (n_samples, 10, graph_schema.NUM_NODES, graph_schema.NUM_COORDS)
        assert "adjacency" in loaded
        assert loaded["adjacency"].shape == (graph_schema.NUM_NODES, graph_schema.NUM_NODES)
        assert "valid_masks" in loaded
        assert loaded["valid_masks"].shape == (n_samples, 10, graph_schema.NUM_NODES)

        # Metadata checks.
        metadata = loaded["metadata"]
        assert metadata["graph_schema_version"] == graph_schema.GRAPH_SCHEMA_VERSION
        assert metadata["num_nodes"] == graph_schema.NUM_NODES
        assert metadata["num_coords"] == graph_schema.NUM_COORDS
        assert metadata["vocabulary"] == VOCAB

    def test_npz_label_ids_match_vocab(self, tmp_path: Path) -> None:
        builder = _make_builder(window_length=5, window_step=1, min_valid_ratio=0.0)
        for i in range(5):
            builder.add_frame(_make_frame(label="swipe_right", ts=i * 33))

        out = builder.save_npz(tmp_path / "out.npz")
        loaded = SequenceDatasetBuilder.load_npz(out)

        expected_id = VOCAB.index("swipe_right")
        assert all(loaded["label_ids"] == expected_id)


class TestOldBuilderStillWorks:
    """Regression: SequenceDatasetBuilder must remain importable and functional."""

    def test_old_builder_importable(self) -> None:
        from maestro.training.dataset_builder import SequenceDatasetBuilder

        assert SequenceDatasetBuilder is not None

    def test_old_builder_load_npz_flat(self, tmp_path: Path) -> None:
        """load_npz on a legacy flat NPZ must not return adjacency."""
        from maestro.training.dataset_builder import (
            LabeledLandmarkFrame,
            SequenceDatasetBuilder,
        )

        builder = SequenceDatasetBuilder(
            vocabulary=VOCAB,
            window_length=5,
            window_step=1,
            min_valid_ratio=0.0,
        )
        for i in range(5):
            frame = LabeledLandmarkFrame(
                recording_id="r1",
                timestamp_ms=i * 33,
                label="swipe_left",
                landmarks=[[0.0, 0.0, 0.0]] * 21,
                valid=True,
            )
            builder.add_frame(frame)

        out = builder.save_npz(tmp_path / "flat.npz")
        loaded = SequenceDatasetBuilder.load_npz(out)

        assert loaded["sequences"].ndim == 3, "Flat NPZ must be 3-D"
        assert "adjacency" not in loaded
