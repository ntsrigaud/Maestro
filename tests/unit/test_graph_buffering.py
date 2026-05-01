"""Unit tests for GraphSequenceBuffer (graph-schema-v1, Phase 1.4)."""

from __future__ import annotations

import numpy as np
import pytest

from maestro.domain.entities import GraphFrame
from maestro.pipeline.graph_buffering import GraphSequenceBuffer
from maestro.pipeline.graph_schema import NUM_COORDS, NUM_NODES


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_graph_frame(
    ts: int = 0,
    valid: bool = True,
    fill: float = 0.0,
) -> GraphFrame:
    """Construct a synthetic GraphFrame with constant node coordinates."""
    nodes = np.full((NUM_NODES, NUM_COORDS), fill, dtype=np.float32)
    mask = np.ones(NUM_NODES, dtype=bool) if valid else np.zeros(NUM_NODES, dtype=bool)
    return GraphFrame(nodes=nodes, valid_mask=mask, timestamp_ms=ts, valid=valid)


def _push_n(buffer: GraphSequenceBuffer, n: int, *, valid: bool = True) -> None:
    """Push *n* frames into *buffer*."""
    for i in range(n):
        buffer.push(_make_graph_frame(ts=i * 33, valid=valid))


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


class TestGraphSequenceBufferBasics:
    def test_not_ready_before_window_length(self) -> None:
        buf = GraphSequenceBuffer(window_length=24, window_step=3)
        _push_n(buf, 23)
        assert not buf.ready()

    def test_ready_after_window_length(self) -> None:
        buf = GraphSequenceBuffer(window_length=24, window_step=3)
        _push_n(buf, 24)
        assert buf.ready()

    def test_window_output_shapes(self) -> None:
        buf = GraphSequenceBuffer(window_length=24, window_step=3)
        _push_n(buf, 24)

        sequences, valid_masks = buf.window()

        assert sequences.shape == (24, NUM_NODES, NUM_COORDS)
        assert valid_masks.shape == (24, NUM_NODES)
        assert sequences.dtype == np.float32
        assert valid_masks.dtype == bool

    def test_window_not_ready_after_emit(self) -> None:
        """After emitting a window, buffer must not be ready until step more frames arrive."""
        buf = GraphSequenceBuffer(window_length=24, window_step=3)
        _push_n(buf, 24)
        buf.window()  # consume
        assert not buf.ready()

    def test_window_ready_again_after_step(self) -> None:
        buf = GraphSequenceBuffer(window_length=24, window_step=3)
        _push_n(buf, 24)
        buf.window()
        _push_n(buf, 3)
        assert buf.ready()

    def test_step_cadence(self) -> None:
        """Window should be emittable exactly every window_step pushes after fill."""
        buf = GraphSequenceBuffer(window_length=24, window_step=3)
        _push_n(buf, 24)

        emit_count = 0
        for _ in range(30):
            if buf.ready():
                buf.window()
                emit_count += 1
            buf.push(_make_graph_frame())

        # 30 extra pushes / step=3 = 10 more windows (first emit was at fill, then
        # every 3 iterations: iters 3, 6, ..., 27 = 9 more; total 10).
        assert emit_count == 10

    def test_window_values_match_pushed_frames(self) -> None:
        """Sequences array must reflect the nodes from pushed frames."""
        buf = GraphSequenceBuffer(window_length=3, window_step=1)
        for v in (1.0, 2.0, 3.0):
            buf.push(_make_graph_frame(fill=v))

        sequences, _ = buf.window()

        assert sequences[0, 0, 0] == pytest.approx(1.0)
        assert sequences[1, 0, 0] == pytest.approx(2.0)
        assert sequences[2, 0, 0] == pytest.approx(3.0)

    def test_raises_if_not_ready(self) -> None:
        buf = GraphSequenceBuffer(window_length=24, window_step=3)
        with pytest.raises(RuntimeError, match="not ready"):
            buf.window()


class TestGraphSequenceBufferValidRatio:
    def test_all_valid_frames(self) -> None:
        buf = GraphSequenceBuffer(window_length=24, window_step=3, min_valid_ratio=0.8)
        _push_n(buf, 24, valid=True)
        assert buf.ready()
        assert buf.current_valid_ratio() == pytest.approx(1.0)

    def test_all_invalid_frames_not_ready(self) -> None:
        buf = GraphSequenceBuffer(window_length=24, window_step=3, min_valid_ratio=0.8)
        _push_n(buf, 24, valid=False)
        assert not buf.ready()

    def test_partial_valid_ratio_threshold(self) -> None:
        """Exactly 80% valid (threshold=0.8) should be ready."""
        buf = GraphSequenceBuffer(window_length=10, window_step=1, min_valid_ratio=0.8)
        _push_n(buf, 8, valid=True)
        _push_n(buf, 2, valid=False)
        assert buf.ready()

    def test_below_valid_ratio_threshold(self) -> None:
        """79% valid (threshold=0.8) should not be ready."""
        buf = GraphSequenceBuffer(window_length=100, window_step=1, min_valid_ratio=0.8)
        _push_n(buf, 79, valid=True)
        _push_n(buf, 21, valid=False)
        assert not buf.ready()


class TestGraphSequenceBufferValidation:
    def test_wrong_node_shape_raises(self) -> None:
        buf = GraphSequenceBuffer()
        bad_frame = GraphFrame(
            nodes=np.zeros((10, 3), dtype=np.float32),  # wrong N
            valid_mask=np.ones(10, dtype=bool),
            timestamp_ms=0,
            valid=True,
        )
        with pytest.raises(ValueError, match="Expected frame.nodes shape"):
            buf.push(bad_frame)

    def test_custom_num_nodes_accepted(self) -> None:
        buf = GraphSequenceBuffer(num_nodes=10, num_coords=3, window_length=5, window_step=1)
        for _ in range(5):
            frame = GraphFrame(
                nodes=np.zeros((10, 3), dtype=np.float32),
                valid_mask=np.ones(10, dtype=bool),
                timestamp_ms=0,
                valid=True,
            )
            buf.push(frame)
        sequences, masks = buf.window()
        assert sequences.shape == (5, 10, 3)
        assert masks.shape == (5, 10)
