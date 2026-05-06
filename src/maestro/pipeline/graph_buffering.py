"""Sliding-window buffer for graph-structured gesture frames.

Mirrors ``SlidingWindowSequenceBuffer`` but operates on ``GraphFrame``
objects and emits ``(T, N, C)`` arrays instead of ``SequenceWindow``.
"""

from __future__ import annotations

from collections import deque

import numpy as np

from maestro.domain.entities import GraphFrame
from maestro.pipeline.graph_schema import MIN_VALID_RATIO, NUM_COORDS, NUM_NODES


class GraphSequenceBuffer:
    """Ring buffer that emits ``(T, N, C)`` windows of normalised graph frames.

    Emits a new window every ``window_step`` pushes once the buffer holds
    ``window_length`` frames **and** the valid-frame ratio meets
    ``min_valid_ratio``.  A frame is considered valid when ``frame.valid``
    is ``True`` (i.e. at least one hand node block is non-zero).

    Args:
        window_length: Number of frames per window (default 24, STGCN pipeline;
            flat-feature pipeline uses 32).
        window_step: Emit a new window every N pushes (default 3).
        min_valid_ratio: Minimum fraction of valid frames required to emit
            (default 0.8). Frames with no detected hand are invalid.
        num_nodes: Node count (default ``NUM_NODES`` = 67).
        num_coords: Coordinate count per node (default ``NUM_COORDS`` = 3).
    """

    def __init__(
        self,
        *,
        window_length: int = 24,
        window_step: int = 3,
        min_valid_ratio: float = MIN_VALID_RATIO,
        num_nodes: int = NUM_NODES,
        num_coords: int = NUM_COORDS,
    ) -> None:
        if window_length <= 0:
            raise ValueError("window_length must be > 0")
        if window_step <= 0:
            raise ValueError("window_step must be > 0")
        if not (0.0 <= min_valid_ratio <= 1.0):
            raise ValueError("min_valid_ratio must be between 0.0 and 1.0")

        self._window_length = window_length
        self._window_step = window_step
        self._min_valid_ratio = min_valid_ratio
        self._num_nodes = num_nodes
        self._num_coords = num_coords

        self._frames: deque[GraphFrame] = deque(maxlen=window_length)
        self._push_count: int = 0
        self._last_emit_push: int = window_length - window_step

    # ------------------------------------------------------------------
    # Public interface
    # ------------------------------------------------------------------

    def push(self, frame: GraphFrame) -> None:
        """Add *frame* to the ring buffer."""
        if frame.nodes.shape != (self._num_nodes, self._num_coords):
            raise ValueError(
                f"Expected frame.nodes shape ({self._num_nodes}, {self._num_coords}), "
                f"got {frame.nodes.shape}"
            )
        self._frames.append(frame)
        self._push_count += 1

    def ready(self) -> bool:
        """Return ``True`` when the buffer can emit a valid window."""
        if len(self._frames) < self._window_length:
            return False
        if (self._push_count - self._last_emit_push) < self._window_step:
            return False
        return self._current_valid_ratio() >= self._min_valid_ratio

    def window(self) -> tuple[np.ndarray, np.ndarray]:
        """Emit the current sliding window.

        Returns:
            ``(sequences, valid_masks)`` where:

            - ``sequences``:   ``np.ndarray (T, N, C)`` float32 — normalised
              graph frames ordered from oldest to newest.
            - ``valid_masks``: ``np.ndarray (T, N)`` bool — per-frame per-node
              validity flags.

        Raises:
            RuntimeError: If the buffer is not ready (see :meth:`ready`).
        """
        if not self.ready():
            raise RuntimeError("Sequence window is not ready")

        ordered = list(self._frames)
        self._last_emit_push = self._push_count

        sequences = np.stack([f.nodes for f in ordered], axis=0).astype(np.float32)  # (T, N, C)
        valid_masks = np.stack([f.valid_mask for f in ordered], axis=0)  # (T, N)

        return sequences, valid_masks

    def current_valid_ratio(self) -> float:
        """Return the fraction of valid frames currently in the buffer."""
        return self._current_valid_ratio()

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _current_valid_ratio(self) -> float:
        if len(self._frames) < self._window_length:
            return 0.0
        valid_count = sum(1 for f in self._frames if f.valid)
        return valid_count / self._window_length
