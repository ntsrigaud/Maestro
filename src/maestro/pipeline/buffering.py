from __future__ import annotations

from collections import deque

import numpy as np

from maestro.domain.entities import FeatureFrame, SequenceWindow


class SlidingWindowSequenceBuffer:
    """Ring buffer that emits sliding windows when enough valid frames exist."""

    def __init__(
        self,
        *,
        window_length: int = 16,
        window_step: int = 3,
        min_valid_ratio: float = 0.8,
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
        self._features: deque[FeatureFrame] = deque(maxlen=window_length)
        self._push_count = 0
        self._last_emit_push = window_length - window_step

    def push(self, feature: FeatureFrame) -> None:
        self._features.append(feature)
        self._push_count += 1

    def ready(self) -> bool:
        if len(self._features) < self._window_length:
            return False

        if (self._push_count - self._last_emit_push) < self._window_step:
            return False

        return self._current_valid_ratio() >= self._min_valid_ratio

    def window(self) -> SequenceWindow:
        if not self.ready():
            raise RuntimeError("Sequence window is not ready")

        ordered = list(self._features)
        self._last_emit_push = self._push_count

        return SequenceWindow(
            features=np.stack([item.vector for item in ordered]).astype(np.float32, copy=False),
            start_timestamp_ms=ordered[0].timestamp_ms,
            end_timestamp_ms=ordered[-1].timestamp_ms,
            valid_ratio=self._current_valid_ratio(),
        )

    def _current_valid_ratio(self) -> float:
        if len(self._features) < self._window_length:
            return 0.0

        valid_count = sum(1 for item in self._features if item.valid)
        return valid_count / self._window_length

    def current_valid_ratio(self) -> float:
        return self._current_valid_ratio()
