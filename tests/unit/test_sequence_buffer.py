from __future__ import annotations

import pytest

from maestro.domain.entities import FeatureFrame
from maestro.pipeline.buffering import SlidingWindowSequenceBuffer


def _feature(index: int, *, valid: bool = True) -> FeatureFrame:
    return FeatureFrame(
        vector=[float(index)] * 126,
        timestamp_ms=index * 10,
        valid=valid,
    )


def test_buffer_not_ready_until_window_filled() -> None:
    buffer = SlidingWindowSequenceBuffer(
        window_length=4, window_step=2, min_valid_ratio=0.5
    )

    buffer.push(_feature(1))
    buffer.push(_feature(2))
    buffer.push(_feature(3))

    assert buffer.ready() is False

    buffer.push(_feature(4))
    assert buffer.ready() is True


def test_window_emits_and_honors_step() -> None:
    buffer = SlidingWindowSequenceBuffer(
        window_length=4, window_step=2, min_valid_ratio=0.5
    )

    for i in range(4):
        buffer.push(_feature(i))

    first = buffer.window()
    assert len(first.features) == 4
    assert first.start_timestamp_ms == 0
    assert first.end_timestamp_ms == 30

    assert buffer.ready() is False
    buffer.push(_feature(4))
    assert buffer.ready() is False
    buffer.push(_feature(5))
    assert buffer.ready() is True


def test_buffer_enforces_min_valid_ratio() -> None:
    buffer = SlidingWindowSequenceBuffer(
        window_length=5, window_step=1, min_valid_ratio=0.8
    )

    items = [
        _feature(0, valid=True),
        _feature(1, valid=True),
        _feature(2, valid=True),
        _feature(3, valid=False),
        _feature(4, valid=False),
    ]
    for item in items:
        buffer.push(item)

    assert buffer.ready() is False

    buffer.push(_feature(5, valid=True))
    assert buffer.ready() is False

    buffer.push(_feature(6, valid=True))
    assert buffer.ready() is False

    buffer.push(_feature(7, valid=True))
    assert buffer.ready() is False

    buffer.push(_feature(8, valid=True))
    assert buffer.ready() is True

    window = buffer.window()
    assert window.valid_ratio >= 0.8


def test_window_raises_if_not_ready() -> None:
    buffer = SlidingWindowSequenceBuffer(
        window_length=3, window_step=1, min_valid_ratio=1.0
    )
    buffer.push(_feature(1, valid=True))

    with pytest.raises(RuntimeError, match="not ready"):
        buffer.window()


def test_invalid_constructor_args_raise() -> None:
    with pytest.raises(ValueError, match="window_length"):
        SlidingWindowSequenceBuffer(window_length=0)

    with pytest.raises(ValueError, match="window_step"):
        SlidingWindowSequenceBuffer(window_step=0)

    with pytest.raises(ValueError, match="min_valid_ratio"):
        SlidingWindowSequenceBuffer(min_valid_ratio=1.1)
