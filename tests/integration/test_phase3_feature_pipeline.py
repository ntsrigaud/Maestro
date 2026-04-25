from __future__ import annotations

from maestro.domain.entities import LandmarkPacket
from maestro.pipeline.buffering import SlidingWindowSequenceBuffer
from maestro.pipeline.feature_engineering import LandmarkFeatureTransformer


def _landmark_packet(index: int, *, valid: bool = True) -> LandmarkPacket:
    if not valid:
        return LandmarkPacket(
            landmarks=[],
            handedness=None,
            detect_conf=0.0,
            track_conf=0.0,
            timestamp_ms=index * 33,
            valid=False,
        )

    landmarks = []
    for i in range(21):
        landmarks.append(
            [
                0.2 + (i * 0.01) + (index * 0.0005),
                0.3 + (i * 0.02) + (index * 0.0003),
                i * 0.005,
            ]
        )

    return LandmarkPacket(
        landmarks=landmarks,
        handedness="Right",
        detect_conf=0.95,
        track_conf=1.0,
        timestamp_ms=index * 33,
        valid=True,
    )


def test_phase3_feature_to_sequence_window_flow() -> None:
    transformer = LandmarkFeatureTransformer()
    buffer = SlidingWindowSequenceBuffer(window_length=6, window_step=2, min_valid_ratio=0.8)

    for i in range(6):
        feature = transformer.transform(_landmark_packet(i, valid=True))
        buffer.push(feature)

    assert buffer.ready() is True
    window = buffer.window()

    assert len(window.features) == 6
    assert len(window.features[0]) == 147
    assert window.start_timestamp_ms == 0
    assert window.end_timestamp_ms == 165
    assert window.valid_ratio == 1.0


def test_phase3_pipeline_blocks_low_valid_ratio_windows() -> None:
    transformer = LandmarkFeatureTransformer()
    buffer = SlidingWindowSequenceBuffer(window_length=5, window_step=1, min_valid_ratio=0.8)

    validity_pattern = [True, True, False, False, True]
    for i, is_valid in enumerate(validity_pattern):
        feature = transformer.transform(_landmark_packet(i, valid=is_valid))
        buffer.push(feature)

    assert buffer.ready() is False

    buffer.push(transformer.transform(_landmark_packet(5, valid=True)))
    assert buffer.ready() is False

    buffer.push(transformer.transform(_landmark_packet(6, valid=True)))
    assert buffer.ready() is False

    buffer.push(transformer.transform(_landmark_packet(7, valid=True)))
    assert buffer.ready() is True
