from __future__ import annotations

import math

from gesture_control.domain.entities import LandmarkPacket
from gesture_control.pipeline.feature_engineering import LandmarkFeatureTransformer


def _landmark_packet(
    *,
    landmarks: list[list[float]],
    timestamp_ms: int,
    valid: bool = True,
) -> LandmarkPacket:
    return LandmarkPacket(
        landmarks=landmarks,
        handedness="Right" if valid else None,
        detect_conf=0.95 if valid else 0.0,
        track_conf=1.0 if valid else 0.0,
        timestamp_ms=timestamp_ms,
        valid=valid,
    )


def _make_hand(
    base_x: float = 1.0, base_y: float = 2.0, scale: float = 1.0
) -> list[list[float]]:
    points: list[list[float]] = []
    for i in range(21):
        x = base_x + scale * (i * 0.01)
        y = base_y + scale * (i * 0.02)
        z = scale * (i * 0.005)
        points.append([x, y, z])
    return points


def test_transform_returns_zero_vector_for_invalid_packet() -> None:
    transformer = LandmarkFeatureTransformer()
    packet = _landmark_packet(landmarks=[], timestamp_ms=100, valid=False)

    result = transformer.transform(packet)

    assert result.valid is False
    assert len(result.vector) == 126
    assert all(value == 0.0 for value in result.vector)


def test_translation_invariance_after_normalization() -> None:
    transformer = LandmarkFeatureTransformer()

    base = _make_hand(base_x=0.2, base_y=0.3, scale=0.5)
    shifted = [[x + 10.0, y - 7.0, z + 2.0] for x, y, z in base]

    base_result = transformer.transform(
        _landmark_packet(landmarks=base, timestamp_ms=100)
    )
    shifted_result = transformer.transform(
        _landmark_packet(landmarks=shifted, timestamp_ms=133)
    )

    base_pos = base_result.vector[:63]
    shifted_pos = shifted_result.vector[:63]

    for a, b in zip(base_pos, shifted_pos, strict=True):
        assert math.isclose(a, b, rel_tol=1e-6, abs_tol=1e-6)


def test_scale_invariance_after_normalization() -> None:
    transformer = LandmarkFeatureTransformer()

    small = _make_hand(base_x=0.5, base_y=0.7, scale=0.5)
    large = _make_hand(base_x=0.5, base_y=0.7, scale=2.0)

    small_result = transformer.transform(
        _landmark_packet(landmarks=small, timestamp_ms=100)
    )
    large_result = transformer.transform(
        _landmark_packet(landmarks=large, timestamp_ms=133)
    )

    small_pos = small_result.vector[:63]
    large_pos = large_result.vector[:63]

    for a, b in zip(small_pos, large_pos, strict=True):
        assert math.isclose(a, b, rel_tol=1e-6, abs_tol=1e-6)


def test_velocity_features_present_on_subsequent_frames() -> None:
    transformer = LandmarkFeatureTransformer()

    first = _make_hand(base_x=0.0, base_y=0.0, scale=1.0)
    second = [[x + 0.02, y + 0.01, z] for x, y, z in first]

    result1 = transformer.transform(_landmark_packet(landmarks=first, timestamp_ms=100))
    result2 = transformer.transform(
        _landmark_packet(landmarks=second, timestamp_ms=200)
    )

    assert all(value == 0.0 for value in result1.vector[63:])
    assert any(abs(value) > 0.0 for value in result2.vector[63:])


def test_non_finite_landmark_marks_feature_invalid() -> None:
    transformer = LandmarkFeatureTransformer()

    points = _make_hand(base_x=1.0, base_y=1.0, scale=1.0)
    points[3][0] = math.nan

    result = transformer.transform(_landmark_packet(landmarks=points, timestamp_ms=100))

    assert result.valid is False
    assert all(value == 0.0 for value in result.vector)
