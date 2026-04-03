from __future__ import annotations

from gesture_control.domain.entities import Prediction
from gesture_control.pipeline.stabilization import PredictionStabilizer


def test_prediction_stabilizer_rejects_low_confidence() -> None:
    stabilizer = PredictionStabilizer(
        default_threshold=0.85,
        class_thresholds={"swipe_right": 0.9},
        vote_window=3,
        ema_alpha=1.0,
    )

    result = stabilizer.update(
        Prediction(
            class_probs={"swipe_right": 0.82, "unknown": 0.18},
            top_class="swipe_right",
            confidence=0.82,
        )
    )

    assert result.accepted is False
    assert result.label == "unknown"


def test_prediction_stabilizer_requires_temporal_consistency() -> None:
    stabilizer = PredictionStabilizer(
        default_threshold=0.8,
        class_thresholds={"swipe_right": 0.8},
        vote_window=3,
        ema_alpha=1.0,
    )

    for _ in range(2):
        result = stabilizer.update(
            Prediction(
                class_probs={"swipe_right": 0.95, "unknown": 0.05},
                top_class="swipe_right",
                confidence=0.95,
            )
        )
        assert result.accepted is False

    result = stabilizer.update(
        Prediction(
            class_probs={"swipe_right": 0.93, "unknown": 0.07},
            top_class="swipe_right",
            confidence=0.93,
        )
    )
    assert result.accepted is True
    assert result.label == "swipe_right"


def test_prediction_stabilizer_ema_smoothing() -> None:
    stabilizer = PredictionStabilizer(
        default_threshold=0.5,
        class_thresholds={},
        vote_window=1,
        ema_alpha=0.5,
    )

    stabilizer.update(
        Prediction(
            class_probs={"swipe_left": 1.0, "unknown": 0.0},
            top_class="swipe_left",
            confidence=1.0,
        )
    )
    result = stabilizer.update(
        Prediction(
            class_probs={"swipe_left": 0.0, "unknown": 1.0},
            top_class="unknown",
            confidence=1.0,
        )
    )

    # With EMA alpha=0.5, smoothed swipe_left becomes 0.5 after abrupt drop.
    assert result.confidence >= 0.5
