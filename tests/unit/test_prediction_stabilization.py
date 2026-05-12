from __future__ import annotations

from maestro.domain.entities import Prediction
from maestro.pipeline.stabilization import PredictionStabilizer


def test_prediction_stabilizer_rejects_low_confidence() -> None:
    stabilizer = PredictionStabilizer(
        default_threshold=0.85,
        class_thresholds={"swiping_right": 0.9},
        vote_window=3,
        ema_alpha=1.0,
    )

    result = stabilizer.update(
        Prediction(
            class_probs={"swiping_right": 0.82, "no_gesture": 0.18},
            top_class="swiping_right",
            confidence=0.82,
        )
    )

    assert result.accepted is False
    assert result.label == "no_gesture"


def test_prediction_stabilizer_requires_temporal_consistency() -> None:
    stabilizer = PredictionStabilizer(
        default_threshold=0.8,
        class_thresholds={"swiping_right": 0.8},
        vote_window=3,
        ema_alpha=1.0,
    )

    for _ in range(2):
        result = stabilizer.update(
            Prediction(
                class_probs={"swiping_right": 0.95, "no_gesture": 0.05},
                top_class="swiping_right",
                confidence=0.95,
            )
        )
        assert result.accepted is False

    result = stabilizer.update(
        Prediction(
            class_probs={"swiping_right": 0.93, "no_gesture": 0.07},
            top_class="swiping_right",
            confidence=0.93,
        )
    )
    assert result.accepted is True
    assert result.label == "swiping_right"


def test_prediction_stabilizer_ema_smoothing() -> None:
    stabilizer = PredictionStabilizer(
        default_threshold=0.5,
        class_thresholds={},
        vote_window=1,
        ema_alpha=0.5,
    )

    stabilizer.update(
        Prediction(
            class_probs={"swiping_left": 1.0, "no_gesture": 0.0},
            top_class="swiping_left",
            confidence=1.0,
        )
    )
    result = stabilizer.update(
        Prediction(
            class_probs={"swiping_left": 0.0, "no_gesture": 1.0},
            top_class="no_gesture",
            confidence=1.0,
        )
    )

    # With EMA alpha=0.5, smoothed swiping_left becomes 0.5 after abrupt drop.
    assert result.confidence >= 0.5


def test_ema_bypassed_for_vote_window_one_class() -> None:
    """Classes with class_vote_windows=1 use raw probs, not EMA-smoothed probs.

    With ema_alpha=0.4 and EMA state primed at no_gesture≈1.0, the smoothed value
    for swiping_right on the next frame would be 0.4 * 0.95 = 0.38 — well below
    the 0.80 threshold.  The bypass must let the raw 0.95 through so that a
    single confident frame is sufficient for acceptance.
    """
    stabilizer = PredictionStabilizer(
        default_threshold=0.8,
        class_thresholds={"swiping_right": 0.8},
        class_vote_windows={"swiping_right": 1},
        vote_window=3,
        ema_alpha=0.4,
    )

    # Prime EMA state: no_gesture is dominant.
    stabilizer.update(
        Prediction(
            class_probs={"swiping_right": 0.0, "no_gesture": 1.0},
            top_class="no_gesture",
            confidence=1.0,
        )
    )

    # One high-confidence swipe frame — must be accepted immediately.
    result = stabilizer.update(
        Prediction(
            class_probs={"swiping_right": 0.95, "no_gesture": 0.05},
            top_class="swiping_right",
            confidence=0.95,
        )
    )

    assert result.accepted is True
    assert result.label == "swiping_right"


def test_ema_still_applied_for_multi_frame_classes() -> None:
    """Classes with class_vote_windows > 1 are still EMA-smoothed.

    zoom_in has vote_window=2, so its raw probability is attenuated by EMA
    before the threshold check.  The first high-confidence frame must be
    rejected while the EMA has not yet ramped up.
    """
    stabilizer = PredictionStabilizer(
        default_threshold=0.8,
        class_thresholds={"zoom_in": 0.8},
        class_vote_windows={"zoom_in": 2},
        vote_window=3,
        ema_alpha=0.4,
    )

    # Prime EMA state: unknown is dominant.
    stabilizer.update(
        Prediction(
            class_probs={"zoom_in": 0.0, "unknown": 1.0},
            top_class="unknown",
            confidence=1.0,
        )
    )

    # Frame 2: raw 0.95, but EMA gives 0.4 * 0.95 + 0.6 * 0.0 = 0.38 < 0.80.
    result = stabilizer.update(
        Prediction(
            class_probs={"zoom_in": 0.95, "unknown": 0.05},
            top_class="zoom_in",
            confidence=0.95,
        )
    )

    assert result.accepted is False
