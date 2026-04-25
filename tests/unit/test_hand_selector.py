"""Unit tests for HandSelector.

HandSelector is the pipeline step responsible for applying the user's
hand_preference to a MultiHandPacket and producing a single LandmarkPacket.
It owns all preference-filtering and validity logic that was previously
scattered across MediaPipeHandsExtractor.

Handedness mapping recap:
    user's right hand → MediaPipe "Left"  label (selfie-view inversion)
    user's left hand  → MediaPipe "Right" label
"""

from __future__ import annotations

from maestro.domain.entities import HandLandmarks, LandmarkPacket, MultiHandPacket
from maestro.pipeline.hand_selection import HandSelector

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _landmarks(n: int = 21) -> list[list[float]]:
    return [[0.5 + i * 0.01, 0.4, 0.0] for i in range(n)]


def _hand(label: str, *, detect_conf: float = 0.95, track_conf: float = 1.0) -> HandLandmarks:
    return HandLandmarks(
        label=label,
        landmarks=_landmarks(),
        detect_conf=detect_conf,
        track_conf=track_conf,
    )


def _packet(*hands: HandLandmarks, ts: int = 100) -> MultiHandPacket:
    return MultiHandPacket(hands=list(hands), timestamp_ms=ts)


# ---------------------------------------------------------------------------
# No detections
# ---------------------------------------------------------------------------


def test_returns_invalid_when_no_hands_detected() -> None:
    selector = HandSelector(hand_preference="both")
    result = selector.select(_packet())

    assert isinstance(result, LandmarkPacket)
    assert result.valid is False
    assert result.landmarks == []
    assert result.handedness is None
    assert result.timestamp_ms == 100


# ---------------------------------------------------------------------------
# hand_preference="both" — pick highest-confidence hand
# ---------------------------------------------------------------------------


def test_both_preference_picks_highest_confidence_hand() -> None:
    """With 'both', the hand with highest detect_conf is selected."""
    selector = HandSelector(hand_preference="both")
    result = selector.select(
        _packet(_hand("Right", detect_conf=0.80), _hand("Left", detect_conf=0.92))
    )

    assert result.valid is True
    assert result.handedness == "Left"
    assert result.detect_conf == 0.92


def test_both_preference_single_hand_is_always_selected() -> None:
    selector = HandSelector(hand_preference="both")
    result = selector.select(_packet(_hand("Right", detect_conf=0.90)))

    assert result.valid is True
    assert result.handedness == "Right"


# ---------------------------------------------------------------------------
# hand_preference="right" — user's physical right → MP "Left" label
# ---------------------------------------------------------------------------


def test_right_preference_accepts_mp_left_label() -> None:
    """User's right hand appears as MP "Left" in a non-mirrored frame."""
    selector = HandSelector(hand_preference="right")
    result = selector.select(_packet(_hand("Left")))

    assert result.valid is True
    assert result.handedness == "Left"


def test_right_preference_rejects_mp_right_label_single_hand() -> None:
    """Only user's left hand in frame → preferred hand absent → invalid."""
    selector = HandSelector(hand_preference="right")
    result = selector.select(_packet(_hand("Right")))

    assert result.valid is False
    assert result.handedness == "Right"  # fallback for diagnostics


def test_right_preference_selects_mp_left_from_two_hands() -> None:
    """When both hands present, the preferred one (MP "Left") must be selected."""
    selector = HandSelector(hand_preference="right")
    result = selector.select(_packet(_hand("Right"), _hand("Left")))

    assert result.valid is True
    assert result.handedness == "Left"


def test_right_preference_selects_second_hand_when_wrong_hand_first() -> None:
    """The preferred hand at index 1 must be chosen over the wrong hand at index 0."""
    selector = HandSelector(hand_preference="right")
    # MP returns wrong hand first
    result = selector.select(
        _packet(_hand("Right", detect_conf=0.99), _hand("Left", detect_conf=0.80))
    )

    assert result.valid is True
    assert result.handedness == "Left"


def test_right_preference_returns_invalid_when_no_mp_left_in_two_hands() -> None:
    selector = HandSelector(hand_preference="right")
    result = selector.select(_packet(_hand("Right"), _hand("Right")))

    assert result.valid is False
    assert result.handedness == "Right"  # fallback from index 0


# ---------------------------------------------------------------------------
# hand_preference="left" — user's physical left → MP "Right" label
# ---------------------------------------------------------------------------


def test_left_preference_accepts_mp_right_label() -> None:
    selector = HandSelector(hand_preference="left")
    result = selector.select(_packet(_hand("Right")))

    assert result.valid is True
    assert result.handedness == "Right"


def test_left_preference_rejects_mp_left_label_single_hand() -> None:
    selector = HandSelector(hand_preference="left")
    result = selector.select(_packet(_hand("Left")))

    assert result.valid is False
    assert result.handedness == "Left"


def test_left_preference_selects_mp_right_from_two_hands() -> None:
    selector = HandSelector(hand_preference="left")
    result = selector.select(_packet(_hand("Left"), _hand("Right")))

    assert result.valid is True
    assert result.handedness == "Right"


# ---------------------------------------------------------------------------
# Validity thresholds
# ---------------------------------------------------------------------------


def test_invalid_when_detect_conf_below_threshold() -> None:
    selector = HandSelector(hand_preference="both", min_handedness_conf=0.7)
    result = selector.select(_packet(_hand("Right", detect_conf=0.5)))

    assert result.valid is False
    assert result.detect_conf == 0.5


def test_valid_when_detect_conf_at_threshold() -> None:
    selector = HandSelector(hand_preference="both", min_handedness_conf=0.7)
    result = selector.select(_packet(_hand("Right", detect_conf=0.7)))

    assert result.valid is True


def test_invalid_when_track_conf_below_threshold() -> None:
    selector = HandSelector(hand_preference="both", min_track_conf=0.9)
    result = selector.select(_packet(_hand("Right", track_conf=0.5)))

    assert result.valid is False
    assert result.track_conf == 0.5


def test_valid_when_track_conf_at_threshold() -> None:
    selector = HandSelector(hand_preference="both", min_track_conf=0.9)
    result = selector.select(_packet(_hand("Right", track_conf=0.9)))

    assert result.valid is True


# ---------------------------------------------------------------------------
# Packet field pass-through
# ---------------------------------------------------------------------------


def test_landmarks_and_timestamp_passed_through() -> None:
    lm = [[float(i), 0.0, 0.0] for i in range(21)]
    hand = HandLandmarks(label="Right", landmarks=lm, detect_conf=0.95, track_conf=1.0)
    selector = HandSelector(hand_preference="both")
    result = selector.select(MultiHandPacket(hands=[hand], timestamp_ms=999))

    assert result.landmarks == lm
    assert result.timestamp_ms == 999


# ---------------------------------------------------------------------------
# Invalid preference value
# ---------------------------------------------------------------------------


def test_invalid_preference_raises_value_error() -> None:
    import pytest

    with pytest.raises(ValueError, match="hand_preference"):
        HandSelector(hand_preference="unknown")
