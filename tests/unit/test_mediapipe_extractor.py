from __future__ import annotations

import math
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import numpy as np

from maestro.domain.entities import FramePacket, LandmarkPacket
from maestro.infrastructure.mediapipe.hands_extractor import (
    MediaPipeHandsExtractor,
)


def _frame_packet() -> FramePacket:
    return FramePacket(
        frame=np.zeros((720, 1280, 3), dtype=np.uint8),
        timestamp_ms=100,
        frame_id=1,
    )


def _landmarks(count: int = 21) -> list[SimpleNamespace]:
    return [SimpleNamespace(x=0.5, y=0.4, z=0.0) for _ in range(count)]


@patch("maestro.infrastructure.mediapipe.hands_extractor.mp.solutions.hands.Hands")
def test_extract_returns_invalid_when_no_hand(mock_hands_cls: MagicMock) -> None:
    hands_obj = MagicMock()
    hands_obj.process.return_value = SimpleNamespace(
        multi_hand_landmarks=None,
        multi_handedness=None,
    )
    mock_hands_cls.return_value = hands_obj

    extractor = MediaPipeHandsExtractor()
    result = extractor.extract(_frame_packet())

    assert isinstance(result, LandmarkPacket)
    assert result.valid is False
    assert result.landmarks == []
    assert result.timestamp_ms == 100


@patch("maestro.infrastructure.mediapipe.hands_extractor.mp.solutions.hands.Hands")
def test_extract_returns_valid_packet_for_detected_hand(
    mock_hands_cls: MagicMock,
) -> None:
    classification = SimpleNamespace(label="Right", score=0.98)
    handedness = SimpleNamespace(classification=[classification])
    hand_landmarks = SimpleNamespace(landmark=_landmarks())

    hands_obj = MagicMock()
    hands_obj.process.return_value = SimpleNamespace(
        multi_handedness=[handedness],
        multi_hand_landmarks=[hand_landmarks],
    )
    mock_hands_cls.return_value = hands_obj

    extractor = MediaPipeHandsExtractor()
    result = extractor.extract(_frame_packet())

    assert isinstance(result, LandmarkPacket)
    assert result.valid is True
    assert result.handedness == "Right"
    assert len(result.landmarks) == 21
    assert result.detect_conf == 0.98
    assert result.track_conf == 1.0
    assert result.timestamp_ms == 100


@patch("maestro.infrastructure.mediapipe.hands_extractor.mp.solutions.hands.Hands")
def test_extract_rejects_low_detection_confidence(mock_hands_cls: MagicMock) -> None:
    classification = SimpleNamespace(label="Left", score=0.2)
    handedness = SimpleNamespace(classification=[classification])
    hand_landmarks = SimpleNamespace(landmark=_landmarks())

    hands_obj = MagicMock()
    hands_obj.process.return_value = SimpleNamespace(
        multi_handedness=[handedness],
        multi_hand_landmarks=[hand_landmarks],
    )
    mock_hands_cls.return_value = hands_obj

    extractor = MediaPipeHandsExtractor(min_detection_confidence=0.7)
    result = extractor.extract(_frame_packet())

    assert result.valid is False


@patch("maestro.infrastructure.mediapipe.hands_extractor.mp.solutions.hands.Hands")
def test_extract_rejects_low_tracking_confidence(mock_hands_cls: MagicMock) -> None:
    classification = SimpleNamespace(label="Right", score=0.98)
    handedness = SimpleNamespace(classification=[classification])
    landmarks = _landmarks()
    landmarks[0] = SimpleNamespace(x=math.nan, y=0.4, z=0.0)
    hand_landmarks = SimpleNamespace(landmark=landmarks)

    hands_obj = MagicMock()
    hands_obj.process.return_value = SimpleNamespace(
        multi_handedness=[handedness],
        multi_hand_landmarks=[hand_landmarks],
    )
    mock_hands_cls.return_value = hands_obj

    extractor = MediaPipeHandsExtractor(min_tracking_confidence=0.99)
    result = extractor.extract(_frame_packet())

    assert result.valid is False
    assert result.detect_conf == 0.98
    assert result.track_conf < 0.99
    assert result.timestamp_ms == 100


@patch("maestro.infrastructure.mediapipe.hands_extractor.mp.solutions.hands.Hands")
def test_close_calls_mediapipe_close(mock_hands_cls: MagicMock) -> None:
    hands_obj = MagicMock()
    mock_hands_cls.return_value = hands_obj

    extractor = MediaPipeHandsExtractor()
    extractor.close()

    hands_obj.close.assert_called_once()


# ---------------------------------------------------------------------------
# Hand-preference filtering tests
# ---------------------------------------------------------------------------


def _make_hands_obj_with_label(label: str) -> MagicMock:
    """Return a mock MediaPipe Hands object that reports *label* ("Left"/"Right")."""
    classification = SimpleNamespace(label=label, score=0.98)
    handedness = SimpleNamespace(classification=[classification])
    hand_landmarks = SimpleNamespace(landmark=_landmarks())
    hands_obj = MagicMock()
    hands_obj.process.return_value = SimpleNamespace(
        multi_handedness=[handedness],
        multi_hand_landmarks=[hand_landmarks],
    )
    return hands_obj


@patch("maestro.infrastructure.mediapipe.hands_extractor.mp.solutions.hands.Hands")
def test_hand_preference_right_accepts_users_right_hand(
    mock_hands_cls: MagicMock,
) -> None:
    """preference='right' (user's right hand) must accept MediaPipe 'Left' label.

    In a non-mirrored webcam frame MediaPipe labels the user's right hand as
    'Left' (it assumes a mirrored/selfie view).  The extractor must invert the
    mapping so that the user-facing preference aligns with physical hands.
    """
    mock_hands_cls.return_value = _make_hands_obj_with_label("Left")

    extractor = MediaPipeHandsExtractor(hand_preference="right")
    result = extractor.extract(_frame_packet())

    assert result.valid is True
    assert result.handedness == "Left"


@patch("maestro.infrastructure.mediapipe.hands_extractor.mp.solutions.hands.Hands")
def test_hand_preference_right_rejects_users_left_hand(
    mock_hands_cls: MagicMock,
) -> None:
    """preference='right' must reject MediaPipe 'Right' label (user's left hand)."""
    mock_hands_cls.return_value = _make_hands_obj_with_label("Right")

    extractor = MediaPipeHandsExtractor(hand_preference="right")
    result = extractor.extract(_frame_packet())

    assert result.valid is False
    assert result.handedness == "Right"


@patch("maestro.infrastructure.mediapipe.hands_extractor.mp.solutions.hands.Hands")
def test_hand_preference_left_accepts_users_left_hand(
    mock_hands_cls: MagicMock,
) -> None:
    """preference='left' (user's left hand) must accept MediaPipe 'Right' label."""
    mock_hands_cls.return_value = _make_hands_obj_with_label("Right")

    extractor = MediaPipeHandsExtractor(hand_preference="left")
    result = extractor.extract(_frame_packet())

    assert result.valid is True
    assert result.handedness == "Right"


@patch("maestro.infrastructure.mediapipe.hands_extractor.mp.solutions.hands.Hands")
def test_hand_preference_left_rejects_users_right_hand(
    mock_hands_cls: MagicMock,
) -> None:
    """preference='left' must reject MediaPipe 'Left' label (user's right hand)."""
    mock_hands_cls.return_value = _make_hands_obj_with_label("Left")

    extractor = MediaPipeHandsExtractor(hand_preference="left")
    result = extractor.extract(_frame_packet())

    assert result.valid is False
    assert result.handedness == "Left"


@patch("maestro.infrastructure.mediapipe.hands_extractor.mp.solutions.hands.Hands")
def test_hand_preference_both_accepts_any_hand(mock_hands_cls: MagicMock) -> None:
    """When hand_preference='both', either hand is accepted (no filtering applied)."""
    mock_hands_cls.return_value = _make_hands_obj_with_label("Left")

    extractor = MediaPipeHandsExtractor(hand_preference="both")
    result = extractor.extract(_frame_packet())

    assert result.valid is True
    assert result.handedness == "Left"
