from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import numpy as np

from gesture_control.domain.entities import FramePacket, LandmarkPacket
from gesture_control.infrastructure.mediapipe.hands_extractor import MediaPipeHandsExtractor


def _frame_packet() -> FramePacket:
    return FramePacket(
        frame=np.zeros((720, 1280, 3), dtype=np.uint8),
        timestamp_ms=100,
        frame_id=1,
    )


def _landmarks(count: int = 21) -> list[SimpleNamespace]:
    return [SimpleNamespace(x=0.5, y=0.4, z=0.0) for _ in range(count)]


@patch("gesture_control.infrastructure.mediapipe.hands_extractor.mp.solutions.hands.Hands")
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


@patch("gesture_control.infrastructure.mediapipe.hands_extractor.mp.solutions.hands.Hands")
def test_extract_returns_valid_packet_for_detected_hand(mock_hands_cls: MagicMock) -> None:
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


@patch("gesture_control.infrastructure.mediapipe.hands_extractor.mp.solutions.hands.Hands")
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


@patch("gesture_control.infrastructure.mediapipe.hands_extractor.mp.solutions.hands.Hands")
def test_close_calls_mediapipe_close(mock_hands_cls: MagicMock) -> None:
    hands_obj = MagicMock()
    mock_hands_cls.return_value = hands_obj

    extractor = MediaPipeHandsExtractor()
    extractor.close()

    hands_obj.close.assert_called_once()