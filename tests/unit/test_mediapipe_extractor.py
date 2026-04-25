"""Unit tests for MediaPipeHandsExtractor.

The extractor is now preference-agnostic: it always runs MediaPipe with
max_num_hands=2 and returns a MultiHandPacket containing every detected hand.
Hand-preference selection and validity checking are the responsibility of
HandSelector (tested in test_hand_selector.py).
"""

from __future__ import annotations

import math
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from maestro.domain.entities import FramePacket, HandLandmarks, MultiHandPacket
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


def _make_mp_hand(label: str, score: float = 0.95) -> tuple[SimpleNamespace, SimpleNamespace]:
    """Return a (handedness_proto, landmark_proto) pair for mocking."""
    classification = SimpleNamespace(label=label, score=score)
    handedness = SimpleNamespace(classification=[classification])
    hand_landmarks = SimpleNamespace(landmark=_landmarks())
    return handedness, hand_landmarks


# ---------------------------------------------------------------------------
# Return type
# ---------------------------------------------------------------------------


@patch("maestro.infrastructure.mediapipe.hands_extractor.mp.solutions.hands.Hands")
def test_extract_always_returns_multi_hand_packet(mock_hands_cls: MagicMock) -> None:
    hands_obj = MagicMock()
    hands_obj.process.return_value = SimpleNamespace(
        multi_hand_landmarks=None, multi_handedness=None
    )
    mock_hands_cls.return_value = hands_obj

    extractor = MediaPipeHandsExtractor()
    result = extractor.extract(_frame_packet())

    assert isinstance(result, MultiHandPacket)


# ---------------------------------------------------------------------------
# No detections
# ---------------------------------------------------------------------------


@patch("maestro.infrastructure.mediapipe.hands_extractor.mp.solutions.hands.Hands")
def test_extract_returns_empty_hands_when_no_detection(mock_hands_cls: MagicMock) -> None:
    hands_obj = MagicMock()
    hands_obj.process.return_value = SimpleNamespace(
        multi_hand_landmarks=None, multi_handedness=None
    )
    mock_hands_cls.return_value = hands_obj

    extractor = MediaPipeHandsExtractor()
    result = extractor.extract(_frame_packet())

    assert result.hands == []
    assert result.timestamp_ms == 100


# ---------------------------------------------------------------------------
# Single detection
# ---------------------------------------------------------------------------


@patch("maestro.infrastructure.mediapipe.hands_extractor.mp.solutions.hands.Hands")
def test_extract_single_hand_populates_hand_landmarks(mock_hands_cls: MagicMock) -> None:
    h, lm = _make_mp_hand("Right", score=0.92)
    hands_obj = MagicMock()
    hands_obj.process.return_value = SimpleNamespace(
        multi_handedness=[h], multi_hand_landmarks=[lm]
    )
    mock_hands_cls.return_value = hands_obj

    extractor = MediaPipeHandsExtractor()
    result = extractor.extract(_frame_packet())

    assert len(result.hands) == 1
    hand = result.hands[0]
    assert isinstance(hand, HandLandmarks)
    assert hand.label == "Right"
    assert hand.detect_conf == 0.92
    assert hand.track_conf == 1.0
    assert len(hand.landmarks) == 21
    assert result.timestamp_ms == 100


@patch("maestro.infrastructure.mediapipe.hands_extractor.mp.solutions.hands.Hands")
def test_extract_preserves_landmark_coordinates(mock_hands_cls: MagicMock) -> None:
    h, _ = _make_mp_hand("Left", score=0.88)
    pt = SimpleNamespace(x=0.25, y=0.75, z=0.1)
    lm = SimpleNamespace(landmark=[pt] + _landmarks(20))
    hands_obj = MagicMock()
    hands_obj.process.return_value = SimpleNamespace(
        multi_handedness=[h], multi_hand_landmarks=[lm]
    )
    mock_hands_cls.return_value = hands_obj

    result = MediaPipeHandsExtractor().extract(_frame_packet())

    assert result.hands[0].landmarks[0] == pytest.approx([0.25, 0.75, 0.1])


# ---------------------------------------------------------------------------
# Two detections
# ---------------------------------------------------------------------------


@patch("maestro.infrastructure.mediapipe.hands_extractor.mp.solutions.hands.Hands")
def test_extract_returns_both_hands_when_two_detected(mock_hands_cls: MagicMock) -> None:
    h0, lm0 = _make_mp_hand("Right", score=0.91)
    h1, lm1 = _make_mp_hand("Left", score=0.87)
    hands_obj = MagicMock()
    hands_obj.process.return_value = SimpleNamespace(
        multi_handedness=[h0, h1], multi_hand_landmarks=[lm0, lm1]
    )
    mock_hands_cls.return_value = hands_obj

    result = MediaPipeHandsExtractor().extract(_frame_packet())

    assert len(result.hands) == 2
    labels = [h.label for h in result.hands]
    assert "Right" in labels
    assert "Left" in labels


@patch("maestro.infrastructure.mediapipe.hands_extractor.mp.solutions.hands.Hands")
def test_extract_preserves_detection_order(mock_hands_cls: MagicMock) -> None:
    """Hands must appear in result.hands in the same order MediaPipe returns them."""
    h0, lm0 = _make_mp_hand("Right", score=0.93)
    h1, lm1 = _make_mp_hand("Left", score=0.80)
    hands_obj = MagicMock()
    hands_obj.process.return_value = SimpleNamespace(
        multi_handedness=[h0, h1], multi_hand_landmarks=[lm0, lm1]
    )
    mock_hands_cls.return_value = hands_obj

    result = MediaPipeHandsExtractor().extract(_frame_packet())

    assert result.hands[0].label == "Right"
    assert result.hands[1].label == "Left"


# ---------------------------------------------------------------------------
# track_conf / NaN handling
# ---------------------------------------------------------------------------


@patch("maestro.infrastructure.mediapipe.hands_extractor.mp.solutions.hands.Hands")
def test_track_conf_is_fraction_of_finite_landmarks(mock_hands_cls: MagicMock) -> None:
    h, _ = _make_mp_hand("Right", score=0.95)
    # 20 valid + 1 NaN → track_conf should be 20/21
    lm_pts = _landmarks(21)
    lm_pts[0] = SimpleNamespace(x=math.nan, y=0.4, z=0.0)
    lm = SimpleNamespace(landmark=lm_pts)

    hands_obj = MagicMock()
    hands_obj.process.return_value = SimpleNamespace(
        multi_handedness=[h], multi_hand_landmarks=[lm]
    )
    mock_hands_cls.return_value = hands_obj

    result = MediaPipeHandsExtractor().extract(_frame_packet())

    assert result.hands[0].track_conf == pytest.approx(20 / 21)


# ---------------------------------------------------------------------------
# Constructor / MediaPipe configuration
# ---------------------------------------------------------------------------


@patch("maestro.infrastructure.mediapipe.hands_extractor.mp.solutions.hands.Hands")
def test_always_uses_max_num_hands_two(mock_hands_cls: MagicMock) -> None:
    """The extractor must always request max_num_hands=2 so all detected hands
    are available for downstream preference selection.
    """
    hands_obj = MagicMock()
    hands_obj.process.return_value = SimpleNamespace(
        multi_hand_landmarks=None, multi_handedness=None
    )
    mock_hands_cls.return_value = hands_obj

    MediaPipeHandsExtractor()

    _, kwargs = mock_hands_cls.call_args
    assert kwargs["max_num_hands"] == 2


# ---------------------------------------------------------------------------
# BGR→RGB conversion
# ---------------------------------------------------------------------------


@patch("maestro.infrastructure.mediapipe.hands_extractor.mp.solutions.hands.Hands")
def test_bgr_to_rgb_uses_cvtcolor(mock_hands_cls: MagicMock) -> None:
    """extract() must use cv2.cvtColor to produce a C-contiguous RGB copy."""
    import cv2 as cv2_real

    hands_obj = MagicMock()
    hands_obj.process.return_value = SimpleNamespace(
        multi_hand_landmarks=None, multi_handedness=None
    )
    mock_hands_cls.return_value = hands_obj

    with patch(
        "maestro.infrastructure.mediapipe.hands_extractor.cv2.cvtColor"
    ) as mock_cvt:
        mock_cvt.return_value = _frame_packet().frame
        MediaPipeHandsExtractor().extract(_frame_packet())

    mock_cvt.assert_called_once()
    assert mock_cvt.call_args[0][1] == cv2_real.COLOR_BGR2RGB


# ---------------------------------------------------------------------------
# Lifecycle
# ---------------------------------------------------------------------------


@patch("maestro.infrastructure.mediapipe.hands_extractor.mp.solutions.hands.Hands")
def test_close_calls_mediapipe_close(mock_hands_cls: MagicMock) -> None:
    hands_obj = MagicMock()
    mock_hands_cls.return_value = hands_obj

    extractor = MediaPipeHandsExtractor()
    extractor.close()

    hands_obj.close.assert_called_once()

