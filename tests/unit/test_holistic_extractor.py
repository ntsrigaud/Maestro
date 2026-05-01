"""Unit tests for HolisticLandmarkExtractor (graph-schema-v1, Phase 1.3)."""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from maestro.domain.entities import FramePacket, HolisticPacket
from maestro.pipeline import graph_schema


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _blank_frame_packet(width: int = 320, height: int = 240, ts: int = 0) -> FramePacket:
    """Return a FramePacket with a black BGR image."""
    import cv2

    bgr = np.zeros((height, width, 3), dtype=np.uint8)
    return FramePacket(frame=bgr, timestamp_ms=ts, frame_id=1)


def _make_mock_result(
    has_right_hand: bool = True,
    has_left_hand: bool = False,
    has_pose: bool = True,
    has_face: bool = True,
) -> MagicMock:
    """Build a mock HolisticLandmarkerResult with plausible landmark lists."""

    def _lm(x: float = 0.5, y: float = 0.5, z: float = 0.0) -> MagicMock:
        lm = MagicMock()
        lm.x = x
        lm.y = y
        lm.z = z
        return lm

    result = MagicMock()

    # Right hand — 21 landmarks.
    result.right_hand_landmarks = [_lm(0.1 * i % 1.0, 0.2, 0.0) for i in range(21)] if has_right_hand else []

    # Left hand — 21 landmarks.
    result.left_hand_landmarks = [_lm(0.9 - 0.1 * i % 1.0, 0.2, 0.0) for i in range(21)] if has_left_hand else []

    # Pose — 33 landmarks (full MediaPipe Pose set).
    result.pose_landmarks = [_lm(0.5, 0.5 + 0.01 * i, 0.0) for i in range(33)] if has_pose else []

    # Face — 468 landmarks (full MediaPipe Face Mesh set).
    result.face_landmarks = [_lm(0.5 + 0.001 * i, 0.5, 0.0) for i in range(468)] if has_face else []

    return result


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


class TestHolisticLandmarkExtractorShapes:
    """Validate output tensor shapes without a real MediaPipe model."""

    def setup_method(self) -> None:
        """Patch _ensure_model and _HolisticLandmarker to skip model loading."""
        self.ensure_patcher = patch(
            "maestro.infrastructure.mediapipe.holistic_extractor._ensure_model"
        )
        self.landmarker_cls_patcher = patch(
            "maestro.infrastructure.mediapipe.holistic_extractor._HolisticLandmarker"
        )

        self.ensure_patcher.start()
        mock_cls = self.landmarker_cls_patcher.start()

        # Mock landmarker instance returned by create_from_options.
        self.mock_landmarker = MagicMock()
        mock_cls.create_from_options.return_value = self.mock_landmarker

    def teardown_method(self) -> None:
        self.ensure_patcher.stop()
        self.landmarker_cls_patcher.stop()

    def _make_extractor(self):
        from maestro.infrastructure.mediapipe.holistic_extractor import HolisticLandmarkExtractor

        return HolisticLandmarkExtractor(model_path="/fake/model.task")

    def test_packet_shapes_all_detected(self) -> None:
        """Output shapes must be (67, 3) and (67,) when all body parts detected."""
        self.mock_landmarker.detect_for_video.return_value = _make_mock_result(
            has_right_hand=True, has_left_hand=True, has_pose=True, has_face=True
        )
        extractor = self._make_extractor()
        packet = extractor.extract(_blank_frame_packet())

        assert isinstance(packet, HolisticPacket)
        assert packet.keypoints.shape == (graph_schema.NUM_NODES, graph_schema.NUM_COORDS)
        assert packet.valid_mask.shape == (graph_schema.NUM_NODES,)
        assert packet.keypoints.dtype == np.float32
        assert packet.valid_mask.dtype == bool

    def test_packet_shapes_blank_detection(self) -> None:
        """Shapes must be correct even when nothing is detected (all zeros)."""
        self.mock_landmarker.detect_for_video.return_value = _make_mock_result(
            has_right_hand=False, has_left_hand=False, has_pose=False, has_face=False
        )
        extractor = self._make_extractor()
        packet = extractor.extract(_blank_frame_packet())

        assert packet.keypoints.shape == (graph_schema.NUM_NODES, graph_schema.NUM_COORDS)
        assert packet.valid_mask.shape == (graph_schema.NUM_NODES,)
        # No landmarks detected — all mask entries should be False.
        assert not packet.valid_mask.any()

    def test_valid_mask_right_hand_only(self) -> None:
        """With only right hand detected, nodes 0–20 should be valid."""
        self.mock_landmarker.detect_for_video.return_value = _make_mock_result(
            has_right_hand=True, has_left_hand=False, has_pose=False, has_face=False
        )
        extractor = self._make_extractor()
        packet = extractor.extract(_blank_frame_packet())

        assert packet.valid_mask[0:21].all(), "Right-hand nodes 0–20 should be valid"
        assert not packet.valid_mask[21:42].any(), "Left-hand nodes 21–41 should be invalid"

    def test_valid_mask_left_hand_only(self) -> None:
        """With only left hand detected, nodes 21–41 should be valid."""
        self.mock_landmarker.detect_for_video.return_value = _make_mock_result(
            has_right_hand=False, has_left_hand=True, has_pose=False, has_face=False
        )
        extractor = self._make_extractor()
        packet = extractor.extract(_blank_frame_packet())

        assert not packet.valid_mask[0:21].any(), "Right-hand nodes should be invalid"
        assert packet.valid_mask[21:42].all(), "Left-hand nodes 21–41 should be valid"

    def test_timestamp_propagated(self) -> None:
        """HolisticPacket must carry the same timestamp_ms as the FramePacket."""
        self.mock_landmarker.detect_for_video.return_value = _make_mock_result()
        extractor = self._make_extractor()
        ts = 123456
        packet = extractor.extract(_blank_frame_packet(ts=ts))

        assert packet.timestamp_ms == ts

    def test_close_calls_landmarker(self) -> None:
        """close() should delegate to the underlying landmarker."""
        extractor = self._make_extractor()
        extractor.close()
        self.mock_landmarker.close.assert_called_once()
