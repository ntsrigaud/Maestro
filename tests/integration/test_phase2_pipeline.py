from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np
import torch

from gesture_control.app.bootstrap import build_container
from gesture_control.domain.entities import FramePacket, LandmarkPacket
from gesture_control.infrastructure.model.checkpoint_loader import InferenceArtifact


def _stub_artifact() -> InferenceArtifact:
    """Minimal InferenceArtifact that satisfies validate_runtime_compatibility."""
    return InferenceArtifact(
        model=MagicMock(spec=torch.nn.Module),
        model_version="test",
        class_labels=["unknown"],
        feature_schema_version="test",
        feature_dim=126,
        window_length=24,
        window_step=3,
        orientation_normalization=False,
        artifact_path=Path("models/exported/stub.pt"),
    )


@patch("gesture_control.app.bootstrap.load_latest_inference_artifact")
@patch("gesture_control.infrastructure.camera.opencv_source.cv2.VideoCapture")
@patch("gesture_control.app.bootstrap.MediaPipeHandsExtractor")
def test_frame_to_landmark_pipeline(
    mock_extractor_cls: MagicMock,
    mock_capture_cls: MagicMock,
    mock_loader: MagicMock,
    config_dir,
) -> None:
    mock_loader.return_value = _stub_artifact()
    capture = MagicMock()
    capture.isOpened.return_value = True
    capture.read.return_value = (True, np.zeros((720, 1280, 3), dtype=np.uint8))
    mock_capture_cls.return_value = capture

    extractor = MagicMock()
    extractor.extract.return_value = LandmarkPacket(
        landmarks=[[0.5, 0.5, 0.0] for _ in range(21)],
        handedness="Right",
        detect_conf=0.95,
        track_conf=1.0,
        timestamp_ms=0,
        valid=True,
    )
    mock_extractor_cls.return_value = extractor

    container = build_container(str(config_dir))
    frame = container.frame_source.read()
    landmark = container.landmark_extractor.extract(frame)

    assert isinstance(frame, FramePacket)
    assert isinstance(landmark, LandmarkPacket)
    assert landmark.valid is True


@patch("gesture_control.app.bootstrap.load_latest_inference_artifact")
@patch("gesture_control.infrastructure.camera.opencv_source.cv2.VideoCapture")
@patch("gesture_control.app.bootstrap.MediaPipeHandsExtractor")
def test_pipeline_handles_no_hand(
    mock_extractor_cls: MagicMock,
    mock_capture_cls: MagicMock,
    mock_loader: MagicMock,
    config_dir,
) -> None:
    mock_loader.return_value = _stub_artifact()
    capture = MagicMock()
    capture.isOpened.return_value = True
    capture.read.return_value = (True, np.zeros((720, 1280, 3), dtype=np.uint8))
    mock_capture_cls.return_value = capture

    extractor = MagicMock()
    extractor.extract.return_value = LandmarkPacket(
        landmarks=[],
        handedness=None,
        detect_conf=0.0,
        track_conf=0.0,
        timestamp_ms=0,
        valid=False,
    )
    mock_extractor_cls.return_value = extractor

    container = build_container(str(config_dir))
    frame = container.frame_source.read()
    landmark = container.landmark_extractor.extract(frame)

    assert isinstance(frame, FramePacket)
    assert isinstance(landmark, LandmarkPacket)
    assert landmark.valid is False
