from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np

from maestro.app.bootstrap import build_container
from maestro.domain.entities import FramePacket, LandmarkPacket
from maestro.infrastructure.model.checkpoint_loader import InferenceArtifact
from maestro.infrastructure.model.two_stream_lstm import TwoStreamGestureLSTM


def _mock_artifact() -> InferenceArtifact:
    """Create a dummy InferenceArtifact matching the current feature-schema-v5 config."""
    model = TwoStreamGestureLSTM(
        input_size=147,
        hidden_size=64,
        num_layers=2,
        dropout=0.1,
        num_classes=8,
    )
    return InferenceArtifact(
        model=model,
        model_version="test_v5",
        class_labels=[
            "swipe_up",
            "swipe_down",
            "swipe_left",
            "swipe_right",
            "open_palm_hold",
            "zoom_in",
            "zoom_out",
            "unknown",
        ],
        feature_schema_version="feature-schema-v5",
        feature_dim=147,
        window_length=24,
        window_step=3,
        orientation_normalization=False,
        artifact_path=Path("/dev/null"),
    )


@patch("maestro.app.bootstrap._load_model_from_source")
@patch("maestro.app.bootstrap.validate_runtime_compatibility")
@patch("maestro.infrastructure.camera.opencv_source.cv2.VideoCapture")
@patch("maestro.app.bootstrap.MediaPipeHandsExtractor")
def test_frame_to_landmark_pipeline(
    mock_extractor_cls: MagicMock,
    mock_capture_cls: MagicMock,
    _mock_validate: MagicMock,
    mock_load_artifact: MagicMock,
    config_dir,
) -> None:
    mock_load_artifact.return_value = _mock_artifact()

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


@patch("maestro.app.bootstrap._load_model_from_source")
@patch("maestro.app.bootstrap.validate_runtime_compatibility")
@patch("maestro.infrastructure.camera.opencv_source.cv2.VideoCapture")
@patch("maestro.app.bootstrap.MediaPipeHandsExtractor")
def test_pipeline_handles_no_hand(
    mock_extractor_cls: MagicMock,
    mock_capture_cls: MagicMock,
    _mock_validate: MagicMock,
    mock_load_artifact: MagicMock,
    config_dir,
) -> None:
    mock_load_artifact.return_value = _mock_artifact()

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
