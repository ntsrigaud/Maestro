from __future__ import annotations

from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from maestro.domain.entities import FramePacket
from maestro.infrastructure.camera.opencv_source import OpenCVFrameSource


def _mock_capture_with_frame(width: int = 1280, height: int = 720) -> MagicMock:
    capture = MagicMock()
    capture.isOpened.return_value = True
    capture.read.return_value = (True, np.zeros((height, width, 3), dtype=np.uint8))
    return capture


@patch("maestro.infrastructure.camera.opencv_source.cv2.VideoCapture")
def test_init_raises_when_camera_unavailable(mock_capture_cls: MagicMock) -> None:
    capture = MagicMock()
    capture.isOpened.return_value = False
    mock_capture_cls.return_value = capture

    with pytest.raises(RuntimeError, match="Failed to open camera"):
        OpenCVFrameSource(camera_index=9)


@patch("maestro.infrastructure.camera.opencv_source.cv2.VideoCapture")
def test_read_returns_frame_packet(mock_capture_cls: MagicMock) -> None:
    mock_capture_cls.return_value = _mock_capture_with_frame()
    source = OpenCVFrameSource(target_fps=30)

    packet = source.read()

    assert isinstance(packet, FramePacket)
    assert packet.frame.shape == (720, 1280, 3)
    assert packet.frame_id == 0
    assert packet.timestamp_ms > 0


@patch("maestro.infrastructure.camera.opencv_source.cv2.VideoCapture")
def test_frame_id_increments(mock_capture_cls: MagicMock) -> None:
    mock_capture_cls.return_value = _mock_capture_with_frame()
    source = OpenCVFrameSource(target_fps=30)

    first = source.read()
    second = source.read()

    assert first.frame_id == 0
    assert second.frame_id == 1


@patch("maestro.infrastructure.camera.opencv_source.cv2.VideoCapture")
def test_read_raises_when_capture_fails(mock_capture_cls: MagicMock) -> None:
    capture = MagicMock()
    capture.isOpened.return_value = True
    capture.read.return_value = (False, None)
    mock_capture_cls.return_value = capture
    source = OpenCVFrameSource(target_fps=30)

    with pytest.raises(RuntimeError, match="Failed to capture frame"):
        source.read()


@patch("maestro.infrastructure.camera.opencv_source.cv2.VideoCapture")
def test_close_releases_camera(mock_capture_cls: MagicMock) -> None:
    capture = _mock_capture_with_frame()
    mock_capture_cls.return_value = capture
    source = OpenCVFrameSource(target_fps=30)

    source.close()

    capture.release.assert_called_once()
