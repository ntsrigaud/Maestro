from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest

from maestro.app.runtime import Runtime


def _container_with_runtime_error() -> SimpleNamespace:
    frame_source = MagicMock()
    frame_source.read.side_effect = RuntimeError("camera read failure")

    gaze_detector = MagicMock()
    landmark_extractor = MagicMock()

    return SimpleNamespace(
        config=SimpleNamespace(app=SimpleNamespace(app_name="test-app")),
        frame_source=frame_source,
        gaze_detector=gaze_detector,
        landmark_extractor=landmark_extractor,
    )


def test_runtime_closes_resources_when_read_raises() -> None:
    container = _container_with_runtime_error()
    runtime = Runtime(container)

    with pytest.raises(RuntimeError, match="camera read failure"):
        runtime.run()

    container.frame_source.close.assert_called_once()
    container.gaze_detector.close.assert_called_once()
    container.landmark_extractor.close.assert_called_once()
