from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import MagicMock

from maestro.app.runtime import Runtime


def _container_with_runtime_error() -> SimpleNamespace:
    frame_source = MagicMock()
    frame_source.read.side_effect = RuntimeError("camera read failure")

    landmark_extractor = MagicMock()
    activation_controller = MagicMock()

    return SimpleNamespace(
        config=SimpleNamespace(app=SimpleNamespace(app_name="test-app")),
        frame_source=frame_source,
        landmark_extractor=landmark_extractor,
        activation_controller=activation_controller,
    )


def test_runtime_closes_resources_when_read_raises() -> None:
    """After _MAX_CONSECUTIVE_FRAME_ERRORS consecutive read failures the runtime
    stops gracefully (no exception propagated) and still closes all resources."""
    container = _container_with_runtime_error()
    runtime = Runtime(container)

    # Should exit cleanly — no RuntimeError propagates after the consecutive
    # failure threshold is reached.
    runtime.run()

    container.frame_source.close.assert_called_once()
    container.landmark_extractor.close.assert_called_once()
