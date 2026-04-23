from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import MagicMock

import numpy as np

from gesture_control.app.runtime import Runtime
from gesture_control.domain.entities import RuntimeState


# ---------------------------------------------------------------------------
# Minimal container factories
# ---------------------------------------------------------------------------

def _make_gaze_permitted_container() -> SimpleNamespace:
    """Container whose gaze detector always permits and whose sequence buffer
    is never ready — so no inference runs, just the full state path."""
    frame = np.zeros((480, 640, 3), dtype=np.uint8)

    frame_source = MagicMock()
    frame_source.read.return_value = frame

    gaze_detector = MagicMock()
    gaze_detector.update.return_value = True

    landmark_packet = MagicMock()
    landmark_packet.valid = True
    landmark_packet.landmarks = []

    landmark_extractor = MagicMock()
    landmark_extractor.extract.return_value = landmark_packet

    feature_frame = MagicMock()
    feature_frame.valid = True

    feature_transformer = MagicMock()
    feature_transformer.transform.return_value = feature_frame

    sequence_buffer = MagicMock()
    sequence_buffer.ready.return_value = False
    sequence_buffer.current_valid_ratio.return_value = 1.0

    activation_controller = MagicMock()
    activation_controller.mode = "idle"

    overlay_frame = MagicMock()
    overlay_frame.frame = frame

    overlay_renderer = MagicMock()
    overlay_renderer.render.return_value = overlay_frame

    return SimpleNamespace(
        config=SimpleNamespace(
            app=SimpleNamespace(app_name="test-app", navigation_mode="workspace"),
            inference=SimpleNamespace(gaze=SimpleNamespace(sample_interval_frames=1)),
        ),
        frame_source=frame_source,
        gaze_detector=gaze_detector,
        landmark_extractor=landmark_extractor,
        feature_transformer=feature_transformer,
        sequence_buffer=sequence_buffer,
        model_artifact=MagicMock(),
        stabilizer=MagicMock(),
        activation_controller=activation_controller,
        dispatcher=MagicMock(),
        overlay_renderer=overlay_renderer,
        benchmark=None,
        show_window=False,
    )


def _make_gaze_absent_container() -> SimpleNamespace:
    """Container whose gaze detector always blocks — exercises the absent branch."""
    frame = np.zeros((480, 640, 3), dtype=np.uint8)

    frame_source = MagicMock()
    frame_source.read.return_value = frame

    gaze_detector = MagicMock()
    gaze_detector.update.return_value = False

    overlay_frame = MagicMock()
    overlay_frame.frame = frame

    overlay_renderer = MagicMock()
    overlay_renderer.render.return_value = overlay_frame

    activation_controller = MagicMock()
    activation_controller.mode = "idle"

    return SimpleNamespace(
        config=SimpleNamespace(
            app=SimpleNamespace(app_name="test-app", navigation_mode="workspace"),
            inference=SimpleNamespace(gaze=SimpleNamespace(sample_interval_frames=1)),
        ),
        frame_source=frame_source,
        gaze_detector=gaze_detector,
        landmark_extractor=MagicMock(),
        feature_transformer=MagicMock(),
        sequence_buffer=MagicMock(),
        model_artifact=MagicMock(),
        stabilizer=MagicMock(),
        activation_controller=activation_controller,
        dispatcher=MagicMock(),
        overlay_renderer=overlay_renderer,
        benchmark=None,
        show_window=False,
    )


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

class TestOnStateCallback:
    def test_on_state_called_per_frame_when_gaze_permitted(self) -> None:
        """on_state is called once per processed frame (gaze path)."""
        n = 5
        container = _make_gaze_permitted_container()
        on_state = MagicMock()

        runtime = Runtime(container, on_state=on_state)
        runtime.run(max_frames=n)

        assert on_state.call_count == n

    def test_on_state_called_per_frame_when_gaze_absent(self) -> None:
        """on_state is called once per processed frame even in the absent branch."""
        on_state = MagicMock()
        container = _make_gaze_absent_container()

        runtime = Runtime(container, on_state=on_state)
        runtime.run(max_frames=3)

        assert on_state.call_count == 3

    def test_on_state_receives_runtime_state(self) -> None:
        """Each on_state call receives a RuntimeState instance."""
        container = _make_gaze_permitted_container()
        states: list[RuntimeState] = []

        runtime = Runtime(container, on_state=states.append)
        runtime.run(max_frames=2)

        assert len(states) == 2
        assert all(isinstance(s, RuntimeState) for s in states)

    def test_on_state_none_does_not_raise(self) -> None:
        """Runtime with on_state=None (default) behaves exactly as before."""
        container = _make_gaze_permitted_container()
        runtime = Runtime(container)  # on_state defaults to None
        runtime.run(max_frames=1)  # must not raise

    def test_gaze_absent_state_has_gaze_permitted_false(self) -> None:
        """States produced in the gaze-absent branch carry gaze_permitted=False."""
        container = _make_gaze_absent_container()
        states: list[RuntimeState] = []

        runtime = Runtime(container, on_state=states.append)
        runtime.run(max_frames=2)

        assert all(not s.gaze_permitted for s in states)
