from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import MagicMock

import numpy as np

from maestro.app.runtime import Runtime
from maestro.domain.entities import RuntimeState


def _make_container() -> SimpleNamespace:
    """Container whose sequence buffer is never ready — no inference path."""
    frame = np.zeros((480, 640, 3), dtype=np.uint8)

    frame_source = MagicMock()
    frame_source.read.return_value = frame

    landmark_packet = MagicMock()
    landmark_packet.valid = True
    landmark_packet.landmarks = []

    multi_packet = MagicMock()

    landmark_extractor = MagicMock()
    landmark_extractor.extract.return_value = multi_packet

    hand_selector = MagicMock()
    hand_selector.select.return_value = landmark_packet

    feature_frame = MagicMock()
    feature_frame.valid = True

    feature_transformer = MagicMock()
    feature_transformer.transform.return_value = feature_frame

    sequence_buffer = MagicMock()
    sequence_buffer.ready.return_value = False
    sequence_buffer.current_valid_ratio.return_value = 1.0

    activation_controller = MagicMock()
    activation_controller.mode = "armed"

    overlay_frame = MagicMock()
    overlay_frame.frame = frame

    overlay_renderer = MagicMock()
    overlay_renderer.render.return_value = overlay_frame

    return SimpleNamespace(
        config=SimpleNamespace(
            app=SimpleNamespace(
                app_name="test-app",
                navigation_mode="workspace",
                mirror_display=True,
            ),
            inference=SimpleNamespace(pointer=SimpleNamespace(gain_x=2.4, gain_y=2.2)),
        ),
        frame_source=frame_source,
        landmark_extractor=landmark_extractor,
        hand_selector=hand_selector,
        feature_transformer=feature_transformer,
        sequence_buffer=sequence_buffer,
        model_artifact=MagicMock(),
        stabilizer=MagicMock(),
        activation_controller=activation_controller,
        dispatcher=MagicMock(),
        overlay_renderer=overlay_renderer,
        benchmark=None,
        show_window=False,
        label_remap={},
    )


class TestOnStateCallback:
    def test_on_state_called_per_frame(self) -> None:
        """on_state is called once per processed frame."""
        n = 5
        container = _make_container()
        on_state = MagicMock()

        runtime = Runtime(container, on_state=on_state)
        runtime.run(max_frames=n)

        assert on_state.call_count == n

    def test_on_state_receives_runtime_state(self) -> None:
        """Each on_state call receives a RuntimeState instance."""
        container = _make_container()
        states: list[RuntimeState] = []

        runtime = Runtime(container, on_state=states.append)
        runtime.run(max_frames=2)

        assert len(states) == 2
        assert all(isinstance(s, RuntimeState) for s in states)

    def test_on_state_none_does_not_raise(self) -> None:
        """Runtime with on_state=None (default) behaves exactly as before."""
        container = _make_container()
        runtime = Runtime(container)  # on_state defaults to None
        runtime.run(max_frames=1)  # must not raise
