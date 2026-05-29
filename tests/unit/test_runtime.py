from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import MagicMock, patch

from maestro.app.runtime import Runtime
from maestro.domain.entities import StabilizedPrediction


def _container_with_runtime_error() -> SimpleNamespace:
    frame_source = MagicMock()
    frame_source.read.side_effect = RuntimeError("camera read failure")

    landmark_extractor = MagicMock()
    activation_controller = MagicMock()
    activation_controller.tick.return_value = False

    return SimpleNamespace(
        config=SimpleNamespace(app=SimpleNamespace(app_name="test-app")),
        frame_source=frame_source,
        landmark_extractor=landmark_extractor,
        activation_controller=activation_controller,
    )


def _make_minimal_container() -> SimpleNamespace:
    """Container for runtime loop tests: no inference worker, stops after one frame."""
    frame_source = MagicMock()
    # One successful frame, then consecutive errors to trigger graceful shutdown.
    frame_source.read.side_effect = [MagicMock()] + [RuntimeError("stop")] * 10

    landmark_extractor = MagicMock()
    landmark_extractor.close = MagicMock()

    hand_selector = MagicMock()
    hand_selector.select.return_value = MagicMock(valid=False, landmarks=None)

    feature_transformer = MagicMock()
    feature_transformer.transform.return_value = MagicMock()

    sequence_buffer = MagicMock()
    sequence_buffer.ready.return_value = False
    sequence_buffer.current_valid_ratio.return_value = 0.0

    activation_controller = MagicMock()
    activation_controller.tick.return_value = False
    activation_controller.mode = "armed"

    overlay_renderer = MagicMock()
    overlay_renderer.render.return_value = MagicMock(frame=None)

    return SimpleNamespace(
        config=SimpleNamespace(
            app=SimpleNamespace(app_name="test-app", mirror_display=True),
            inference=SimpleNamespace(pointer=SimpleNamespace(gain_x=2.4, gain_y=2.2)),
        ),
        frame_source=frame_source,
        landmark_extractor=landmark_extractor,
        hand_selector=hand_selector,
        feature_transformer=feature_transformer,
        sequence_buffer=sequence_buffer,
        model_artifact=None,
        activation_controller=activation_controller,
        dispatcher=MagicMock(),
        overlay_renderer=overlay_renderer,
        benchmark=None,
        show_window=False,
        label_remap={},
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


def test_runtime_calls_tick_on_activation_controller_each_frame() -> None:
    """tick() must be called on the activation controller on every successful
    frame iteration.  This guarantees the COOLDOWN → ARMED transition fires
    even when no inference results are available (hand left the camera)."""
    container = _make_minimal_container()
    runtime = Runtime(container)
    runtime.run()

    # Exactly one frame was read successfully before consecutive errors triggered
    # graceful shutdown.  tick() must have been called for that frame.
    container.activation_controller.tick.assert_called_once()


def test_runtime_resets_label_on_tick_transition() -> None:
    """When tick() returns True the state emitted for that frame must have
    last_label='unknown' (not the stale previous-gesture label)."""
    container = _make_minimal_container()
    # Simulate tick() returning True on the first (only) successful frame.
    container.activation_controller.tick.return_value = True

    states: list = []
    runtime = Runtime(container, on_state=states.append)
    runtime.run()

    assert len(states) >= 1, "Expected at least one state emission"
    assert states[0].last_label == "unknown", (
        f"Label should be 'unknown' after tick() cooldown reset, got {states[0].last_label!r}"
    )


def test_runtime_resets_continuous_label_when_both_trackers_inactive() -> None:
    """When the inference worker still emits a stale continuous gesture label
    (e.g. 'point_one') after the hand has left the frame, but neither the
    laser tracker nor the pen tracker is active, the runtime must override
    latest_label to 'unknown' before building the state payload."""
    container = _make_minimal_container()

    # Mark 'point_one' as a continuous label so the runtime knows to reset it.
    container.activation_controller.continuous_labels = frozenset({"point_one"})
    # route() must return None (continuous labels bypass dispatch).
    container.activation_controller.route.return_value = None

    # Provide a non-None artifact so the InferenceWorker is instantiated.
    container.model_artifact = MagicMock()

    # Stabiliser returns a stale 'point_one' prediction (hand just left frame).
    container.stabilizer = MagicMock()
    container.stabilizer.update.return_value = StabilizedPrediction(
        label="point_one", confidence=0.95, accepted=False, reason="stale"
    )

    states: list = []
    with patch("maestro.app.runtime._InferenceWorker") as MockWorker:
        mock_worker = MagicMock()
        MockWorker.return_value = mock_worker
        # Worker polls a result so the stabiliser path is exercised.
        mock_worker.poll.return_value = (MagicMock(), 5.0)

        runtime = Runtime(container, on_state=states.append)

        # Replace both continuous trackers with inactive mocks.
        runtime._laser_tracker = MagicMock(active=False)
        runtime._pen_tracker = MagicMock(active=False)

        runtime.run()

    assert len(states) >= 1, "Expected at least one state emission"
    assert states[0].last_label == "unknown", (
        f"Expected 'unknown' (continuous tracker inactive), got {states[0].last_label!r}"
    )
