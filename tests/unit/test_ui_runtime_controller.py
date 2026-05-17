from __future__ import annotations

from unittest.mock import MagicMock, patch

from maestro.app.ui.controller import RuntimeController
from maestro.domain.entities import RuntimeState

# Patch targets — prevent real build_container / Runtime.run from executing
_PATCH_BUILD = "maestro.app.ui.controller.build_container"
_PATCH_RUNTIME = "maestro.app.ui.controller.Runtime"


def _stub_container() -> MagicMock:
    container = MagicMock()
    container.config.app.app_name = "test-app"
    container.config.app.navigation_mode = "workspace"
    container.show_window = False
    return container


class TestRuntimeControllerLifecycle:
    def test_start_spawns_thread(self, qtbot) -> None:
        """Calling start() makes is_running True then completes."""
        controller = RuntimeController()

        with (
            patch(_PATCH_BUILD, return_value=_stub_container()),
            patch(_PATCH_RUNTIME) as mock_rt,
        ):
            mock_rt.return_value.run.return_value = None

            with qtbot.waitSignal(controller.runtime_stopped, timeout=3000):
                controller.start("configs")

        assert not controller.is_running

    def test_stop_requests_runtime_stop(self, qtbot) -> None:
        """stop() calls Runtime.stop() on the live runtime."""
        controller = RuntimeController()

        import threading

        stop_event = threading.Event()

        def fake_run():
            stop_event.wait(timeout=5)

        with (
            patch(_PATCH_BUILD, return_value=_stub_container()),
            patch(_PATCH_RUNTIME) as mock_rt,
        ):
            mock_rt.return_value.run.side_effect = lambda: fake_run()
            mock_rt.return_value.stop.side_effect = stop_event.set

            controller.start("configs")
            qtbot.wait(100)
            controller.stop()
            exited = controller.wait(timeout_ms=3000)

        assert exited, "Worker thread did not exit within timeout"

    def test_runtime_started_signal_emitted(self, qtbot) -> None:
        """runtime_started signal fires after build_container() succeeds."""
        controller = RuntimeController()

        with (
            patch(_PATCH_BUILD, return_value=_stub_container()),
            patch(_PATCH_RUNTIME) as mock_rt,
        ):
            mock_rt.return_value.run.return_value = None

            with qtbot.waitSignal(controller.runtime_started, timeout=3000):
                controller.start("configs")

            controller.wait(timeout_ms=2000)

    def test_error_signal_on_build_failure(self, qtbot) -> None:
        """error_occurred carries the exception message when build fails."""
        controller = RuntimeController()

        with patch(_PATCH_BUILD, side_effect=RuntimeError("camera not found")):
            with qtbot.waitSignal(controller.error_occurred, timeout=3000) as blocker:
                controller.start("configs")

        assert "camera not found" in blocker.args[0]

    def test_state_updated_signal_delivered(self, qtbot) -> None:
        """state_updated signal delivers RuntimeState to the main thread."""
        controller = RuntimeController()
        fake_state = RuntimeState(mode="idle", active_mode="test")

        def fake_runtime_init(container, on_state=None):
            instance = MagicMock()

            def fake_run():
                if on_state is not None:
                    on_state(fake_state)

            instance.run.side_effect = fake_run
            instance.stop.return_value = None
            return instance

        with (
            patch(_PATCH_BUILD, return_value=_stub_container()),
            patch(_PATCH_RUNTIME, side_effect=fake_runtime_init),
        ):
            with qtbot.waitSignal(controller.state_updated, timeout=3000) as blocker:
                # emit_every=1 so every frame triggers state_updated
                controller.start("configs", emit_every=1)

            assert isinstance(blocker.args[0], RuntimeState)
            controller.wait(timeout_ms=2000)

    def test_double_start_is_noop(self, qtbot) -> None:
        """A second start() call while running is silently ignored."""
        controller = RuntimeController()

        import threading

        stop_event = threading.Event()

        def fake_run():
            stop_event.wait(timeout=5)

        with (
            patch(_PATCH_BUILD, return_value=_stub_container()),
            patch(_PATCH_RUNTIME) as mock_rt,
        ):
            mock_rt.return_value.run.side_effect = lambda: fake_run()
            mock_rt.return_value.stop.side_effect = stop_event.set

            controller.start("configs")
            qtbot.wait(100)
            first_thread_id = id(controller._thread)
            controller.start("configs")  # second call; should be ignored
            assert id(controller._thread) == first_thread_id

            controller.stop()
            controller.wait(timeout_ms=2000)
