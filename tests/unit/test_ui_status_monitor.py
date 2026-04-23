from __future__ import annotations

from gesture_control.app.ui.status_monitor import StatusMonitor
from gesture_control.domain.entities import RuntimeState


class TestStatusMonitorUpdates:
    def test_on_state_updates_mode(self, qtbot) -> None:
        monitor = StatusMonitor()
        qtbot.addWidget(monitor)
        state = RuntimeState(mode="armed", gaze_permitted=True, active_mode="desktop")
        monitor.on_state(state)
        assert monitor._mode_label.text() == "ARMED"

    def test_on_state_gaze_active(self, qtbot) -> None:
        monitor = StatusMonitor()
        qtbot.addWidget(monitor)
        state = RuntimeState(mode="idle", gaze_permitted=True)
        monitor.on_state(state)
        assert "Active" in monitor._gaze_label.text()

    def test_on_state_gaze_absent(self, qtbot) -> None:
        monitor = StatusMonitor()
        qtbot.addWidget(monitor)
        state = RuntimeState(mode="idle", gaze_permitted=False)
        monitor.on_state(state)
        assert "Absent" in monitor._gaze_label.text()

    def test_on_state_updates_fps(self, qtbot) -> None:
        monitor = StatusMonitor()
        qtbot.addWidget(monitor)
        state = RuntimeState(mode="idle", info={"fps": 29.8})
        monitor.on_state(state)
        assert "29.8" in monitor._fps_label.text()

    def test_on_state_updates_confidence_bar(self, qtbot) -> None:
        monitor = StatusMonitor()
        qtbot.addWidget(monitor)
        state = RuntimeState(mode="idle", last_confidence=0.92)
        monitor.on_state(state)
        assert monitor._confidence_bar.value() == 92

    def test_missing_info_keys_do_not_raise(self, qtbot) -> None:
        """Partial info dict (e.g. absent-state) must not crash the monitor."""
        monitor = StatusMonitor()
        qtbot.addWidget(monitor)
        state = RuntimeState(mode="idle", gaze_permitted=False, info={})
        monitor.on_state(state)  # must not raise
