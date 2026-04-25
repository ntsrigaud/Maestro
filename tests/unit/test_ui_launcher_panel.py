from __future__ import annotations

from PySide6.QtCore import Qt

from maestro.app.ui.launcher_panel import LauncherPanel


class TestLauncherPanelStateMachine:
    def test_initial_state_is_idle(self, qtbot) -> None:
        panel = LauncherPanel()
        qtbot.addWidget(panel)
        assert panel.state == "idle"
        assert panel._btn.text() == "Start Session"
        assert panel._btn.isEnabled()

    def test_on_start_clicked_transitions_to_building(self, qtbot) -> None:
        panel = LauncherPanel()
        qtbot.addWidget(panel)
        panel.on_start_clicked()
        assert panel.state == "building"
        assert panel._btn.text() == "Starting\u2026"
        assert not panel._btn.isEnabled()

    def test_on_runtime_started_transitions_to_running(self, qtbot) -> None:
        panel = LauncherPanel()
        qtbot.addWidget(panel)
        panel.on_start_clicked()
        panel.on_runtime_started()
        assert panel.state == "running"
        assert panel._btn.text() == "Stop Session"
        assert panel._btn.isEnabled()

    def test_on_runtime_stopped_transitions_to_idle(self, qtbot) -> None:
        panel = LauncherPanel()
        qtbot.addWidget(panel)
        panel.on_start_clicked()
        panel.on_runtime_started()
        panel.on_runtime_stopped()
        assert panel.state == "idle"

    def test_on_error_transitions_to_error(self, qtbot) -> None:
        panel = LauncherPanel()
        qtbot.addWidget(panel)
        panel.on_start_clicked()
        panel.on_error("build failed")
        assert panel.state == "error"
        assert panel._btn.text() == "Retry"
        assert panel._btn.isEnabled()

    def test_start_requested_emitted_on_click_when_idle(self, qtbot) -> None:
        panel = LauncherPanel()
        qtbot.addWidget(panel)
        with qtbot.waitSignal(panel.start_requested, timeout=500):
            qtbot.mouseClick(panel._btn, Qt.LeftButton)

    def test_stop_requested_emitted_on_click_when_running(self, qtbot) -> None:
        panel = LauncherPanel()
        qtbot.addWidget(panel)
        panel.on_start_clicked()
        panel.on_runtime_started()
        with qtbot.waitSignal(panel.stop_requested, timeout=500):
            qtbot.mouseClick(panel._btn, Qt.LeftButton)

    def test_click_noop_when_building(self, qtbot) -> None:
        panel = LauncherPanel()
        qtbot.addWidget(panel)
        panel.on_start_clicked()
        # Button is disabled; no signal should fire
        with qtbot.assertNotEmitted(panel.start_requested):
            # Clicking a disabled button does not fire clicked signal in Qt
            panel._btn.click()
