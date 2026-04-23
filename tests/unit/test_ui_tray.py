from __future__ import annotations

from unittest.mock import MagicMock, patch

from PySide6.QtWidgets import QApplication

from gesture_control.app.ui.main_window import MainWindow


class TestSystemTray:
    def test_tray_created_and_visible(self, qtbot) -> None:
        with (
            patch("gesture_control.app.ui.controller.build_container"),
            patch("gesture_control.app.ui.controller.Runtime"),
        ):
            window = MainWindow(config_path="configs")
            qtbot.addWidget(window)
            assert window._tray.isVisible()

    def test_tray_icon_is_qicon(self, qtbot) -> None:
        with (
            patch("gesture_control.app.ui.controller.build_container"),
            patch("gesture_control.app.ui.controller.Runtime"),
        ):
            window = MainWindow(config_path="configs")
            qtbot.addWidget(window)
            icon = window._tray.icon()
            assert not icon.isNull()


class TestLauncherPanelShowWindow:
    def test_show_window_defaults_false(self, qtbot) -> None:
        from gesture_control.app.ui.launcher_panel import LauncherPanel

        panel = LauncherPanel()
        qtbot.addWidget(panel)
        assert panel.show_window is False

    def test_show_window_reflects_checkbox(self, qtbot) -> None:
        from gesture_control.app.ui.launcher_panel import LauncherPanel

        panel = LauncherPanel()
        qtbot.addWidget(panel)
        panel._show_window_cb.setChecked(True)
        assert panel.show_window is True
