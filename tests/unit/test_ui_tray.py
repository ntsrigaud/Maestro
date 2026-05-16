from __future__ import annotations

from unittest.mock import patch

from maestro.app.ui.main_window import MainWindow


class _RunningThread:
    def isRunning(self) -> bool:
        return True

    def stop(self) -> None:
        pass

    def wait(self, _timeout: int = 0) -> bool:
        return True


class TestSystemTray:
    def test_tray_created_and_visible(self, qtbot) -> None:
        with (
            patch("maestro.app.ui.controller.build_container"),
            patch("maestro.app.ui.controller.Runtime"),
        ):
            window = MainWindow(config_path="configs")
            qtbot.addWidget(window)
            assert window._tray.isVisible()

    def test_tray_icon_is_qicon(self, qtbot) -> None:
        with (
            patch("maestro.app.ui.controller.build_container"),
            patch("maestro.app.ui.controller.Runtime"),
        ):
            window = MainWindow(config_path="configs")
            qtbot.addWidget(window)
            icon = window._tray.icon()
            assert not icon.isNull()

    def test_tray_menu_exposes_active_and_preview_actions(self, qtbot) -> None:
        with (
            patch("maestro.app.ui.controller.build_container"),
            patch("maestro.app.ui.controller.Runtime"),
        ):
            window = MainWindow(config_path="configs")
            qtbot.addWidget(window)

            actions = window._tray.contextMenu().actions()
            assert actions[0].text() == "Active"
            assert actions[1].text() == "Preview"

    def test_preview_toggle_shows_overlay_when_runtime_running(self, qtbot) -> None:
        with (
            patch("maestro.app.ui.controller.build_container"),
            patch("maestro.app.ui.controller.Runtime"),
        ):
            window = MainWindow(config_path="configs")
            qtbot.addWidget(window)

            window._controller._thread = _RunningThread()
            window._controller.runtime_started.emit()
            window._preview_action.setChecked(True)

            assert window._preview_window.isVisible()
