from __future__ import annotations

from unittest.mock import patch

from maestro.app.ui.main_window import MainWindow
from maestro.domain.entities import RuntimeState


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

    def test_tray_colour_mapping_matches_preview_badge_priority(self, qtbot) -> None:
        with (
            patch("maestro.app.ui.controller.build_container"),
            patch("maestro.app.ui.controller.Runtime"),
        ):
            window = MainWindow(config_path="configs")
            qtbot.addWidget(window)

            state_gaze_off = RuntimeState(mode="idle", info={"gaze_enabled": False})
            assert window._tray_colour_for_state(state_gaze_off) == "#1E293B"

            state_absent = RuntimeState(mode="armed", gaze_permitted=False, info={"gaze_enabled": True})
            assert window._tray_colour_for_state(state_absent) == "#4B5563"

            state_laser = RuntimeState(
                mode="armed",
                gaze_permitted=True,
                info={"gaze_enabled": True, "laser_active": True, "pen_active": False},
            )
            assert window._tray_colour_for_state(state_laser) == "#2563EB"

            state_pen = RuntimeState(
                mode="armed",
                gaze_permitted=True,
                info={"gaze_enabled": True, "laser_active": False, "pen_active": True},
            )
            assert window._tray_colour_for_state(state_pen) == "#7C3AED"

            state_armed = RuntimeState(mode="armed", gaze_permitted=True, info={"gaze_enabled": True})
            assert window._tray_colour_for_state(state_armed) == "#059669"

            state_idle = RuntimeState(mode="idle", gaze_permitted=True, info={"gaze_enabled": True})
            assert window._tray_colour_for_state(state_idle) == "#D97706"
