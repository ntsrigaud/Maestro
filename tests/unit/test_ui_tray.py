from __future__ import annotations

import os
import sys
from unittest.mock import patch

import pytest

from maestro.app.ui.main_window import MainWindow
from maestro.domain.entities import RuntimeState


class _RunningThread:
    def isRunning(self) -> bool:
        return True

    def stop(self) -> None:
        pass

    def wait(self, _timeout: int = 0) -> bool:
        return True


@pytest.mark.skipif(bool(os.getenv("CI")), reason="UI tray/window tests are skipped in CI")
class TestSystemTray:
    def test_control_surface_window_is_created(self, qtbot) -> None:
        with (
            patch("maestro.app.ui.controller.build_container"),
            patch("maestro.app.ui.controller.Runtime"),
        ):
            window = MainWindow(config_path="configs")
            qtbot.addWidget(window)
            # On macOS the widget window is hidden; the tray is the sole control surface.
            expected_visible = sys.platform != "darwin"
            assert window.isVisible() == expected_visible

    def test_window_icon_is_qicon(self, qtbot) -> None:
        with (
            patch("maestro.app.ui.controller.build_container"),
            patch("maestro.app.ui.controller.Runtime"),
        ):
            window = MainWindow(config_path="configs")
            qtbot.addWidget(window)
            icon = window.windowIcon()
            assert not icon.isNull()

    def test_controls_menu_exposes_active_and_preview_actions(self, qtbot) -> None:
        with (
            patch("maestro.app.ui.controller.build_container"),
            patch("maestro.app.ui.controller.Runtime"),
        ):
            window = MainWindow(config_path="configs")
            qtbot.addWidget(window)

            actions = window._controls_menu.actions()
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

            state_laser = RuntimeState(
                mode="armed",
                info={"laser_active": True, "pen_active": False},
            )
            assert window._tray_colour_for_state(state_laser) == "#2563EB"

            state_pen = RuntimeState(
                mode="armed",
                info={"laser_active": False, "pen_active": True},
            )
            assert window._tray_colour_for_state(state_pen) == "#7C3AED"

            state_armed = RuntimeState(mode="armed")
            assert window._tray_colour_for_state(state_armed) == "#059669"

            state_cooldown = RuntimeState(mode="cooldown")
            assert window._tray_colour_for_state(state_cooldown) == "#D97706"

    def test_window_controls_remain_available_without_system_tray(self, qtbot) -> None:
        with (
            patch(
                "maestro.app.ui.main_window.QSystemTrayIcon.isSystemTrayAvailable",
                return_value=False,
            ),
            patch("maestro.app.ui.controller.build_container"),
            patch("maestro.app.ui.controller.Runtime"),
        ):
            window = MainWindow(config_path="configs")
            qtbot.addWidget(window)

            assert window.isVisible() == (sys.platform != "darwin")
            assert window._tray is None
            assert window.windowIcon().isNull() is False
            assert window._menu_button.menu() is not None
            assert window._controls_menu.actions()[0].text() == "Active"

    def test_hand_preference_submenu_defaults_to_right(self, qtbot) -> None:
        with (
            patch("maestro.app.ui.controller.build_container"),
            patch("maestro.app.ui.controller.Runtime"),
        ):
            window = MainWindow(config_path="configs")
            qtbot.addWidget(window)

            # The hand submenu must be present in the controls menu.
            hand_action = next(
                (a for a in window._controls_menu.actions() if a.text() == "Hand"), None
            )
            assert hand_action is not None, "Hand submenu not found in controls menu"

            hand_entries = window._hand_menu.actions()
            assert len(hand_entries) == 3
            assert [a.text() for a in hand_entries] == ["Right Hand", "Left Hand", "Both Hands"]
            assert [a.data() for a in hand_entries] == ["right", "left", "both"]

            # configs/app.yaml has hand_preference: right — so Right Hand is checked.
            checked = [a for a in hand_entries if a.isChecked()]
            assert len(checked) == 1
            assert checked[0].data() == "right"

    def test_hand_preference_change_updates_internal_state(self, qtbot) -> None:
        with (
            patch("maestro.app.ui.controller.build_container"),
            patch("maestro.app.ui.controller.Runtime"),
        ):
            window = MainWindow(config_path="configs")
            qtbot.addWidget(window)

            left_action = next(a for a in window._hand_menu.actions() if a.data() == "left")
            left_action.trigger()

            assert window._hand_preference == "left"
