"""Unit tests for PresentationAdapter.

All OS-level side-effects (pyautogui) are patched so the tests run cleanly
in headless / CI environments.
"""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import maestro.infrastructure.dispatch.app_adapters.presentation_adapter as presentation_module
from maestro.domain.entities import DispatchResult
from maestro.infrastructure.dispatch.app_adapters.keyboard_adapter import (
    KeyboardAdapter,
)
from maestro.infrastructure.dispatch.app_adapters.presentation_adapter import (
    PresentationAdapter,
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _mock_keyboard() -> KeyboardAdapter:
    """Return a KeyboardAdapter whose press_hotkey is mocked."""
    kb = MagicMock(spec=KeyboardAdapter)
    return kb


# ---------------------------------------------------------------------------
# Tests: navigation commands
# ---------------------------------------------------------------------------


class TestPresentationAdapterNavigation:
    def test_next_slide_presses_right(self):
        kb = _mock_keyboard()
        adapter = PresentationAdapter(keyboard=kb)
        result = adapter.execute("next_slide")
        kb.press_hotkey.assert_called_once_with("right")
        assert result.success is True
        assert result.command_id == "next_slide"
        assert result.target_app == "presentation"

    def test_previous_slide_presses_left(self):
        kb = _mock_keyboard()
        adapter = PresentationAdapter(keyboard=kb)
        result = adapter.execute("previous_slide")
        kb.press_hotkey.assert_called_once_with("left")
        assert result.success is True

    def test_scroll_forward_presses_down(self):
        kb = _mock_keyboard()
        adapter = PresentationAdapter(keyboard=kb)
        result = adapter.execute("scroll_forward")
        kb.press_hotkey.assert_called_once_with("down")
        assert result.success is True

    def test_scroll_back_presses_up(self):
        kb = _mock_keyboard()
        adapter = PresentationAdapter(keyboard=kb)
        result = adapter.execute("scroll_back")
        kb.press_hotkey.assert_called_once_with("up")
        assert result.success is True


# ---------------------------------------------------------------------------
# Tests: zoom commands
# ---------------------------------------------------------------------------


class TestPresentationAdapterZoom:
    def test_zoom_in_view_calls_zoom_positive(self):
        kb = _mock_keyboard()
        adapter = PresentationAdapter(keyboard=kb)
        result = adapter.execute("zoom_in_view")
        kb.zoom.assert_called_once()
        (clicks,) = kb.zoom.call_args.args
        assert clicks > 0, "zoom_in_view must pass a positive click count"
        assert result.success is True

    def test_zoom_out_view_calls_zoom_negative(self):
        kb = _mock_keyboard()
        adapter = PresentationAdapter(keyboard=kb)
        result = adapter.execute("zoom_out_view")
        kb.zoom.assert_called_once()
        (clicks,) = kb.zoom.call_args.args
        assert clicks < 0, "zoom_out_view must pass a negative click count"
        assert result.success is True

    def test_zoom_in_view_passes_modifier(self):
        """Modifier key must be forwarded so the correct OS zoom is triggered."""
        kb = _mock_keyboard()
        adapter = PresentationAdapter(keyboard=kb)
        adapter.execute("zoom_in_view")
        _, kwargs = kb.zoom.call_args
        assert "modifier" in kwargs, "zoom() must receive the modifier kwarg"
        assert kwargs["modifier"] in {"command", "ctrl"}

    def test_zoom_out_view_passes_modifier(self):
        kb = _mock_keyboard()
        adapter = PresentationAdapter(keyboard=kb)
        adapter.execute("zoom_out_view")
        _, kwargs = kb.zoom.call_args
        assert "modifier" in kwargs
        assert kwargs["modifier"] in {"command", "ctrl"}

    def test_zoom_out_totally_calls_zoom_negative_maximum(self):
        kb = _mock_keyboard()
        adapter = PresentationAdapter(keyboard=kb)
        result = adapter.execute("zoom_out_totally")
        kb.zoom.assert_called_once()
        (clicks,) = kb.zoom.call_args.args
        assert clicks <= -5, "zoom_out_totally must zoom out at maximum magnitude"
        assert result.success is True

    def test_zoom_out_totally_passes_modifier(self):
        kb = _mock_keyboard()
        adapter = PresentationAdapter(keyboard=kb)
        adapter.execute("zoom_out_totally")
        _, kwargs = kb.zoom.call_args
        assert "modifier" in kwargs
        assert kwargs["modifier"] in {"command", "ctrl"}


# ---------------------------------------------------------------------------
# Tests: erase annotations
# ---------------------------------------------------------------------------


class TestPresentationAdapterEraseAnnotations:
    def test_erase_annotations_presses_e(self):
        kb = _mock_keyboard()
        adapter = PresentationAdapter(keyboard=kb)
        result = adapter.execute("erase_annotations")
        kb.press_hotkey.assert_called_once_with("e")
        assert result.success is True

    def test_erase_annotations_blocked_outside_presentation_scope(self):
        kb = _mock_keyboard()
        kb.press_hotkey.return_value = False
        adapter = PresentationAdapter(keyboard=kb)
        result = adapter.execute("erase_annotations")
        kb.press_hotkey.assert_called_once_with("e")
        assert result.success is False
        assert result.detail == "blocked_outside_presentation_scope"


# ---------------------------------------------------------------------------
# Tests: start/stop presentation
# ---------------------------------------------------------------------------


class TestPresentationAdapterSlideshow:
    def test_start_presentation_sends_slideshow_keys(self):
        kb = _mock_keyboard()
        adapter = PresentationAdapter(keyboard=kb)
        result = adapter.execute("start_presentation")
        # Platform-specific: Cmd+Shift+Return on macOS, F5 on Win/Linux.
        kb.press_hotkey.assert_called_once()
        assert result.success is True

    def test_start_presentation_uses_cmd_shift_enter_on_macos_browser(self):
        kb = _mock_keyboard()
        with (
            patch.object(presentation_module.sys, "platform", "darwin"),
            patch.object(
                presentation_module,
                "_foreground_identity",
                return_value="org.mozilla.firefox firefox my deck - google slides",
            ),
        ):
            adapter = PresentationAdapter(keyboard=kb)
            result = adapter.execute("start_presentation")

        kb.press_hotkey.assert_called_once_with("command", "shift", "enter")
        assert result.success is True

    def test_start_presentation_uses_cmd_enter_on_macos_native_apps(self):
        kb = _mock_keyboard()
        with (
            patch.object(presentation_module.sys, "platform", "darwin"),
            patch.object(
                presentation_module,
                "_foreground_identity",
                return_value="com.microsoft.powerpoint microsoft powerpoint",
            ),
        ):
            adapter = PresentationAdapter(keyboard=kb)
            result = adapter.execute("start_presentation")

        kb.press_hotkey.assert_called_once_with("command", "enter")
        assert result.success is True

    def test_start_presentation_uses_ctrl_f5_on_windows_browser(self):
        kb = _mock_keyboard()
        with (
            patch.object(presentation_module.sys, "platform", "win32"),
            patch.object(
                presentation_module,
                "_foreground_identity",
                return_value="chrome.exe google slides",
            ),
        ):
            adapter = PresentationAdapter(keyboard=kb)
            result = adapter.execute("start_presentation")

        kb.press_hotkey.assert_called_once_with("ctrl", "f5")
        assert result.success is True

    def test_stop_presentation_sends_escape(self):
        kb = _mock_keyboard()
        adapter = PresentationAdapter(keyboard=kb)
        result = adapter.execute("stop_presentation")
        kb.press_hotkey.assert_called_once_with("escape")
        assert result.success is True

    def test_start_stop_presentation_sends_keys(self):
        # Legacy command kept for backward compat.
        kb = _mock_keyboard()
        adapter = PresentationAdapter(keyboard=kb)
        result = adapter.execute("start_stop_presentation")
        kb.press_hotkey.assert_called_once()
        assert result.success is True


# ---------------------------------------------------------------------------
# Tests: mouse pointer
# ---------------------------------------------------------------------------


class TestPresentationAdapterMousePointer:
    def test_mouse_pointer_presses_ctrl_a(self):
        kb = _mock_keyboard()
        adapter = PresentationAdapter(keyboard=kb)
        result = adapter.execute("mouse_pointer")
        kb.press_hotkey.assert_called_once_with("ctrl", "a")
        assert result.success is True
        assert result.command_id == "mouse_pointer"
        assert result.target_app == "presentation"


# ---------------------------------------------------------------------------
# Tests: no_action and unknown commands
# ---------------------------------------------------------------------------


class TestPresentationAdapterNoOp:
    def test_no_action_returns_not_executed(self):
        kb = _mock_keyboard()
        adapter = PresentationAdapter(keyboard=kb)
        result = adapter.execute("no_action")
        kb.press_hotkey.assert_not_called()
        assert result.success is False
        assert result.detail == "no_action"

    def test_unknown_command_returns_not_executed(self):
        kb = _mock_keyboard()
        adapter = PresentationAdapter(keyboard=kb)
        result = adapter.execute("open_application")
        kb.press_hotkey.assert_not_called()
        assert result.success is False
        assert result.detail is not None
        assert "unknown_command" in result.detail

    def test_target_app_is_presentation(self):
        adapter = PresentationAdapter(keyboard=_mock_keyboard())
        assert adapter.target_app == "presentation"

    def test_all_commands_handled(self):
        """Smoke test: all canonical commands (new and legacy) do not raise."""
        commands = [
            # Primary gesture-mapped commands
            "start_presentation",
            "stop_presentation",
            "next_slide",
            "previous_slide",
            "zoom_in_view",
            "zoom_out_view",
            "zoom_out_totally",
            "erase_annotations",
            "mouse_pointer",
            "no_action",
            # LD-CONG gesture set
            "mouse_click",
            "toggle_black_screen",
            "toggle_white_screen",
            "go_to_last_slide",
            "go_to_first_slide",
            # Legacy / extended commands (kept for backward compat)
            "scroll_forward",
            "scroll_back",
            "start_stop_presentation",
            "activate_laser_pointer",
            "deactivate_laser_pointer",
        ]
        kb = _mock_keyboard()
        adapter = PresentationAdapter(keyboard=kb)
        for cmd in commands:
            result = adapter.execute(cmd)
            assert isinstance(result, DispatchResult)


# ---------------------------------------------------------------------------
# Tests: laser pointer commands
# ---------------------------------------------------------------------------


class TestPresentationAdapterLaserPointer:
    def test_activate_laser_pointer_presses_ctrl_l(self):
        kb = _mock_keyboard()
        adapter = PresentationAdapter(keyboard=kb)
        result = adapter.execute("activate_laser_pointer")
        kb.press_hotkey.assert_called_once_with("ctrl", "l")
        assert result.success is True
        assert result.command_id == "activate_laser_pointer"
        assert result.target_app == "presentation"

    def test_deactivate_laser_pointer_presses_ctrl_a(self):
        kb = _mock_keyboard()
        adapter = PresentationAdapter(keyboard=kb)
        result = adapter.execute("deactivate_laser_pointer")
        kb.press_hotkey.assert_called_once_with("ctrl", "a")
        assert result.success is True
        assert result.command_id == "deactivate_laser_pointer"
        assert result.target_app == "presentation"


# ---------------------------------------------------------------------------
# Tests: screen blanking and slide jump commands (LD-CONG gesture set)
# ---------------------------------------------------------------------------


class TestPresentationAdapterScreenAndNavigation:
    def test_mouse_click_calls_mouse_click(self):
        kb = _mock_keyboard()
        adapter = PresentationAdapter(keyboard=kb)
        result = adapter.execute("mouse_click")
        kb.mouse_click.assert_called_once_with()
        assert result.success is True
        assert result.command_id == "mouse_click"
        assert result.target_app == "presentation"

    def test_toggle_black_screen_presses_b(self):
        kb = _mock_keyboard()
        adapter = PresentationAdapter(keyboard=kb)
        result = adapter.execute("toggle_black_screen")
        kb.press_hotkey.assert_called_once_with("b")
        assert result.success is True
        assert result.command_id == "toggle_black_screen"
        assert result.target_app == "presentation"

    def test_toggle_white_screen_presses_w(self):
        kb = _mock_keyboard()
        adapter = PresentationAdapter(keyboard=kb)
        result = adapter.execute("toggle_white_screen")
        kb.press_hotkey.assert_called_once_with("w")
        assert result.success is True
        assert result.command_id == "toggle_white_screen"
        assert result.target_app == "presentation"

    def test_go_to_last_slide_presses_end(self):
        kb = _mock_keyboard()
        adapter = PresentationAdapter(keyboard=kb)
        result = adapter.execute("go_to_last_slide")
        kb.press_hotkey.assert_called_once_with("end")
        assert result.success is True
        assert result.command_id == "go_to_last_slide"
        assert result.target_app == "presentation"

    def test_go_to_first_slide_presses_home(self):
        kb = _mock_keyboard()
        adapter = PresentationAdapter(keyboard=kb)
        result = adapter.execute("go_to_first_slide")
        kb.press_hotkey.assert_called_once_with("home")
        assert result.success is True
        assert result.command_id == "go_to_first_slide"
        assert result.target_app == "presentation"
