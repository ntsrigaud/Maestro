"""Unit tests for PresentationAdapter.

All OS-level side-effects (pyautogui) are patched so the tests run cleanly
in headless / CI environments.
"""

from __future__ import annotations

from unittest.mock import MagicMock

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
    def test_zoom_in_view_sends_modifier_equal(self):
        kb = _mock_keyboard()
        adapter = PresentationAdapter(keyboard=kb)
        result = adapter.execute("zoom_in_view")
        # Modifier varies by platform; verify the second key is "equal".
        args = kb.press_hotkey.call_args[0]
        assert args[-1] == "equal"
        assert result.success is True

    def test_zoom_out_view_sends_modifier_minus(self):
        kb = _mock_keyboard()
        adapter = PresentationAdapter(keyboard=kb)
        result = adapter.execute("zoom_out_view")
        args = kb.press_hotkey.call_args[0]
        assert args[-1] == "minus"
        assert result.success is True


# ---------------------------------------------------------------------------
# Tests: start/stop presentation
# ---------------------------------------------------------------------------


class TestPresentationAdapterSlideshow:
    def test_start_stop_presentation_sends_keys(self):
        kb = _mock_keyboard()
        adapter = PresentationAdapter(keyboard=kb)
        result = adapter.execute("start_stop_presentation")
        kb.press_hotkey.assert_called_once()
        assert result.success is True


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

    def test_all_eight_commands_handled(self):
        """Smoke test: none of the eight canonical commands raise."""
        commands = [
            "next_slide",
            "previous_slide",
            "scroll_forward",
            "scroll_back",
            "zoom_in_view",
            "zoom_out_view",
            "start_stop_presentation",
            "no_action",
        ]
        kb = _mock_keyboard()
        adapter = PresentationAdapter(keyboard=kb)
        for cmd in commands:
            result = adapter.execute(cmd)
            assert isinstance(result, DispatchResult)
