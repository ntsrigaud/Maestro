"""Unit tests for DesktopAdapter.

All OS side-effects are patched; pyautogui is never invoked during tests.
Platform selection is tested via the ``platform`` constructor argument so
tests are OS-independent.
"""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest

from gesture_control.domain.entities import DispatchResult
from gesture_control.infrastructure.dispatch.app_adapters.desktop_adapter import (
    DesktopAdapter,
)
from gesture_control.infrastructure.dispatch.app_adapters.keyboard_adapter import (
    KeyboardAdapter,
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

_PYA_PATH = "gesture_control.infrastructure.dispatch.app_adapters.keyboard_adapter.pyautogui"

SCROLL_CLICKS = 5
ZOOM_CLICKS = 3


def _adapter(
    *,
    platform: str = "darwin",
    navigation_mode: str = "workspace",
    scroll_clicks: int = SCROLL_CLICKS,
    zoom_clicks: int = ZOOM_CLICKS,
) -> tuple[DesktopAdapter, MagicMock]:
    mock_pya = MagicMock()
    with patch(_PYA_PATH, mock_pya):
        keyboard = KeyboardAdapter()
        adapter = DesktopAdapter(
            keyboard=keyboard,
            platform=platform,
            navigation_mode=navigation_mode,
            scroll_clicks=scroll_clicks,
            zoom_clicks=zoom_clicks,
        )
    return adapter, mock_pya


# ---------------------------------------------------------------------------
# Basic properties
# ---------------------------------------------------------------------------


class TestDesktopAdapterProperties:
    def test_target_app(self):
        adapter, _ = _adapter()
        assert adapter.target_app == "desktop"

    def test_unknown_platform_raises_value_error(self):
        with pytest.raises(ValueError, match="unsupported platform"):
            DesktopAdapter(platform="sunos")

    def test_linux2_normalised_to_linux(self):
        """CPython 2 reported sys.platform as 'linux2'; must be accepted."""
        adapter, _ = _adapter(platform="linux2")
        assert adapter.target_app == "desktop"


# ---------------------------------------------------------------------------
# Scroll and zoom
# ---------------------------------------------------------------------------


class TestScrollCommands:
    def test_scroll_forward_calls_negative_scroll(self):
        adapter, mock_pya = _adapter()
        with patch(_PYA_PATH, mock_pya):
            result = adapter.execute("scroll_forward")
        assert result.success is True
        mock_pya.scroll.assert_called_once_with(-SCROLL_CLICKS)

    def test_scroll_back_calls_positive_scroll(self):
        adapter, mock_pya = _adapter()
        with patch(_PYA_PATH, mock_pya):
            result = adapter.execute("scroll_back")
        assert result.success is True
        mock_pya.scroll.assert_called_once_with(SCROLL_CLICKS)

    def test_custom_scroll_clicks_respected(self):
        adapter, mock_pya = _adapter(scroll_clicks=10)
        with patch(_PYA_PATH, mock_pya):
            adapter.execute("scroll_forward")
        mock_pya.scroll.assert_called_once_with(-10)


class TestZoomCommands:
    def test_zoom_in_calls_modifier_scroll_positive(self):
        adapter, mock_pya = _adapter(platform="darwin")
        with patch(_PYA_PATH, mock_pya):
            result = adapter.execute("zoom_in_view")
        assert result.success is True
        mock_pya.keyDown.assert_called_once_with("command")
        mock_pya.scroll.assert_called_once_with(ZOOM_CLICKS)
        mock_pya.keyUp.assert_called_once_with("command")

    def test_zoom_out_calls_modifier_scroll_negative(self):
        adapter, mock_pya = _adapter(platform="darwin")
        with patch(_PYA_PATH, mock_pya):
            result = adapter.execute("zoom_out_view")
        assert result.success is True
        mock_pya.keyDown.assert_called_once_with("command")
        mock_pya.scroll.assert_called_once_with(-ZOOM_CLICKS)
        mock_pya.keyUp.assert_called_once_with("command")

    def test_zoom_on_windows_uses_ctrl(self):
        adapter, mock_pya = _adapter(platform="win32")
        with patch(_PYA_PATH, mock_pya):
            adapter.execute("zoom_in_view")
        mock_pya.keyDown.assert_called_once_with("ctrl")

    def test_zoom_on_linux_uses_ctrl(self):
        adapter, mock_pya = _adapter(platform="linux")
        with patch(_PYA_PATH, mock_pya):
            adapter.execute("zoom_in_view")
        mock_pya.keyDown.assert_called_once_with("ctrl")

    def test_custom_zoom_clicks_respected(self):
        adapter, mock_pya = _adapter(zoom_clicks=2)
        with patch(_PYA_PATH, mock_pya):
            adapter.execute("zoom_in_view")
        mock_pya.scroll.assert_called_once_with(2)


# ---------------------------------------------------------------------------
# Navigation — workspace mode
# ---------------------------------------------------------------------------


class TestNavigationWorkspaceMode:
    def _exec(self, command_id: str, platform: str) -> tuple[DispatchResult, MagicMock]:
        adapter, mock_pya = _adapter(platform=platform, navigation_mode="workspace")
        with patch(_PYA_PATH, mock_pya):
            result = adapter.execute(command_id)
        return result, mock_pya

    def test_darwin_forward_uses_ctrl_right(self):
        result, mock_pya = self._exec("navigate_forward", "darwin")
        assert result.success is True
        mock_pya.hotkey.assert_called_once_with("ctrl", "right")

    def test_darwin_back_uses_ctrl_left(self):
        result, mock_pya = self._exec("navigate_back", "darwin")
        assert result.success is True
        mock_pya.hotkey.assert_called_once_with("ctrl", "left")

    def test_win32_forward_uses_ctrl_win_right(self):
        result, mock_pya = self._exec("navigate_forward", "win32")
        assert result.success is True
        mock_pya.hotkey.assert_called_once_with("ctrl", "win", "right")

    def test_linux_forward_uses_ctrl_alt_right(self):
        result, mock_pya = self._exec("navigate_forward", "linux")
        assert result.success is True
        mock_pya.hotkey.assert_called_once_with("ctrl", "alt", "right")


# ---------------------------------------------------------------------------
# Navigation — tabs mode
# ---------------------------------------------------------------------------


class TestNavigationTabsMode:
    def _exec(self, command_id: str, platform: str) -> tuple[DispatchResult, MagicMock]:
        adapter, mock_pya = _adapter(platform=platform, navigation_mode="tabs")
        with patch(_PYA_PATH, mock_pya):
            result = adapter.execute(command_id)
        return result, mock_pya

    def test_darwin_forward_uses_ctrl_tab(self):
        result, mock_pya = self._exec("navigate_forward", "darwin")
        assert result.success is True
        mock_pya.hotkey.assert_called_once_with("ctrl", "tab")

    def test_darwin_back_uses_ctrl_shift_tab(self):
        result, mock_pya = self._exec("navigate_back", "darwin")
        assert result.success is True
        mock_pya.hotkey.assert_called_once_with("ctrl", "shift", "tab")

    def test_win32_tabs_uses_ctrl_tab(self):
        result, mock_pya = self._exec("navigate_forward", "win32")
        assert result.success is True
        mock_pya.hotkey.assert_called_once_with("ctrl", "tab")


# ---------------------------------------------------------------------------
# Mission Control / overview
# ---------------------------------------------------------------------------


class TestOverviewCommands:
    def _exec(self, platform: str) -> tuple[DispatchResult, MagicMock]:
        adapter, mock_pya = _adapter(platform=platform)
        with patch(_PYA_PATH, mock_pya):
            result = adapter.execute("mission_control")
        return result, mock_pya

    def test_darwin_uses_ctrl_up(self):
        result, mock_pya = self._exec("darwin")
        assert result.success is True
        mock_pya.hotkey.assert_called_once_with("ctrl", "up")

    def test_win32_uses_win_tab(self):
        result, mock_pya = self._exec("win32")
        assert result.success is True
        mock_pya.hotkey.assert_called_once_with("win", "tab")

    def test_linux_uses_super_a(self):
        result, mock_pya = self._exec("linux")
        assert result.success is True
        mock_pya.hotkey.assert_called_once_with("super", "a")

    def test_linux_kde_uses_super_w(self):
        """KDE desktop environment should override the GNOME default."""
        keyboard = KeyboardAdapter()
        mock_pya = MagicMock()
        with patch(_PYA_PATH, mock_pya):
            adapter = DesktopAdapter(
                keyboard=keyboard,
                platform="linux",
                linux_de="kde",
            )
            result = adapter.execute("mission_control")
        assert result.success is True
        mock_pya.hotkey.assert_called_once_with("super", "w")


# ---------------------------------------------------------------------------
# Edge cases
# ---------------------------------------------------------------------------


class TestDesktopAdapterEdgeCases:
    def test_no_action_succeeds_without_hotkey(self):
        adapter, mock_pya = _adapter()
        with patch(_PYA_PATH, mock_pya):
            result = adapter.execute("no_action")
        assert result.success is True
        assert result.detail == "no_action"
        mock_pya.hotkey.assert_not_called()
        mock_pya.scroll.assert_not_called()

    def test_unknown_command_returns_failure(self):
        adapter, mock_pya = _adapter()
        with patch(_PYA_PATH, mock_pya):
            result = adapter.execute("nonexistent_command")
        assert result.success is False
        assert "no_handler_for" in (result.detail or "")

    def test_keyboard_exception_returns_failure(self):
        """An OS error in keyboard delivery must not propagate."""
        adapter, mock_pya = _adapter()
        mock_pya.scroll.side_effect = OSError("kernel error")
        with patch(_PYA_PATH, mock_pya):
            result = adapter.execute("scroll_forward")
        assert result.success is False
        assert "adapter_error" in (result.detail or "")
