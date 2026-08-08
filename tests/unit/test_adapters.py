"""Unit tests for app adapters and CommandDispatcher.

All tests run without any OS side-effects. The OS backend is mocked via Dependency Injection.
"""

from __future__ import annotations

import sys
from unittest.mock import MagicMock, call, patch

import pytest

from maestro.domain.entities import DispatchResult, ExecutableCommand
from maestro.infrastructure.dispatch.app_adapters.keyboard_adapter import (
    KeyboardAdapter,
    OSKeyboardBackend,
    PyAutoGUIKeyboardAdapter,
    WindowsKeyboardAdapter,
)
from maestro.infrastructure.dispatch.command_dispatcher import (
    CommandDispatcher,
)

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


def _command(command_id: str = "next_slide", gesture: str = "swipe_right") -> ExecutableCommand:
    return ExecutableCommand(
        command_id=command_id,
        source_gesture=gesture,
        issued_at_ms=1_000,
    )


# ---------------------------------------------------------------------------
# KeyboardAdapter tests (Facade / Logic)
# ---------------------------------------------------------------------------


class TestKeyboardAdapter:
    @pytest.fixture(autouse=True)
    def _force_presentation_scope(self):
        with patch.object(KeyboardAdapter, "_ensure_presentation_scope", return_value=True):
            yield

    @pytest.fixture
    def mock_backend(self):
        backend = MagicMock(spec=OSKeyboardBackend)
        backend.press_hotkey.return_value = True
        backend.scroll.return_value = True
        backend.zoom.return_value = True
        backend.move_mouse.return_value = True
        backend.mouse_drag_to.return_value = True
        backend.mouse_down.return_value = True
        backend.mouse_up.return_value = True
        backend.mouse_click.return_value = True
        backend.screen_size.return_value = (1920, 1080)
        return backend

    def test_valid_keys_call_backend(self, mock_backend):
        adapter = KeyboardAdapter(backend=mock_backend)
        adapter.press_hotkey("ctrl", "right")
        mock_backend.press_hotkey.assert_called_once_with("ctrl", "right")

    def test_single_valid_key(self, mock_backend):
        adapter = KeyboardAdapter(backend=mock_backend)
        adapter.press_hotkey("f5")
        mock_backend.press_hotkey.assert_called_once_with("f5")

    def test_invalid_key_raises_value_error(self, mock_backend):
        adapter = KeyboardAdapter(backend=mock_backend)
        with pytest.raises(ValueError, match="safe key allowlist"):
            adapter.press_hotkey("ctrl", "EXEC_SHELL")

    def test_empty_key_tuple_raises_value_error(self, mock_backend):
        adapter = KeyboardAdapter(backend=mock_backend)
        with pytest.raises((ValueError, TypeError)):
            adapter.press_hotkey()

    def test_invalid_key_never_calls_backend(self, mock_backend):
        adapter = KeyboardAdapter(backend=mock_backend)
        with pytest.raises(ValueError):
            adapter.press_hotkey("ctrl", "malicious")
        mock_backend.press_hotkey.assert_not_called()

    def test_new_platform_keys_in_allowlist(self, mock_backend):
        """win, super, command must pass the safe key allowlist."""
        adapter = KeyboardAdapter(backend=mock_backend)
        for key in ("win", "super", "command"):
            mock_backend.reset_mock()
            adapter.press_hotkey(key)
            mock_backend.press_hotkey.assert_called_once_with(key)

    def test_scroll_calls_backend_scroll_with_clamped_value(self, mock_backend):
        adapter = KeyboardAdapter(backend=mock_backend)
        adapter.scroll(5)
        mock_backend.scroll.assert_called_once_with(5)

    def test_scroll_clamps_to_minus_20(self, mock_backend):
        adapter = KeyboardAdapter(backend=mock_backend)
        adapter.scroll(-999)
        mock_backend.scroll.assert_called_once_with(-20)

    def test_scroll_clamps_to_plus_20(self, mock_backend):
        adapter = KeyboardAdapter(backend=mock_backend)
        adapter.scroll(999)
        mock_backend.scroll.assert_called_once_with(20)

    def test_zoom_delegates_to_backend(self, mock_backend):
        adapter = KeyboardAdapter(backend=mock_backend)
        adapter.zoom(3, modifier="ctrl", max_abs_clicks=15)
        mock_backend.zoom.assert_called_once_with(3, modifier="ctrl", max_abs_clicks=15)

    def test_zoom_rejects_non_positive_max_abs_clicks(self, mock_backend):
        adapter = KeyboardAdapter(backend=mock_backend)
        with pytest.raises(ValueError, match="max_abs_clicks"):
            adapter.zoom(1, modifier="ctrl", max_abs_clicks=0)

    def test_move_mouse_calls_backend(self, mock_backend):
        adapter = KeyboardAdapter(backend=mock_backend)
        adapter.move_mouse(960, 540)
        mock_backend.move_mouse.assert_called_once_with(960, 540)

    def test_screen_size_returns_tuple_from_backend(self, mock_backend):
        adapter = KeyboardAdapter(backend=mock_backend)
        mock_backend.screen_size.return_value = (2560, 1440)
        w, h = adapter.screen_size()
        assert w == 2560
        assert h == 1440

    def test_l_key_is_in_safe_allowlist(self, mock_backend):
        """Ctrl+L (laser pointer shortcut) must pass the safe key allowlist."""
        adapter = KeyboardAdapter(backend=mock_backend)
        adapter.press_hotkey("ctrl", "l")
        mock_backend.press_hotkey.assert_called_once_with("ctrl", "l")

    def test_scope_guard_blocks_hotkey_outside_presentation(self, mock_backend):
        adapter = KeyboardAdapter(backend=mock_backend)
        with patch.object(adapter, "_ensure_presentation_scope", return_value=False):
            ok = adapter.press_hotkey("ctrl", "right")
            assert ok is False
            mock_backend.press_hotkey.assert_not_called()

    def test_scope_guard_blocks_mouse_click_outside_presentation(self, mock_backend):
        adapter = KeyboardAdapter(backend=mock_backend)
        with patch.object(adapter, "_ensure_presentation_scope", return_value=False):
            ok = adapter.mouse_click()
            assert ok is False
            mock_backend.mouse_click.assert_not_called()


# ---------------------------------------------------------------------------
# Scope-check unit tests
# ---------------------------------------------------------------------------


class TestKeyboardAdapterScopeEnforcement:
    """Tests for _ensure_presentation_scope without the autouse fixture."""

    _KB_MOD = "maestro.infrastructure.dispatch.app_adapters.keyboard_adapter"

    def test_scope_check_blocks_when_identity_is_none(self):
        adapter = KeyboardAdapter()
        with patch(f"{self._KB_MOD}._foreground_identity", return_value=None):
            result = adapter._ensure_presentation_scope("hotkey:e")
        assert result is False

    def test_scope_check_allows_powerpoint_foreground(self):
        adapter = KeyboardAdapter()
        with patch(
            f"{self._KB_MOD}._foreground_identity",
            return_value="com.microsoft.powerpoint microsoft powerpoint",
        ):
            result = adapter._ensure_presentation_scope("hotkey:right")
        assert result is True

    def test_scope_check_blocks_keynote_foreground(self):
        adapter = KeyboardAdapter()
        with patch(
            f"{self._KB_MOD}._foreground_identity",
            return_value="com.apple.keynote keynote",
        ):
            result = adapter._ensure_presentation_scope("hotkey:right")
        assert result is False

    def test_scope_check_blocks_libreoffice_foreground(self):
        adapter = KeyboardAdapter()
        with patch(
            f"{self._KB_MOD}._foreground_identity",
            return_value="soffice libreoffice impress",
        ):
            result = adapter._ensure_presentation_scope("hotkey:right")
        assert result is False

    def test_scope_check_env_var_override_adds_custom_token(self):
        import os

        adapter = KeyboardAdapter()
        env = {"MAESTRO_PRESENTATION_SCOPE_ALLOWLIST": "myslides"}
        with (
            patch(f"{self._KB_MOD}._foreground_identity", return_value="myslides app"),
            patch.dict(os.environ, env, clear=False),
        ):
            result = adapter._ensure_presentation_scope("hotkey:right")
        assert result is True

    def test_erase_annotations_key_blocked_when_not_powerpoint(self):
        mock_backend = MagicMock(spec=OSKeyboardBackend)
        adapter = KeyboardAdapter(backend=mock_backend)
        with patch(f"{self._KB_MOD}._foreground_identity", return_value=None):
            ok = adapter.press_hotkey("e")
        assert ok is False
        mock_backend.press_hotkey.assert_not_called()


# ---------------------------------------------------------------------------
# Backend-specific Unit Tests
# ---------------------------------------------------------------------------


class TestPyAutoGUIKeyboardAdapter:
    @pytest.fixture
    def mock_pya(self):
        with patch("maestro.infrastructure.dispatch.app_adapters.keyboard_adapter.pyautogui") as m:
            yield m

    def test_zoom_calls_keydown_scroll_keyup(self, mock_pya):
        adapter = PyAutoGUIKeyboardAdapter()
        adapter.zoom(3, modifier="ctrl", max_abs_clicks=15)
        assert mock_pya.keyDown.call_count == 1
        assert mock_pya.scroll.call_count == 1
        assert mock_pya.keyUp.call_count == 1
        assert mock_pya.keyDown.call_args[0][0] == mock_pya.keyUp.call_args[0][0]

    def test_zoom_clamps_to_configured_limit(self, mock_pya):
        adapter = PyAutoGUIKeyboardAdapter()
        adapter.zoom(-99, modifier="ctrl", max_abs_clicks=15)
        mock_pya.scroll.assert_called_once_with(-15)

    def test_zoom_releases_modifier_on_scroll_exception(self, mock_pya):
        adapter = PyAutoGUIKeyboardAdapter()
        mock_pya.scroll.side_effect = RuntimeError("display error")
        with pytest.raises(RuntimeError):
            adapter.zoom(2, modifier="ctrl", max_abs_clicks=15)
        assert mock_pya.keyUp.call_count == 1

    def test_mouse_drag_to(self, mock_pya):
        adapter = PyAutoGUIKeyboardAdapter()
        adapter.mouse_drag_to(100, 200)
        mock_pya.dragTo.assert_called_once_with(
            100,
            200,
            duration=0.0,
            button="left",
            mouseDownUp=False,
            _pause=False,
        )


class TestWindowsKeyboardAdapter:
    @pytest.fixture
    def mock_ctypes(self):
        with patch(
            "maestro.infrastructure.dispatch.app_adapters.keyboard_adapter.ctypes"
        ) as mock_c:
            yield mock_c

    @pytest.fixture
    def mock_user32(self, mock_ctypes):
        user32 = MagicMock()
        user32.GetSystemMetrics.side_effect = lambda m: 1920 if m == 0 else 1080
        mock_ctypes.windll.user32 = user32
        return user32

    @pytest.mark.skipif(sys.platform != "win32", reason="Windows specific logic")
    def test_mouse_drag_to_sends_win32_move_event_on_windows(self, mock_user32):
        adapter = WindowsKeyboardAdapter()
        adapter.mouse_drag_to(960, 540)
        # MOVE | ABSOLUTE = 0x8001
        # Normalised coords: 65536 * 960 // 1920 + 1 = 32769
        mock_user32.mouse_event.assert_called_once_with(0x8001, 32769, 32769, 0, 0)

    @pytest.mark.skipif(sys.platform != "win32", reason="Windows specific logic")
    def test_mouse_down_sends_win32_move_then_leftdown_on_windows(self, mock_user32):
        adapter = WindowsKeyboardAdapter()
        adapter.mouse_down(960, 540)
        assert mock_user32.mouse_event.call_count == 2
        first_call, second_call = mock_user32.mouse_event.call_args_list
        assert first_call == call(0x8001, 32769, 32769, 0, 0)
        assert second_call == call(0x0002, 0, 0, 0, 0)


# ---------------------------------------------------------------------------
# CommandDispatcher tests
# ---------------------------------------------------------------------------


def _make_dispatcher(
    *,
    adapter=None,
    allowed: set[str] | None = None,
    rate_limit: int = 10,
    window_ms: int = 1000,
) -> tuple[CommandDispatcher, MagicMock]:
    mock_adapter = adapter or MagicMock()
    mock_adapter.target_app = "test_app"
    mock_adapter.execute.return_value = DispatchResult(
        success=True, command_id="next_slide", target_app="test_app", detail="ok"
    )
    cmd_set = allowed if allowed is not None else {"next_slide", "previous_slide", "raise_hand"}
    dispatcher = CommandDispatcher(
        adapter=mock_adapter,
        allowed_commands=cmd_set,
        rate_limit_per_window=rate_limit,
        rate_limit_window_ms=window_ms,
    )
    return dispatcher, mock_adapter


class TestCommandDispatcherAllowlist:
    def test_allowed_command_passes_to_adapter(self):
        dispatcher, mock_adapter = _make_dispatcher()
        result = dispatcher.dispatch(_command("next_slide"))
        mock_adapter.execute.assert_called_once_with("next_slide")
        assert result.success is True

    def test_unlisted_command_rejected_before_adapter(self):
        dispatcher, mock_adapter = _make_dispatcher()
        result = dispatcher.dispatch(_command("very_dangerous_command"))
        mock_adapter.execute.assert_not_called()
        assert result.success is False
        assert "rejected_by_allowlist" in (result.detail or "")

    def test_allowed_commands_property_returns_frozenset(self):
        dispatcher, _ = _make_dispatcher(allowed={"next_slide"})
        assert isinstance(dispatcher.allowed_commands, frozenset)
        assert "next_slide" in dispatcher.allowed_commands

    def test_empty_allowlist_rejects_all(self):
        dispatcher, mock_adapter = _make_dispatcher(allowed=set())
        result = dispatcher.dispatch(_command("next_slide"))
        mock_adapter.execute.assert_not_called()
        assert result.success is False


class TestCommandDispatcherRateLimiting:
    def test_commands_within_limit_all_pass(self):
        dispatcher, mock_adapter = _make_dispatcher(rate_limit=3, window_ms=1000)
        for _ in range(3):
            result = dispatcher.dispatch(_command())
            assert result.success is True
        assert mock_adapter.execute.call_count == 3

    def test_command_exceeding_limit_is_dropped(self):
        dispatcher, mock_adapter = _make_dispatcher(rate_limit=2, window_ms=10_000)
        dispatcher.dispatch(_command())
        dispatcher.dispatch(_command())
        result = dispatcher.dispatch(_command())  # 3rd — should be rate-limited
        assert result.success is False
        assert "rate_limited" in (result.detail or "")
        assert mock_adapter.execute.call_count == 2

    def test_zero_rate_limit_disables_limiting(self):
        dispatcher, mock_adapter = _make_dispatcher(rate_limit=0, window_ms=1000)
        for _ in range(20):
            dispatcher.dispatch(_command())
        assert mock_adapter.execute.call_count == 20

    def test_rate_limit_window_expires(self):
        dispatcher, mock_adapter = _make_dispatcher(rate_limit=1, window_ms=100)
        dispatcher.dispatch(_command())
        dispatcher._dispatch_timestamps[0] -= 200  # 200 ms in the past
        result = dispatcher.dispatch(_command())
        assert result.success is True
        assert mock_adapter.execute.call_count == 2

    def test_invalid_rate_limit_raises(self):
        with pytest.raises(ValueError):
            CommandDispatcher(
                adapter=MagicMock(),
                allowed_commands={"a"},
                rate_limit_per_window=-1,
            )

    def test_invalid_window_ms_raises(self):
        with pytest.raises(ValueError):
            CommandDispatcher(
                adapter=MagicMock(),
                allowed_commands={"a"},
                rate_limit_window_ms=0,
            )


class TestCommandDispatcherAdapterExceptions:
    def test_adapter_exception_returns_failure_not_crash(self):
        dispatcher, mock_adapter = _make_dispatcher()
        mock_adapter.execute.side_effect = RuntimeError("boom")
        result = dispatcher.dispatch(_command("next_slide"))
        assert result.success is False
        assert "adapter_exception" in (result.detail or "")

    def test_adapter_returns_failure_is_passed_through(self):
        dispatcher, mock_adapter = _make_dispatcher()
        mock_adapter.execute.return_value = DispatchResult(
            success=False,
            command_id="next_slide",
            target_app="test_app",
            detail="hw error",
        )
        result = dispatcher.dispatch(_command("next_slide"))
        assert result.success is False
        assert result.detail == "hw error"


class TestFocusDetection:
    def test_keyboard_adapter_is_target_app_focused_true(self):
        adapter = KeyboardAdapter(backend=MagicMock())
        with patch(
            "maestro.infrastructure.dispatch.app_adapters.keyboard_adapter._foreground_identity",
            return_value="microsoft powerpoint",
        ):
            assert adapter.is_target_app_focused() is True

    def test_keyboard_adapter_is_target_app_focused_false(self):
        adapter = KeyboardAdapter(backend=MagicMock())
        with patch(
            "maestro.infrastructure.dispatch.app_adapters.keyboard_adapter._foreground_identity",
            return_value="safari",
        ):
            assert adapter.is_target_app_focused() is False

    def test_command_dispatcher_is_target_app_focused(self):
        mock_adapter = MagicMock()
        mock_adapter.is_target_app_focused.return_value = False
        dispatcher = CommandDispatcher(adapter=mock_adapter, allowed_commands={"next_slide"})
        assert dispatcher.is_target_app_focused() is False
