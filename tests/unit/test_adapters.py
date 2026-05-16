"""Unit tests for app adapters and CommandDispatcher.

All tests run without any OS side-effects because pyautogui.hotkey is
patched throughout.
"""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest

from maestro.domain.entities import DispatchResult, ExecutableCommand
from maestro.infrastructure.dispatch.app_adapters.keyboard_adapter import (
    KeyboardAdapter,
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
# KeyboardAdapter tests
# ---------------------------------------------------------------------------


class TestKeyboardAdapter:
    @pytest.fixture(autouse=True)
    def _force_presentation_scope(self):
        with patch.object(KeyboardAdapter, "_ensure_presentation_scope", return_value=True):
            yield

    def test_valid_keys_call_pyautogui(self):
        adapter = KeyboardAdapter()
        with patch(
            "maestro.infrastructure.dispatch.app_adapters.keyboard_adapter.pyautogui"
        ) as mock_pya:
            adapter.press_hotkey("ctrl", "right")
            mock_pya.hotkey.assert_called_once_with("ctrl", "right")

    def test_single_valid_key(self):
        adapter = KeyboardAdapter()
        with patch(
            "maestro.infrastructure.dispatch.app_adapters.keyboard_adapter.pyautogui"
        ) as mock_pya:
            adapter.press_hotkey("f5")
            mock_pya.hotkey.assert_called_once_with("f5")

    def test_invalid_key_raises_value_error(self):
        adapter = KeyboardAdapter()
        with pytest.raises(ValueError, match="safe key allowlist"):
            adapter.press_hotkey("ctrl", "EXEC_SHELL")

    def test_empty_key_tuple_raises_value_error(self):
        adapter = KeyboardAdapter()
        with pytest.raises((ValueError, TypeError)):
            adapter.press_hotkey()

    def test_invalid_key_never_calls_pyautogui(self):
        adapter = KeyboardAdapter()
        with patch(
            "maestro.infrastructure.dispatch.app_adapters.keyboard_adapter.pyautogui"
        ) as mock_pya:
            with pytest.raises(ValueError):
                adapter.press_hotkey("ctrl", "malicious")
            mock_pya.hotkey.assert_not_called()

    def test_new_platform_keys_in_allowlist(self):
        """win, super, command must pass the safe key allowlist."""
        adapter = KeyboardAdapter()
        with patch(
            "maestro.infrastructure.dispatch.app_adapters.keyboard_adapter.pyautogui"
        ) as mock_pya:
            for key in ("win", "super", "command"):
                mock_pya.reset_mock()
                adapter.press_hotkey(key)
                mock_pya.hotkey.assert_called_once_with(key)

    def test_scroll_calls_pyautogui_scroll_with_clamped_value(self):
        adapter = KeyboardAdapter()
        with patch(
            "maestro.infrastructure.dispatch.app_adapters.keyboard_adapter.pyautogui"
        ) as mock_pya:
            adapter.scroll(5)
            mock_pya.scroll.assert_called_once_with(5)

    def test_scroll_clamps_to_minus_20(self):
        adapter = KeyboardAdapter()
        with patch(
            "maestro.infrastructure.dispatch.app_adapters.keyboard_adapter.pyautogui"
        ) as mock_pya:
            adapter.scroll(-999)
            mock_pya.scroll.assert_called_once_with(-20)

    def test_scroll_clamps_to_plus_20(self):
        adapter = KeyboardAdapter()
        with patch(
            "maestro.infrastructure.dispatch.app_adapters.keyboard_adapter.pyautogui"
        ) as mock_pya:
            adapter.scroll(999)
            mock_pya.scroll.assert_called_once_with(20)

    def test_zoom_calls_keydown_scroll_keyup(self):
        adapter = KeyboardAdapter()
        with patch(
            "maestro.infrastructure.dispatch.app_adapters.keyboard_adapter.pyautogui"
        ) as mock_pya:
            adapter.zoom(3, modifier="ctrl")
            assert mock_pya.keyDown.call_count == 1
            assert mock_pya.scroll.call_count == 1
            assert mock_pya.keyUp.call_count == 1
            # Modifier must be released with the same key it was pressed with.
            assert mock_pya.keyDown.call_args[0][0] == mock_pya.keyUp.call_args[0][0]

    def test_zoom_clamps_to_minus_5(self):
        adapter = KeyboardAdapter()
        with patch(
            "maestro.infrastructure.dispatch.app_adapters.keyboard_adapter.pyautogui"
        ) as mock_pya:
            adapter.zoom(-99, modifier="ctrl")
            mock_pya.scroll.assert_called_once_with(-5)

    def test_zoom_releases_modifier_on_scroll_exception(self):
        """keyUp must be called even when pyautogui.scroll raises."""
        adapter = KeyboardAdapter()
        with patch(
            "maestro.infrastructure.dispatch.app_adapters.keyboard_adapter.pyautogui"
        ) as mock_pya:
            mock_pya.scroll.side_effect = RuntimeError("display error")
            with pytest.raises(RuntimeError):
                adapter.zoom(2, modifier="ctrl")
            assert mock_pya.keyUp.call_count == 1

    def test_move_mouse_calls_pyautogui_moveto(self):
        adapter = KeyboardAdapter()
        with patch(
            "maestro.infrastructure.dispatch.app_adapters.keyboard_adapter.pyautogui"
        ) as mock_pya:
            adapter.move_mouse(960, 540)
            mock_pya.moveTo.assert_called_once_with(960, 540, duration=0.0)

    def test_screen_size_returns_tuple_from_pyautogui(self):
        adapter = KeyboardAdapter()
        with patch(
            "maestro.infrastructure.dispatch.app_adapters.keyboard_adapter.pyautogui"
        ) as mock_pya:
            mock_pya.size.return_value = (2560, 1440)
            w, h = adapter.screen_size()
            assert w == 2560
            assert h == 1440

    def test_l_key_is_in_safe_allowlist(self):
        """Ctrl+L (laser pointer shortcut) must pass the safe key allowlist."""
        adapter = KeyboardAdapter()
        with patch(
            "maestro.infrastructure.dispatch.app_adapters.keyboard_adapter.pyautogui"
        ) as mock_pya:
            adapter.press_hotkey("ctrl", "l")
            mock_pya.hotkey.assert_called_once_with("ctrl", "l")

    def test_scope_guard_blocks_hotkey_outside_presentation(self):
        adapter = KeyboardAdapter()
        with (
            patch(
                "maestro.infrastructure.dispatch.app_adapters.keyboard_adapter.pyautogui"
            ) as mock_pya,
            patch.object(adapter, "_ensure_presentation_scope", return_value=False),
        ):
            ok = adapter.press_hotkey("ctrl", "right")
            assert ok is False
            mock_pya.hotkey.assert_not_called()

    def test_scope_guard_blocks_mouse_click_outside_presentation(self):
        adapter = KeyboardAdapter()
        with (
            patch(
                "maestro.infrastructure.dispatch.app_adapters.keyboard_adapter.pyautogui"
            ) as mock_pya,
            patch.object(adapter, "_ensure_presentation_scope", return_value=False),
        ):
            ok = adapter.mouse_click()
            assert ok is False
            mock_pya.click.assert_not_called()


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
        """After the window expires the counter resets."""
        dispatcher, mock_adapter = _make_dispatcher(rate_limit=1, window_ms=100)
        dispatcher.dispatch(_command())

        # Manually age the recorded timestamp so it falls outside the window
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
