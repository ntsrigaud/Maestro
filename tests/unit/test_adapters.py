"""Unit tests for app adapters and CommandDispatcher.

All tests run without any OS side-effects because pyautogui.hotkey is
patched throughout.
"""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest

from gesture_control.domain.entities import DispatchResult, ExecutableCommand
from gesture_control.infrastructure.dispatch.app_adapters.keyboard_adapter import (
    KeyboardAdapter,
)
from gesture_control.infrastructure.dispatch.app_adapters.powerpoint_adapter import (
    PowerPointAdapter,
)
from gesture_control.infrastructure.dispatch.app_adapters.zoom_adapter import (
    ZoomAdapter,
)
from gesture_control.infrastructure.dispatch.command_dispatcher import (
    CommandDispatcher,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


def _command(
    command_id: str = "next_slide", gesture: str = "swipe_right"
) -> ExecutableCommand:
    return ExecutableCommand(
        command_id=command_id,
        source_gesture=gesture,
        issued_at_ms=1_000,
    )


# ---------------------------------------------------------------------------
# KeyboardAdapter tests
# ---------------------------------------------------------------------------


class TestKeyboardAdapter:
    def test_valid_keys_call_pyautogui(self):
        adapter = KeyboardAdapter()
        with patch(
            "gesture_control.infrastructure.dispatch.app_adapters.keyboard_adapter.pyautogui"
        ) as mock_pya:
            adapter.press_hotkey("ctrl", "right")
            mock_pya.hotkey.assert_called_once_with("ctrl", "right")

    def test_single_valid_key(self):
        adapter = KeyboardAdapter()
        with patch(
            "gesture_control.infrastructure.dispatch.app_adapters.keyboard_adapter.pyautogui"
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
            "gesture_control.infrastructure.dispatch.app_adapters.keyboard_adapter.pyautogui"
        ) as mock_pya:
            with pytest.raises(ValueError):
                adapter.press_hotkey("ctrl", "malicious")
            mock_pya.hotkey.assert_not_called()


# ---------------------------------------------------------------------------
# PowerPointAdapter tests
# ---------------------------------------------------------------------------


class TestPowerPointAdapter:
    def setup_method(self):
        self.kb = KeyboardAdapter()
        self.adapter = PowerPointAdapter(keyboard=self.kb)
        self._patcher = patch(
            "gesture_control.infrastructure.dispatch.app_adapters.keyboard_adapter.pyautogui"
        )
        self.mock_pya = self._patcher.start()

    def teardown_method(self):
        self._patcher.stop()

    def test_target_app(self):
        assert self.adapter.target_app == "powerpoint"

    def test_next_slide_delegates_correct_keys(self):
        result = self.adapter.execute("next_slide")
        assert result.success is True
        self.mock_pya.hotkey.assert_called_once_with("right")

    def test_previous_slide_delegates_correct_keys(self):
        result = self.adapter.execute("previous_slide")
        assert result.success is True
        self.mock_pya.hotkey.assert_called_once_with("left")

    def test_start_presentation_uses_f5(self):
        result = self.adapter.execute("start_presentation")
        assert result.success is True
        self.mock_pya.hotkey.assert_called_once_with("f5")

    def test_toggle_pointer_mode_uses_ctrl_l(self):
        result = self.adapter.execute("toggle_pointer_mode")
        assert result.success is True
        self.mock_pya.hotkey.assert_called_once_with("ctrl", "l")

    def test_unknown_command_returns_failure(self):
        result = self.adapter.execute("unknown_command_xyz")
        assert result.success is False
        assert "no_shortcut_for" in (result.detail or "")

    def test_keyboard_exception_returns_failure(self):
        self.mock_pya.hotkey.side_effect = OSError("kbd error")
        result = self.adapter.execute("next_slide")
        assert result.success is False
        assert "kbd error" in (result.detail or "")


# ---------------------------------------------------------------------------
# ZoomAdapter tests
# ---------------------------------------------------------------------------


class TestZoomAdapter:
    def setup_method(self):
        self.kb = KeyboardAdapter()
        self.adapter = ZoomAdapter(keyboard=self.kb)
        self._patcher = patch(
            "gesture_control.infrastructure.dispatch.app_adapters.keyboard_adapter.pyautogui"
        )
        self.mock_pya = self._patcher.start()

    def teardown_method(self):
        self._patcher.stop()

    def test_target_app(self):
        assert self.adapter.target_app == "zoom"

    def test_toggle_pointer_mode_uses_alt_f2(self):
        result = self.adapter.execute("toggle_pointer_mode")
        assert result.success is True
        self.mock_pya.hotkey.assert_called_once_with("alt", "f2")

    def test_raise_hand_uses_alt_y(self):
        result = self.adapter.execute("raise_hand")
        assert result.success is True
        self.mock_pya.hotkey.assert_called_once_with("alt", "y")

    def test_unknown_command_returns_failure(self):
        result = self.adapter.execute("does_not_exist")
        assert result.success is False


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
    cmd_set = (
        allowed
        if allowed is not None
        else {"next_slide", "previous_slide", "raise_hand"}
    )
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
