"""Integration tests: full gesture → command dispatch pipeline.

These tests exercise the real ``GestureActivationController`` +
``CommandDispatcher`` + ``PowerPointAdapter`` chain without triggering any
OS-level side-effects.  ``pyautogui`` is patched at the module level of
``keyboard_adapter`` so the allowlist logic still runs, but no actual
keystrokes are sent.
"""

from __future__ import annotations

from unittest.mock import MagicMock, patch

from gesture_control.domain.entities import StabilizedPrediction
from gesture_control.domain.enums import RuntimeMode
from gesture_control.infrastructure.dispatch.app_adapters.keyboard_adapter import (
    KeyboardAdapter,
)
from gesture_control.infrastructure.dispatch.app_adapters.powerpoint_adapter import (
    PowerPointAdapter,
)
from gesture_control.infrastructure.dispatch.command_dispatcher import (
    CommandDispatcher,
)
from gesture_control.pipeline.activation import GestureActivationController

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

_COMMAND_MAP: dict[str, str] = {
    "swipe_right": "next_slide",
    "swipe_left": "previous_slide",
    "swipe_up": "raise_hand",
    "swipe_down": "start_presentation",
    "zoom_in": "toggle_pointer_mode",
    "zoom_out": "exit_presentation",
}

_ACTIVATION_GESTURE = "open_palm_hold"
_ACTIVATION_HOLD_MS = 500
_IDLE_TIMEOUT_MS = 3000
_COOLDOWN_MS = 800


def _pred(
    label: str, accepted: bool = True, confidence: float = 0.95
) -> StabilizedPrediction:
    return StabilizedPrediction(label=label, confidence=confidence, accepted=accepted)


def _make_controller() -> GestureActivationController:
    return GestureActivationController(
        activation_gesture=_ACTIVATION_GESTURE,
        command_map=_COMMAND_MAP,
        activation_hold_ms=_ACTIVATION_HOLD_MS,
        idle_timeout_ms=_IDLE_TIMEOUT_MS,
        command_cooldown_ms=_COOLDOWN_MS,
    )


def _make_dispatcher(
    allowed: set[str] | None = None,
    rate_limit: int = 10,
    window_ms: int = 1000,
) -> tuple[CommandDispatcher, MagicMock]:
    mock_pya = MagicMock()
    kb = KeyboardAdapter()
    adapter = PowerPointAdapter(keyboard=kb)
    cmd_set = allowed if allowed is not None else frozenset(_COMMAND_MAP.values())
    dispatcher = CommandDispatcher(
        adapter=adapter,
        allowed_commands=cmd_set,
        rate_limit_per_window=rate_limit,
        rate_limit_window_ms=window_ms,
    )
    return dispatcher, mock_pya


def _arm_controller(
    controller: GestureActivationController, mock_time: MagicMock
) -> None:
    """Drive controller from IDLE → ARMED via activation hold."""
    t0 = 1000
    mock_time.return_value = t0
    controller.route(_pred(_ACTIVATION_GESTURE))  # start hold timer
    mock_time.return_value = t0 + _ACTIVATION_HOLD_MS + 1
    controller.route(_pred(_ACTIVATION_GESTURE))  # completes hold → ARMED


# ---------------------------------------------------------------------------
# Test: State machine transitions
# ---------------------------------------------------------------------------


class TestActivationStateMachine:
    def test_idle_to_armed_on_hold(self):
        controller = _make_controller()
        with patch.object(type(controller), "_now_ms", staticmethod(lambda: 1000)):
            controller.route(_pred(_ACTIVATION_GESTURE))
        assert controller.mode is RuntimeMode.IDLE  # still waiting for hold

        with patch.object(
            type(controller),
            "_now_ms",
            staticmethod(lambda: 1000 + _ACTIVATION_HOLD_MS + 1),
        ):
            controller.route(_pred(_ACTIVATION_GESTURE))
        assert controller.mode is RuntimeMode.ARMED

    def test_armed_returns_command_for_known_gesture(self):
        controller = _make_controller()
        with patch.object(type(controller), "_now_ms", staticmethod(lambda: 1000)):
            controller.route(_pred(_ACTIVATION_GESTURE))
        with patch.object(type(controller), "_now_ms", staticmethod(lambda: 1700)):
            controller.route(_pred(_ACTIVATION_GESTURE))
        assert controller.mode is RuntimeMode.ARMED

        with patch.object(type(controller), "_now_ms", staticmethod(lambda: 1800)):
            cmd = controller.route(_pred("swipe_right"))
        assert cmd is not None
        assert cmd.command_id == "next_slide"
        assert cmd.source_gesture == "swipe_right"

    def test_command_triggers_cooldown(self):
        controller = _make_controller()
        with patch.object(type(controller), "_now_ms", staticmethod(lambda: 0)):
            controller.route(_pred(_ACTIVATION_GESTURE))
        with patch.object(
            type(controller), "_now_ms", staticmethod(lambda: _ACTIVATION_HOLD_MS + 1)
        ):
            controller.route(_pred(_ACTIVATION_GESTURE))
        with patch.object(
            type(controller), "_now_ms", staticmethod(lambda: _ACTIVATION_HOLD_MS + 2)
        ):
            controller.route(_pred("swipe_right"))
        assert controller.mode is RuntimeMode.COOLDOWN

    def test_cooldown_expires_back_to_armed(self):
        controller = _make_controller()
        with patch.object(type(controller), "_now_ms", staticmethod(lambda: 0)):
            controller.route(_pred(_ACTIVATION_GESTURE))
        with patch.object(
            type(controller), "_now_ms", staticmethod(lambda: _ACTIVATION_HOLD_MS + 1)
        ):
            controller.route(_pred(_ACTIVATION_GESTURE))
        with patch.object(
            type(controller), "_now_ms", staticmethod(lambda: _ACTIVATION_HOLD_MS + 2)
        ):
            controller.route(_pred("swipe_right"))
        with patch.object(
            type(controller),
            "_now_ms",
            staticmethod(lambda: _ACTIVATION_HOLD_MS + _COOLDOWN_MS + 10),
        ):
            cmd = controller.route(_pred("swipe_left"))
        assert cmd is not None
        assert cmd.command_id == "previous_slide"

    def test_idle_on_missing_hands(self):
        controller = _make_controller()
        with patch.object(type(controller), "_now_ms", staticmethod(lambda: 0)):
            controller.route(_pred(_ACTIVATION_GESTURE))
        with patch.object(
            type(controller), "_now_ms", staticmethod(lambda: _ACTIVATION_HOLD_MS + 100)
        ):
            controller.route(_pred(_ACTIVATION_GESTURE))
        assert controller.mode is RuntimeMode.ARMED

        # Simulate no gesture for longer than idle timeout
        with patch.object(
            type(controller),
            "_now_ms",
            staticmethod(lambda: _ACTIVATION_HOLD_MS + _IDLE_TIMEOUT_MS + 200),
        ):
            controller.route(_pred("unknown", accepted=False))
        assert controller.mode is RuntimeMode.IDLE

    def test_unmapped_gesture_does_not_dispatch(self):
        controller = _make_controller()
        with patch.object(type(controller), "_now_ms", staticmethod(lambda: 0)):
            controller.route(_pred(_ACTIVATION_GESTURE))
        with patch.object(
            type(controller), "_now_ms", staticmethod(lambda: _ACTIVATION_HOLD_MS + 1)
        ):
            controller.route(_pred(_ACTIVATION_GESTURE))
        with patch.object(
            type(controller), "_now_ms", staticmethod(lambda: _ACTIVATION_HOLD_MS + 2)
        ):
            cmd = controller.route(_pred("unknown_gesture_xyz"))
        assert cmd is None
        assert controller.mode is RuntimeMode.ARMED


# ---------------------------------------------------------------------------
# Test: Full gesture → command dispatch chain
# ---------------------------------------------------------------------------


class TestGestureDispatchPipeline:
    """End-to-end: activation → dispatch with pyautogui patched out."""

    def setup_method(self):
        self.controller = _make_controller()
        self.kb = KeyboardAdapter()
        self.adapter = PowerPointAdapter(keyboard=self.kb)
        self.dispatcher = CommandDispatcher(
            adapter=self.adapter,
            allowed_commands=frozenset(_COMMAND_MAP.values()),
            rate_limit_per_window=10,
            rate_limit_window_ms=1000,
        )
        self._patcher = patch(
            "gesture_control.infrastructure.dispatch.app_adapters.keyboard_adapter.pyautogui"
        )
        self.mock_pya = self._patcher.start()

    def teardown_method(self):
        self._patcher.stop()

    def _arm(self, t0: int = 0) -> None:
        with patch.object(type(self.controller), "_now_ms", staticmethod(lambda: t0)):
            self.controller.route(_pred(_ACTIVATION_GESTURE))
        with patch.object(
            type(self.controller),
            "_now_ms",
            staticmethod(lambda: t0 + _ACTIVATION_HOLD_MS + 1),
        ):
            self.controller.route(_pred(_ACTIVATION_GESTURE))

    def test_swipe_right_dispatches_next_slide(self):
        self._arm()
        with patch.object(
            type(self.controller),
            "_now_ms",
            staticmethod(lambda: _ACTIVATION_HOLD_MS + 50),
        ):
            cmd = self.controller.route(_pred("swipe_right"))
        assert cmd is not None
        result = self.dispatcher.dispatch(cmd)
        assert result.success is True
        self.mock_pya.hotkey.assert_called_once_with("right")

    def test_swipe_left_dispatches_previous_slide(self):
        self._arm()
        with patch.object(
            type(self.controller),
            "_now_ms",
            staticmethod(lambda: _ACTIVATION_HOLD_MS + 50),
        ):
            cmd = self.controller.route(_pred("swipe_left"))
        assert cmd is not None
        result = self.dispatcher.dispatch(cmd)
        assert result.success is True
        self.mock_pya.hotkey.assert_called_once_with("left")

    def test_swipe_up_dispatches_raise_hand(self):
        self._arm()
        with patch.object(
            type(self.controller),
            "_now_ms",
            staticmethod(lambda: _ACTIVATION_HOLD_MS + 50),
        ):
            cmd = self.controller.route(_pred("swipe_up"))
        assert cmd is not None
        result = self.dispatcher.dispatch(cmd)
        assert result.success is True
        self.mock_pya.hotkey.assert_called_once_with(
            "ctrl", "shift", "k"
        )  # PowerPoint raise_hand

    def test_command_dispatched_exactly_once_per_gesture(self):
        self._arm()
        with patch.object(
            type(self.controller),
            "_now_ms",
            staticmethod(lambda: _ACTIVATION_HOLD_MS + 50),
        ):
            cmd = self.controller.route(_pred("swipe_right"))
        # During cooldown the same gesture produces no second command
        with patch.object(
            type(self.controller),
            "_now_ms",
            staticmethod(lambda: _ACTIVATION_HOLD_MS + 100),
        ):
            cmd2 = self.controller.route(_pred("swipe_right"))
        assert cmd is not None
        assert cmd2 is None  # cooldown suppresses repeat

    def test_allowlist_blocks_unknown_command_id(self):
        """Even if a command somehow bypasses the activation controller,
        the dispatcher allowlist is the last line of defense."""
        from gesture_control.domain.entities import ExecutableCommand

        bad_cmd = ExecutableCommand(
            command_id="delete_everything",
            source_gesture="unknown",
            issued_at_ms=0,
        )
        result = self.dispatcher.dispatch(bad_cmd)
        assert result.success is False
        assert "rejected_by_allowlist" in (result.detail or "")
        self.mock_pya.hotkey.assert_not_called()

    def test_adapter_exception_does_not_crash_pipeline(self):
        self.mock_pya.hotkey.side_effect = OSError("display error")
        self._arm()
        with patch.object(
            type(self.controller),
            "_now_ms",
            staticmethod(lambda: _ACTIVATION_HOLD_MS + 50),
        ):
            cmd = self.controller.route(_pred("swipe_right"))
        assert cmd is not None
        result = self.dispatcher.dispatch(cmd)
        assert result.success is False  # graceful failure


# ---------------------------------------------------------------------------
# Test: Rate limiting across repeated commands
# ---------------------------------------------------------------------------


class TestIntegrationRateLimiting:
    def test_burst_commands_beyond_limit_are_dropped(self):
        kb = KeyboardAdapter()
        adapter = PowerPointAdapter(keyboard=kb)
        dispatcher = CommandDispatcher(
            adapter=adapter,
            allowed_commands=frozenset(_COMMAND_MAP.values()),
            rate_limit_per_window=2,
            rate_limit_window_ms=5_000,
        )

        from gesture_control.domain.entities import ExecutableCommand

        results = []
        with patch(
            "gesture_control.infrastructure.dispatch.app_adapters.keyboard_adapter.pyautogui"
        ):
            for i in range(5):
                cmd = ExecutableCommand(
                    command_id="next_slide",
                    source_gesture="swipe_right",
                    issued_at_ms=i * 10,
                )
                results.append(dispatcher.dispatch(cmd))

        successes = [r for r in results if r.success]
        dropped = [r for r in results if not r.success]
        assert len(successes) == 2
        assert len(dropped) == 3
        assert all("rate_limited" in (r.detail or "") for r in dropped)


# ---------------------------------------------------------------------------
# Test: Activation hold and ARMED persistence (IPN regression fixes)
# ---------------------------------------------------------------------------


class TestActivationPersistenceAndHold:
    """Verify the two regressions surfaced after IPN vocabulary migration.

    Fix 1 — Hold timer starts before temporal stability:
        ``open_palm_hold`` requires vote_window=3 in the stabiliser, meaning
        the first two windows are always ``accepted=False``.  The controller
        must begin the hold timer on the first above-threshold prediction,
        not only after the first ``accepted=True``.

    Fix 2 — ARMED state persists during presentation pauses:
        With ``idle_timeout_ms=3000`` the system dropped back to IDLE after
        only 3 seconds of inactivity.  The new default is 300 000 ms (5 min)
        and ``_last_active_ms`` is refreshed on any non-unknown gesture.
    """

    # ------------------------------------------------------------------
    # Fix 1: hold timer starts with accepted=False (temporal_inconsistency)
    # ------------------------------------------------------------------

    def test_hold_timer_starts_before_temporal_stability(self):
        """Hold timer begins on the first above-threshold frame even when
        the stabiliser marks the prediction as not yet temporally stable."""
        controller = GestureActivationController(
            activation_gesture=_ACTIVATION_GESTURE,
            command_map=_COMMAND_MAP,
            activation_hold_ms=_ACTIVATION_HOLD_MS,
            idle_timeout_ms=_IDLE_TIMEOUT_MS,
            command_cooldown_ms=_COOLDOWN_MS,
        )
        # Window 1: confidence above threshold but not yet temporally stable.
        unaccepted = StabilizedPrediction(
            label=_ACTIVATION_GESTURE,
            confidence=0.93,
            accepted=False,
            reason="temporal_inconsistency",
        )
        with patch.object(type(controller), "_now_ms", staticmethod(lambda: 0)):
            controller.route(unaccepted)
        assert controller.mode is RuntimeMode.IDLE  # hold timer started but not expired

        # Window 2 (still not accepted) — hold duration just short of threshold.
        with patch.object(
            type(controller), "_now_ms", staticmethod(lambda: _ACTIVATION_HOLD_MS - 1)
        ):
            controller.route(unaccepted)
        assert controller.mode is RuntimeMode.IDLE  # not yet expired

        # Window 3 — hold duration exceeds threshold, still not accepted.
        with patch.object(
            type(controller), "_now_ms", staticmethod(lambda: _ACTIVATION_HOLD_MS + 1)
        ):
            controller.route(unaccepted)
        assert controller.mode is RuntimeMode.ARMED  # armed despite never accepted=True

    def test_hold_timer_resets_on_unknown_label(self):
        """If confidence drops to unknown between frames the hold timer resets."""
        controller = GestureActivationController(
            activation_gesture=_ACTIVATION_GESTURE,
            command_map=_COMMAND_MAP,
            activation_hold_ms=_ACTIVATION_HOLD_MS,
            idle_timeout_ms=_IDLE_TIMEOUT_MS,
            command_cooldown_ms=_COOLDOWN_MS,
        )
        activation_pred = StabilizedPrediction(
            label=_ACTIVATION_GESTURE,
            confidence=0.93,
            accepted=False,
            reason="temporal_inconsistency",
        )
        unknown_pred = StabilizedPrediction(
            label="unknown",
            confidence=0.40,
            accepted=False,
            reason="below_threshold_or_unknown",
        )
        # Timer starts at T=0.
        with patch.object(type(controller), "_now_ms", staticmethod(lambda: 0)):
            controller.route(activation_pred)

        # Confidence drops at T=200 — label becomes unknown — timer resets.
        with patch.object(type(controller), "_now_ms", staticmethod(lambda: 200)):
            controller.route(unknown_pred)

        # Activation gesture resumes at T=201 — timer must restart from here.
        with patch.object(type(controller), "_now_ms", staticmethod(lambda: 201)):
            controller.route(activation_pred)

        # One ms before the new hold window expires — still IDLE.
        t_near = 201 + _ACTIVATION_HOLD_MS - 1
        with patch.object(type(controller), "_now_ms", staticmethod(lambda: t_near)):
            controller.route(activation_pred)
        assert controller.mode is RuntimeMode.IDLE

        # One ms after the new hold window expires — ARMED.
        t_arm = 201 + _ACTIVATION_HOLD_MS + 1
        with patch.object(type(controller), "_now_ms", staticmethod(lambda: t_arm)):
            controller.route(activation_pred)
        assert controller.mode is RuntimeMode.ARMED

    # ------------------------------------------------------------------
    # Fix 2: ARMED state persists through extended inactivity
    # ------------------------------------------------------------------

    def test_armed_persists_within_long_idle_timeout(self):
        """System stays ARMED for the duration of idle_timeout without any gesture."""
        long_timeout_ms = 300_000  # 5 minutes
        controller = GestureActivationController(
            activation_gesture=_ACTIVATION_GESTURE,
            command_map=_COMMAND_MAP,
            activation_hold_ms=_ACTIVATION_HOLD_MS,
            idle_timeout_ms=long_timeout_ms,
            command_cooldown_ms=_COOLDOWN_MS,
        )
        with patch.object(type(controller), "_now_ms", staticmethod(lambda: 0)):
            controller.route(_pred(_ACTIVATION_GESTURE))
        with patch.object(
            type(controller), "_now_ms", staticmethod(lambda: _ACTIVATION_HOLD_MS + 1)
        ):
            controller.route(_pred(_ACTIVATION_GESTURE))
        assert controller.mode is RuntimeMode.ARMED

        # 3 minutes later — only unknown predictions (no hand gesture).
        t_3min = _ACTIVATION_HOLD_MS + 180_000
        with patch.object(type(controller), "_now_ms", staticmethod(lambda: t_3min)):
            controller.route(_pred("unknown", accepted=False))
        assert controller.mode is RuntimeMode.ARMED  # still armed after 3 min

        # Just inside the 5-minute window.
        t_5min_minus_1 = _ACTIVATION_HOLD_MS + long_timeout_ms - 1
        with patch.object(
            type(controller), "_now_ms", staticmethod(lambda: t_5min_minus_1)
        ):
            controller.route(_pred("unknown", accepted=False))
        assert controller.mode is RuntimeMode.ARMED  # still within timeout

        # One millisecond past the 5-minute window → IDLE.
        t_expired = _ACTIVATION_HOLD_MS + long_timeout_ms + 2
        with patch.object(type(controller), "_now_ms", staticmethod(lambda: t_expired)):
            controller.route(_pred("unknown", accepted=False))
        assert controller.mode is RuntimeMode.IDLE

    def test_any_gesture_resets_idle_timer(self):
        """A non-unknown gesture (even if not dispatched) resets the armed timer."""
        idle_timeout_ms = 5_000
        controller = GestureActivationController(
            activation_gesture=_ACTIVATION_GESTURE,
            command_map=_COMMAND_MAP,
            activation_hold_ms=_ACTIVATION_HOLD_MS,
            idle_timeout_ms=idle_timeout_ms,
            command_cooldown_ms=_COOLDOWN_MS,
        )
        # Arm at T=0.
        with patch.object(type(controller), "_now_ms", staticmethod(lambda: 0)):
            controller.route(_pred(_ACTIVATION_GESTURE))
        with patch.object(
            type(controller), "_now_ms", staticmethod(lambda: _ACTIVATION_HOLD_MS + 1)
        ):
            controller.route(_pred(_ACTIVATION_GESTURE))
        assert controller.mode is RuntimeMode.ARMED

        # At T=4 000 ms (within 5 s timeout), user makes swipe_right → dispatch.
        t_swipe = 4_000
        with patch.object(type(controller), "_now_ms", staticmethod(lambda: t_swipe)):
            controller.route(_pred("swipe_right"))
        # In cooldown now; _last_active_ms was refreshed to t_swipe.

        # Cooldown ends at t_swipe + cooldown_ms.
        t_after_cool = t_swipe + _COOLDOWN_MS + 1
        with patch.object(
            type(controller), "_now_ms", staticmethod(lambda: t_after_cool)
        ):
            controller.route(_pred("unknown", accepted=False))
        assert controller.mode is RuntimeMode.ARMED

        # 4 999 ms after the swipe — still within the reset window.
        t_near = t_swipe + idle_timeout_ms - 1
        with patch.object(type(controller), "_now_ms", staticmethod(lambda: t_near)):
            controller.route(_pred("unknown", accepted=False))
        assert controller.mode is RuntimeMode.ARMED

        # 5 001 ms after the swipe — idle timeout expires from the swipe.
        t_over = t_swipe + idle_timeout_ms + 2
        with patch.object(type(controller), "_now_ms", staticmethod(lambda: t_over)):
            controller.route(_pred("unknown", accepted=False))
        assert controller.mode is RuntimeMode.IDLE

    def test_activation_gesture_when_armed_does_not_dispatch(self):
        """Re-showing the activation gesture when ARMED keeps the timer alive
        but must never emit a command."""
        controller = GestureActivationController(
            activation_gesture=_ACTIVATION_GESTURE,
            command_map=_COMMAND_MAP,
            activation_hold_ms=_ACTIVATION_HOLD_MS,
            idle_timeout_ms=_IDLE_TIMEOUT_MS,
            command_cooldown_ms=_COOLDOWN_MS,
        )
        with patch.object(type(controller), "_now_ms", staticmethod(lambda: 0)):
            controller.route(_pred(_ACTIVATION_GESTURE))
        with patch.object(
            type(controller), "_now_ms", staticmethod(lambda: _ACTIVATION_HOLD_MS + 1)
        ):
            controller.route(_pred(_ACTIVATION_GESTURE))
        assert controller.mode is RuntimeMode.ARMED

        # Hold palm again while already ARMED — should return None.
        with patch.object(
            type(controller), "_now_ms", staticmethod(lambda: _ACTIVATION_HOLD_MS + 100)
        ):
            cmd = controller.route(_pred(_ACTIVATION_GESTURE))
        assert cmd is None
        assert controller.mode is RuntimeMode.ARMED
