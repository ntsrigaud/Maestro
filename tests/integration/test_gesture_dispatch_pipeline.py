"""Integration tests: full gesture → command dispatch pipeline.

These tests exercise the real ``GestureActivationController`` +
``CommandDispatcher`` + ``PresentationAdapter`` chain without triggering any
OS-level side-effects.  ``pyautogui`` is patched at the module level of
``keyboard_adapter`` so the allowlist logic still runs, but no actual
OS events are sent.
"""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from maestro.domain.entities import ExecutableCommand, StabilizedPrediction
from maestro.domain.enums import RuntimeMode
from maestro.infrastructure.dispatch.app_adapters.keyboard_adapter import (
    KeyboardAdapter,
)
from maestro.infrastructure.dispatch.app_adapters.presentation_adapter import (
    PresentationAdapter,
)
from maestro.infrastructure.dispatch.command_dispatcher import (
    CommandDispatcher,
)
from maestro.pipeline.activation import GestureActivationController


@pytest.fixture(autouse=True)
def _force_presentation_scope_for_existing_pipeline_tests():
    """Existing pipeline tests validate mapping/routing, not foreground app detection."""
    with patch.object(KeyboardAdapter, "_ensure_presentation_scope", return_value=True):
        yield


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

_COMMAND_MAP: dict[str, str] = {
    "swiping_up": "start_presentation",
    "swiping_down": "stop_presentation",
    "swiping_right": "next_slide",
    "swiping_left": "previous_slide",
    "zooming_in_full_hand": "zoom_in_view",
    "zooming_out_full_hand": "zoom_out_view",
    "fist": "erase_annotations",
    "point_one": "mouse_pointer",
    "point_two": "mouse_pointer",
    "unknown": "no_action",
}

# Kept as a timing base constant so that test timestamps are non-trivially
# non-zero.  The value no longer relates to any hold duration.
_T_BASE_MS = 500
_IDLE_TIMEOUT_MS = 3000
_COOLDOWN_MS = 800

# Legacy alias — used in TestEdgeTriggerSingleDispatch test bodies; kept to
# avoid touching every timestamp expression in those tests.
_ACTIVATION_HOLD_MS = _T_BASE_MS


def _pred(label: str, accepted: bool = True, confidence: float = 0.95) -> StabilizedPrediction:
    return StabilizedPrediction(label=label, confidence=confidence, accepted=accepted)


def _make_controller() -> GestureActivationController:
    return GestureActivationController(
        command_map=_COMMAND_MAP,
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
    adapter = PresentationAdapter(keyboard=kb)
    cmd_set = allowed if allowed is not None else frozenset(_COMMAND_MAP.values())
    dispatcher = CommandDispatcher(
        adapter=adapter,
        allowed_commands=cmd_set,
        rate_limit_per_window=rate_limit,
        rate_limit_window_ms=window_ms,
    )
    return dispatcher, mock_pya


def _arm_controller(
    controller: GestureActivationController,
    mock_time: MagicMock,  # noqa: ARG001
) -> None:
    """Transition the controller from IDLE to ARMED."""
    controller.rearm()
    assert controller.mode is RuntimeMode.ARMED


# ---------------------------------------------------------------------------
# Test: State machine transitions
# ---------------------------------------------------------------------------


class TestActivationStateMachine:
    def test_starts_in_idle_state(self):
        controller = _make_controller()
        assert controller.mode is RuntimeMode.IDLE

    def test_armed_returns_command_for_known_gesture(self):
        controller = _make_controller()
        controller.rearm()
        with patch.object(type(controller), "_now_ms", staticmethod(lambda: 1000)):
            cmd = controller.route(_pred("swiping_right"))
        assert cmd is not None
        assert cmd.command_id == "next_slide"
        assert cmd.source_gesture == "swiping_right"

    def test_command_triggers_cooldown(self):
        controller = _make_controller()
        controller.rearm()
        with patch.object(type(controller), "_now_ms", staticmethod(lambda: 0)):
            controller.route(_pred("swiping_right"))
        assert controller.mode is RuntimeMode.COOLDOWN

    def test_cooldown_expires_back_to_armed(self):
        controller = _make_controller()
        controller.rearm()
        with patch.object(type(controller), "_now_ms", staticmethod(lambda: 0)):
            controller.route(_pred("swiping_right"))
        # unknown resets edge trigger; after cooldown a different gesture fires.
        with patch.object(
            type(controller),
            "_now_ms",
            staticmethod(lambda: _COOLDOWN_MS + 10),
        ):
            cmd = controller.route(_pred("swiping_left"))
        assert cmd is not None
        assert cmd.command_id == "previous_slide"

    def test_idle_on_missing_hands(self):
        # Patch _now_ms during construction AND rearm() so _last_active_ms starts at 0.
        with patch.object(GestureActivationController, "_now_ms", staticmethod(lambda: 0)):
            controller = _make_controller()
            controller.rearm()  # IDLE → ARMED; _last_active_ms = 0 while still patched
        assert controller.mode is RuntimeMode.ARMED

        # Simulate no gesture for longer than idle timeout
        with patch.object(
            type(controller),
            "_now_ms",
            staticmethod(lambda: _IDLE_TIMEOUT_MS + 200),
        ):
            controller.route(_pred("unknown", accepted=False))
        assert controller.mode is RuntimeMode.IDLE

    def test_unmapped_gesture_does_not_dispatch(self):
        controller = _make_controller()
        controller.rearm()
        with patch.object(type(controller), "_now_ms", staticmethod(lambda: 0)):
            cmd = controller.route(_pred("unknown_gesture_xyz"))
        assert cmd is None
        assert controller.mode is RuntimeMode.ARMED

    def test_force_idle_then_rearm_dispatches_normally(self):
        """force_idle() + rearm() restores full dispatch capability."""
        controller = _make_controller()
        controller.force_idle()
        assert controller.mode is RuntimeMode.IDLE

        controller.rearm()
        assert controller.mode is RuntimeMode.ARMED

        with patch.object(type(controller), "_now_ms", staticmethod(lambda: 100)):
            cmd = controller.route(_pred("swiping_right"))
        assert cmd is not None
        assert cmd.command_id == "next_slide"
        assert cmd.source_gesture == "swiping_right"


# ---------------------------------------------------------------------------
# Test: Full gesture → command dispatch chain
# ---------------------------------------------------------------------------


class TestGestureDispatchPipeline:
    """End-to-end: activation → dispatch with pyautogui patched out."""

    def setup_method(self):
        self.controller = _make_controller()
        self.kb = KeyboardAdapter()
        self.adapter = PresentationAdapter(keyboard=self.kb)
        self.dispatcher = CommandDispatcher(
            adapter=self.adapter,
            allowed_commands=frozenset(_COMMAND_MAP.values()),
            rate_limit_per_window=10,
            rate_limit_window_ms=1000,
        )
        self._patcher = patch(
            "maestro.infrastructure.dispatch.app_adapters.keyboard_adapter.pyautogui"
        )
        self.mock_pya = self._patcher.start()

    def teardown_method(self):
        self._patcher.stop()

    def _arm(self, t0: int = 0) -> None:  # noqa: ARG002
        """Transition controller from IDLE to ARMED."""
        self.controller.rearm()
        assert self.controller.mode is RuntimeMode.ARMED

    def test_swipe_right_dispatches_next_slide(self):
        self._arm()
        with patch.object(
            type(self.controller),
            "_now_ms",
            staticmethod(lambda: _ACTIVATION_HOLD_MS + 50),
        ):
            cmd = self.controller.route(_pred("swiping_right"))
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
            cmd = self.controller.route(_pred("swiping_left"))
        assert cmd is not None
        result = self.dispatcher.dispatch(cmd)
        assert result.success is True
        self.mock_pya.hotkey.assert_called_once_with("left")

    def test_swipe_up_dispatches_start_presentation(self):
        self._arm()
        with patch.object(
            type(self.controller),
            "_now_ms",
            staticmethod(lambda: _ACTIVATION_HOLD_MS + 50),
        ):
            cmd = self.controller.route(_pred("swiping_up"))
        assert cmd is not None
        result = self.dispatcher.dispatch(cmd)
        assert result.success is True
        # Platform-specific shortcut (Cmd+Return on macOS, F5 elsewhere).
        self.mock_pya.hotkey.assert_called_once()

    def test_command_dispatched_exactly_once_per_gesture(self):
        self._arm()
        with patch.object(
            type(self.controller),
            "_now_ms",
            staticmethod(lambda: _ACTIVATION_HOLD_MS + 50),
        ):
            cmd = self.controller.route(_pred("swiping_right"))
        # During cooldown the same gesture produces no second command
        with patch.object(
            type(self.controller),
            "_now_ms",
            staticmethod(lambda: _ACTIVATION_HOLD_MS + 100),
        ):
            cmd2 = self.controller.route(_pred("swiping_right"))
        assert cmd is not None
        assert cmd2 is None  # cooldown suppresses repeat

    def test_allowlist_blocks_unknown_command_id(self):
        """Even if a command somehow bypasses the activation controller,
        the dispatcher allowlist is the last line of defense."""
        from maestro.domain.entities import ExecutableCommand

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
            cmd = self.controller.route(_pred("swiping_right"))
        assert cmd is not None
        result = self.dispatcher.dispatch(cmd)
        assert result.success is False  # graceful failure


# ---------------------------------------------------------------------------
# Test: Rate limiting across repeated commands
# ---------------------------------------------------------------------------


class TestIntegrationRateLimiting:
    def test_burst_commands_beyond_limit_are_dropped(self):
        kb = KeyboardAdapter()
        adapter = PresentationAdapter(keyboard=kb)
        dispatcher = CommandDispatcher(
            adapter=adapter,
            allowed_commands=frozenset(_COMMAND_MAP.values()),
            rate_limit_per_window=2,
            rate_limit_window_ms=5_000,
        )

        from maestro.domain.entities import ExecutableCommand

        results = []
        with patch("maestro.infrastructure.dispatch.app_adapters.keyboard_adapter.pyautogui"):
            for i in range(5):
                cmd = ExecutableCommand(
                    command_id="next_slide",
                    source_gesture="swiping_right",
                    issued_at_ms=i * 10,
                )
                results.append(dispatcher.dispatch(cmd))

        successes = [r for r in results if r.success]
        dropped = [r for r in results if not r.success]
        assert len(successes) == 2
        assert len(dropped) == 3
        assert all("rate_limited" in (r.detail or "") for r in dropped)


# ---------------------------------------------------------------------------
# Test: Edge-trigger single-dispatch guarantee
# ---------------------------------------------------------------------------


class TestEdgeTriggerSingleDispatch:
    """Verify the 'one gesture → one command' invariant introduced by edge-
    trigger detection.

    Scenario modelled:
        unknown, unknown, swiping_right, swiping_right, swiping_right, unknown
                                       ↑
                                   dispatch here (rising edge)
                                                  ↑↑↑
                                           suppressed (same label)
    """

    def _make_armed_controller(
        self, edge_trigger_enabled: bool = True
    ) -> GestureActivationController:
        ctrl = GestureActivationController(
            command_map=_COMMAND_MAP,
            idle_timeout_ms=300_000,
            command_cooldown_ms=_COOLDOWN_MS,
            edge_trigger_enabled=edge_trigger_enabled,
        )
        ctrl.rearm()
        assert ctrl.mode is RuntimeMode.ARMED
        return ctrl

    # ------------------------------------------------------------------
    # Core invariant: one swipe → one command
    # ------------------------------------------------------------------

    def test_single_swipe_dispatches_exactly_once(self):
        """A stream of repeated swiping_right predictions (simulating overlapping
        inference windows during a single slow swipe) yields only one command."""
        ctrl = self._make_armed_controller()
        t_base = _ACTIVATION_HOLD_MS + 50

        commands = []
        # Simulate 8 consecutive accepted swiping_right predictions (one per
        # 100 ms window at step=3, 30 FPS), as produced by a 0.8-second swipe.
        for i in range(8):
            t = t_base + i * 100
            with patch.object(type(ctrl), "_now_ms", staticmethod(lambda _t=t: _t)):
                cmd = ctrl.route(_pred("swiping_right"))
            if cmd is not None:
                commands.append(cmd)

        assert len(commands) == 1, (
            f"Expected exactly 1 command but got {len(commands)}. "
            "Edge-trigger failed to suppress repeated predictions."
        )
        assert commands[0].command_id == "next_slide"

    def test_gesture_resets_after_unknown_then_redispatches(self):
        """After the gesture transitions through 'unknown', the next accepted
        swiping_right occurrence is treated as a new gesture and fires again."""
        ctrl = self._make_armed_controller()
        t = _ACTIVATION_HOLD_MS + 50

        # First swiping_right — fires.
        with patch.object(type(ctrl), "_now_ms", staticmethod(lambda: t)):
            cmd1 = ctrl.route(_pred("swiping_right"))
        assert cmd1 is not None
        assert cmd1.command_id == "next_slide"

        # Hand returns to unknown — resets the edge-trigger.
        t_unknown = t + _COOLDOWN_MS + 10
        with patch.object(type(ctrl), "_now_ms", staticmethod(lambda: t_unknown)):
            no_cmd = ctrl.route(_pred("unknown", accepted=False))
        assert no_cmd is None

        # Second distinct swiping_right (new gesture) — should fire again.
        t2 = t_unknown + 50
        with patch.object(type(ctrl), "_now_ms", staticmethod(lambda: t2)):
            cmd2 = ctrl.route(_pred("swiping_right"))
        assert cmd2 is not None
        assert cmd2.command_id == "next_slide"

    def test_repeated_swipe_through_expired_cooldown_still_suppressed(self):
        """The most critical regression: after COOLDOWN expires, if the swipe
        gesture is still recognized (slow swipe or EMA lag), the edge-trigger
        prevents a second dispatch because the label has not changed."""
        ctrl = self._make_armed_controller()
        t_base = _ACTIVATION_HOLD_MS + 50

        # First swiping_right — fires, enters COOLDOWN.
        with patch.object(type(ctrl), "_now_ms", staticmethod(lambda: t_base)):
            cmd1 = ctrl.route(_pred("swiping_right"))
        assert cmd1 is not None
        assert ctrl.mode is RuntimeMode.COOLDOWN

        # Cooldown expires, back to ARMED.  Gesture is still recognized
        # (slow swipe or EMA memory — the label has not changed).
        t_after_cool = t_base + _COOLDOWN_MS + 10
        with patch.object(type(ctrl), "_now_ms", staticmethod(lambda: t_after_cool)):
            cmd2 = ctrl.route(_pred("swiping_right"))
        assert ctrl.mode is RuntimeMode.ARMED
        assert cmd2 is None, (
            "Edge-trigger must suppress re-dispatch when the gesture label "
            "has not changed since the previous prediction."
        )

    def test_different_gesture_after_cooldown_does_dispatch(self):
        """A *different* gesture after cooldown must fire normally — edge
        triggering should not block genuinely new command gestures."""
        ctrl = self._make_armed_controller()
        t_base = _ACTIVATION_HOLD_MS + 50

        # swiping_right fires.
        with patch.object(type(ctrl), "_now_ms", staticmethod(lambda: t_base)):
            cmd1 = ctrl.route(_pred("swiping_right"))
        assert cmd1 is not None

        # After cooldown, a swiping_left arrives — different label → fires.
        t_after = t_base + _COOLDOWN_MS + 10
        with patch.object(type(ctrl), "_now_ms", staticmethod(lambda: t_after)):
            cmd2 = ctrl.route(_pred("swiping_left"))
        assert cmd2 is not None
        assert cmd2.command_id == "previous_slide"

    def test_edge_trigger_disabled_falls_back_to_cooldown_only(self):
        """When edge_trigger_enabled=False, the controller falls back to
        legacy cooldown-only behaviour: same gesture dispatches again after
        the cooldown window."""
        ctrl = self._make_armed_controller(edge_trigger_enabled=False)
        t_base = _ACTIVATION_HOLD_MS + 50

        with patch.object(type(ctrl), "_now_ms", staticmethod(lambda: t_base)):
            cmd1 = ctrl.route(_pred("swiping_right"))
        assert cmd1 is not None

        # After cooldown, same gesture fires again (legacy behaviour).
        t_after = t_base + _COOLDOWN_MS + 10
        with patch.object(type(ctrl), "_now_ms", staticmethod(lambda: t_after)):
            cmd2 = ctrl.route(_pred("swiping_right"))
        assert cmd2 is not None, (
            "With edge_trigger_enabled=False the cooldown-only guard should "
            "allow re-dispatch after the cooldown window expires."
        )

    # ------------------------------------------------------------------
    # Full pipeline: prediction stream → dispatcher
    # ------------------------------------------------------------------

    def test_full_pipeline_single_swipe_one_dispatch(self):
        """End-to-end: a realistic prediction stream for one swipe gesture
        (unknown × 2, swiping_right × 5, unknown × 2) calls pyautogui exactly
        once."""
        ctrl = self._make_armed_controller()
        kb = KeyboardAdapter()
        adapter = PresentationAdapter(keyboard=kb)
        dispatcher = CommandDispatcher(
            adapter=adapter,
            allowed_commands=frozenset(_COMMAND_MAP.values()),
            rate_limit_per_window=10,
            rate_limit_window_ms=5000,
        )

        prediction_stream = [
            _pred("unknown", accepted=False),
            _pred("unknown", accepted=False),
            _pred("swiping_right"),  # rising edge → dispatch
            _pred("swiping_right"),  # suppressed by edge-trigger
            _pred("swiping_right"),  # suppressed
            _pred("swiping_right"),  # suppressed
            _pred("swiping_right"),  # suppressed
            _pred("unknown", accepted=False),
            _pred("unknown", accepted=False),
        ]

        dispatched = []
        with patch(
            "maestro.infrastructure.dispatch.app_adapters.keyboard_adapter.pyautogui"
        ) as mock_pya:
            t = _ACTIVATION_HOLD_MS + 50
            for pred in prediction_stream:
                with patch.object(type(ctrl), "_now_ms", staticmethod(lambda _t=t: _t)):
                    cmd = ctrl.route(pred)
                if cmd is not None:
                    result = dispatcher.dispatch(cmd)
                    if result.success:
                        dispatched.append(cmd)
                t += 100  # 100 ms per window

        assert len(dispatched) == 1, f"Expected 1 dispatch but got {len(dispatched)}"
        assert dispatched[0].command_id == "next_slide"
        mock_pya.hotkey.assert_called_once_with("right")

    def test_temporal_voting_buildup_does_not_consume_rising_edge(self):
        """Regression test for the _prev_label / _last_dispatched_label bug.

        With vote_window=3, the stabiliser emits 2 *unaccepted* frames of a
        new gesture before the first ``accepted=True`` frame.  The old
        ``_prev_label`` implementation advanced the tracker on every call
        (including uanccepted calls), so by the time the first eligible
        accepted frame arrived, ``prev_label`` already equalled the gesture
        label and the edge-trigger suppressed it.  Every second gesture in a
        session was silently dropped.

        The fix: compare against ``_last_dispatched_label`` (only updated on
        actual dispatch, reset to unknown on neutral prediction).  Unaccepted
        frames must never consume the rising edge.
        """
        ctrl = self._make_armed_controller()
        t_base = _ACTIVATION_HOLD_MS + 50

        # --- First gesture: swiping_right fires normally ---
        with patch.object(type(ctrl), "_now_ms", staticmethod(lambda: t_base)):
            cmd1 = ctrl.route(_pred("swiping_right"))
        assert cmd1 is not None
        assert cmd1.command_id == "next_slide"

        # --- COOLDOWN + reset: gesture ends, hand goes neutral ---
        t_rest = t_base + _COOLDOWN_MS + 20
        with patch.object(type(ctrl), "_now_ms", staticmethod(lambda: t_rest)):
            ctrl.route(_pred("unknown", accepted=False))

        # --- Second gesture: swiping_left builds up via temporal voting ---
        # Frames 1 and 2: stabiliser not yet temporally consistent → accepted=False
        t1 = t_rest + 50
        with patch.object(type(ctrl), "_now_ms", staticmethod(lambda: t1)):
            no_cmd = ctrl.route(_pred("swiping_left", accepted=False))
        assert no_cmd is None

        t2 = t1 + 100
        with patch.object(type(ctrl), "_now_ms", staticmethod(lambda: t2)):
            no_cmd = ctrl.route(_pred("swiping_left", accepted=False))
        assert no_cmd is None

        # Frame 3: vote window satisfied → accepted=True.  The rising edge
        # must fire here; unaccepted frames must not have consumed it.
        t3 = t2 + 100
        with patch.object(type(ctrl), "_now_ms", staticmethod(lambda: t3)):
            cmd2 = ctrl.route(_pred("swiping_left", accepted=True))
        assert cmd2 is not None, (
            "Second gesture suppressed. Unaccepted temporal-voting frames "
            "must not advance the edge-trigger counter (_prev_label regression)."
        )
        assert cmd2.command_id == "previous_slide"

    def test_sequential_swipe_right_then_left_then_right(self):
        """Three sequential distinct gestures must each fire exactly once,
        regardless of whether the same label is re-used later."""
        ctrl = self._make_armed_controller()
        t = _ACTIVATION_HOLD_MS + 50

        def _route(label: str, accepted: bool = True) -> ExecutableCommand | None:
            nonlocal t
            with patch.object(type(ctrl), "_now_ms", staticmethod(lambda _t=t: _t)):
                cmd = ctrl.route(_pred(label, accepted))
            t += 150
            return cmd

        # Gesture 1: swiping_right
        cmd1 = _route("swiping_right")
        assert cmd1 is not None and cmd1.command_id == "next_slide"

        # Cooldown + neutral
        for _ in range(8):
            _route("unknown", accepted=False)

        # Gesture 2: temporal buildup (2 unaccepted) then accepted
        assert _route("swiping_left", accepted=False) is None
        assert _route("swiping_left", accepted=False) is None
        cmd2 = _route("swiping_left", accepted=True)
        assert cmd2 is not None and cmd2.command_id == "previous_slide"

        # Cooldown + neutral
        for _ in range(8):
            _route("unknown", accepted=False)

        # Gesture 3: same label as gesture 1 — must fire again after neutral reset
        assert _route("swiping_right", accepted=False) is None
        assert _route("swiping_right", accepted=False) is None
        cmd3 = _route("swiping_right", accepted=True)
        assert cmd3 is not None and cmd3.command_id == "next_slide"


# ---------------------------------------------------------------------------
# Test: Gaze gate integration
# ---------------------------------------------------------------------------


class TestGazeGateIntegration:
    """Verify the gaze gate layer in context of the gesture dispatch chain.

    The gaze gate lives in ``Runtime.run()`` and bypasses all pipeline stages
    when gaze is absent.  These tests simulate that gate at the controller
    boundary:

    1. ``NullGazeDetector`` always returns ``True`` — pipeline behaves
       identically to the pre-gaze baseline.

    2. When ``force_idle()`` is called (simulating the runtime calling it on
       grace-period expiry), the controller transitions to IDLE and no
       subsequent commands are emitted from that state.

    3. The transition from ARMED to IDLE on gaze loss and back to ARMED on
       a fresh hold cycle works end-to-end with the dispatcher.
    """

    def setup_method(self) -> None:
        self.controller = _make_controller()
        self.kb = KeyboardAdapter()
        self.adapter = PresentationAdapter(keyboard=self.kb)
        self.dispatcher = CommandDispatcher(
            adapter=self.adapter,
            allowed_commands=frozenset(_COMMAND_MAP.values()),
            rate_limit_per_window=10,
            rate_limit_window_ms=1000,
        )
        self._patcher = patch(
            "maestro.infrastructure.dispatch.app_adapters.keyboard_adapter.pyautogui"
        )
        self.mock_pya = self._patcher.start()

    def teardown_method(self) -> None:
        self._patcher.stop()

    def _arm(self, t0: int = 0) -> None:  # noqa: ARG002
        """Transition controller from IDLE to ARMED."""
        self.controller.rearm()
        assert self.controller.mode is RuntimeMode.ARMED

    # -- NullGazeDetector -----------------------------------------------------

    def test_null_detector_update_always_permits_gaze(self) -> None:
        """NullGazeDetector.update() must return True on every call, preserving
        the pre-gaze pipeline baseline with zero behavioural change."""
        from maestro.domain.entities import FramePacket
        from maestro.infrastructure.gaze.null_detector import NullGazeDetector

        null_det = NullGazeDetector()
        frame = FramePacket(
            frame=np.zeros((480, 640, 3), dtype=np.uint8),
            timestamp_ms=0.0,
            frame_id=0,
        )
        for _ in range(20):
            assert null_det.update(frame) is True

    def test_null_detector_enables_full_dispatch_pipeline(self) -> None:
        """With NullGazeDetector in place the command pipeline operates normally."""
        self._arm()
        with patch.object(
            type(self.controller),
            "_now_ms",
            staticmethod(lambda: _ACTIVATION_HOLD_MS + 50),
        ):
            cmd = self.controller.route(_pred("swiping_right"))
        assert cmd is not None
        result = self.dispatcher.dispatch(cmd)
        assert result.success is True
        self.mock_pya.hotkey.assert_called_once_with("right")  # next_slide

    # -- Gaze loss → force_idle -----------------------------------------------

    def test_force_idle_on_gaze_loss_prevents_command_dispatch(self) -> None:
        """Simulates runtime: controller ARMED, gaze grace expires → force_idle(),
        gesture arrives while IDLE → no command emitted."""
        self._arm()

        # Gaze grace period expires: runtime calls force_idle().
        self.controller.force_idle()
        assert self.controller.mode is RuntimeMode.IDLE

        # A command gesture arrives, but the pipeline is idle (gaze absent).
        with patch.object(
            type(self.controller),
            "_now_ms",
            staticmethod(lambda: _ACTIVATION_HOLD_MS + 200),
        ):
            cmd = self.controller.route(_pred("swiping_right"))
        assert cmd is None
        self.mock_pya.hotkey.assert_not_called()

    def test_force_idle_prevents_dispatch_for_all_mapped_gestures(self) -> None:
        """Every mapped gesture is suppressed after force_idle(), not just
        swiping_right."""
        self._arm()
        self.controller.force_idle()

        t = _ACTIVATION_HOLD_MS + 100
        for gesture in (
            "swiping_right",
            "swiping_left",
            "swiping_up",
            "swiping_down",
            "zooming_in_full_hand",
            "zooming_out_full_hand",
        ):
            with patch.object(type(self.controller), "_now_ms", staticmethod(lambda _t=t: _t)):
                cmd = self.controller.route(_pred(gesture))
            assert cmd is None, f"Expected no command for {gesture} when gaze is absent"
            t += 50
        self.mock_pya.hotkey.assert_not_called()

    # -- Re-arming after gaze restored ----------------------------------------

    def test_rearm_after_gaze_restored_dispatches_correctly(self) -> None:
        """After gaze is lost (force_idle) and the user re-establishes gaze,
        calling rearm() transitions back to ARMED and the pipeline dispatches."""
        self._arm()

        # Gaze lost.
        self.controller.force_idle()
        assert self.controller.mode is RuntimeMode.IDLE

        # Gaze re-established: runtime calls rearm().
        self.controller.rearm()
        assert self.controller.mode is RuntimeMode.ARMED

        # Gesture dispatches normally after re-arming.
        with patch.object(
            type(self.controller),
            "_now_ms",
            staticmethod(lambda: 600),
        ):
            cmd = self.controller.route(_pred("swiping_up"))
        assert cmd is not None
        result = self.dispatcher.dispatch(cmd)
        assert result.success is True
        # start_presentation uses platform-specific slideshow keys.
        self.mock_pya.hotkey.assert_called_once()  # start_presentation

    def test_swipe_down_does_not_dispatch_when_gaze_absent(self) -> None:
        """swiping_down must be suppressed when the gaze gate is closed (IDLE)."""
        self._arm()
        self.controller.force_idle()
        with patch.object(
            type(self.controller),
            "_now_ms",
            staticmethod(lambda: _ACTIVATION_HOLD_MS + 50),
        ):
            cmd = self.controller.route(_pred("swiping_down"))
        assert cmd is None
        self.mock_pya.hotkey.assert_not_called()
