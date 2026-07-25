"""Integration tests: full gesture → command dispatch pipeline.

These tests exercise the real ``GestureActivationController`` +
``CommandDispatcher`` + ``PresentationAdapter`` chain without triggering any
OS-level side-effects.  The OS backend is mocked via Dependency Injection.
"""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest

from maestro.domain.entities import ExecutableCommand, StabilizedPrediction
from maestro.domain.enums import RuntimeMode
from maestro.infrastructure.dispatch.app_adapters.keyboard_adapter import (
    KeyboardAdapter,
    OSKeyboardBackend,
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

# Timing base constant for test timestamps
_T_BASE_MS = 500
_COOLDOWN_MS = 800
_ACTIVATION_HOLD_MS = _T_BASE_MS
_ZOOM_CLICKS = 3


def _pred(label: str, accepted: bool = True, confidence: float = 0.95) -> StabilizedPrediction:
    return StabilizedPrediction(label=label, confidence=confidence, accepted=accepted)


def _make_controller() -> GestureActivationController:
    return GestureActivationController(
        command_map=_COMMAND_MAP,
        command_cooldown_ms=_COOLDOWN_MS,
    )


def _make_adapter(kb: KeyboardAdapter) -> PresentationAdapter:
    return PresentationAdapter(
        keyboard=kb,
        zoom_clicks=_ZOOM_CLICKS,
    )


# ---------------------------------------------------------------------------
# Test: State machine transitions
# ---------------------------------------------------------------------------


class TestActivationStateMachine:
    def test_starts_in_armed_state(self):
        controller = _make_controller()
        assert controller.mode is RuntimeMode.ARMED

    def test_armed_returns_command_for_known_gesture(self):
        controller = _make_controller()
        with patch.object(type(controller), "_now_ms", staticmethod(lambda: 1000)):
            cmd = controller.route(_pred("swiping_right"))
        assert cmd is not None
        assert cmd.command_id == "next_slide"
        assert cmd.source_gesture == "swiping_right"

    def test_command_triggers_cooldown(self):
        controller = _make_controller()
        with patch.object(type(controller), "_now_ms", staticmethod(lambda: 0)):
            controller.route(_pred("swiping_right"))
        assert controller.mode is RuntimeMode.COOLDOWN

    def test_cooldown_expires_back_to_armed(self):
        controller = _make_controller()
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

    def test_unmapped_gesture_does_not_dispatch(self):
        controller = _make_controller()
        with patch.object(type(controller), "_now_ms", staticmethod(lambda: 0)):
            cmd = controller.route(_pred("unknown_gesture_xyz"))
        assert cmd is None
        assert controller.mode is RuntimeMode.ARMED


# ---------------------------------------------------------------------------
# Test: Full gesture → command dispatch chain
# ---------------------------------------------------------------------------


class TestGestureDispatchPipeline:
    """End-to-end: activation → dispatch with mock OS backend."""

    def setup_method(self):
        self.controller = _make_controller()
        self.mock_backend = MagicMock(spec=OSKeyboardBackend)
        self.mock_backend.press_hotkey.return_value = True
        self.kb = KeyboardAdapter(backend=self.mock_backend)
        self.adapter = _make_adapter(self.kb)
        self.dispatcher = CommandDispatcher(
            adapter=self.adapter,
            allowed_commands=frozenset(_COMMAND_MAP.values()),
            rate_limit_per_window=10,
            rate_limit_window_ms=1000,
        )

    def _arm(self) -> None:
        """Verify controller is in ARMED state (default at startup)."""
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
        self.mock_backend.press_hotkey.assert_called_once_with("right")

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
        self.mock_backend.press_hotkey.assert_called_once_with("left")

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
        self.mock_backend.press_hotkey.assert_called_once()

    def test_command_dispatched_exactly_once_per_gesture(self):
        self._arm()
        with patch.object(
            type(self.controller),
            "_now_ms",
            staticmethod(lambda: _ACTIVATION_HOLD_MS + 50),
        ):
            cmd = self.controller.route(_pred("swiping_right"))
        with patch.object(
            type(self.controller),
            "_now_ms",
            staticmethod(lambda: _ACTIVATION_HOLD_MS + 100),
        ):
            cmd2 = self.controller.route(_pred("swiping_right"))
        assert cmd is not None
        assert cmd2 is None  # cooldown suppresses repeat

    def test_allowlist_blocks_unknown_command_id(self):
        bad_cmd = ExecutableCommand(
            command_id="delete_everything",
            source_gesture="unknown",
            issued_at_ms=0,
        )
        result = self.dispatcher.dispatch(bad_cmd)
        assert result.success is False
        assert "rejected_by_allowlist" in (result.detail or "")
        self.mock_backend.press_hotkey.assert_not_called()

    def test_adapter_exception_does_not_crash_pipeline(self):
        self.mock_backend.press_hotkey.side_effect = OSError("display error")
        self._arm()
        with patch.object(
            type(self.controller),
            "_now_ms",
            staticmethod(lambda: _ACTIVATION_HOLD_MS + 50),
        ):
            cmd = self.controller.route(_pred("swiping_right"))
        assert cmd is not None
        result = self.dispatcher.dispatch(cmd)
        assert result.success is False


# ---------------------------------------------------------------------------
# Test: Rate limiting across repeated commands
# ---------------------------------------------------------------------------


class TestIntegrationRateLimiting:
    def test_burst_commands_beyond_limit_are_dropped(self):
        kb = KeyboardAdapter(backend=MagicMock(spec=OSKeyboardBackend))
        adapter = _make_adapter(kb)
        dispatcher = CommandDispatcher(
            adapter=adapter,
            allowed_commands=frozenset(_COMMAND_MAP.values()),
            rate_limit_per_window=2,
            rate_limit_window_ms=5_000,
        )

        results = []
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
    def _make_armed_controller(
        self, edge_trigger_enabled: bool = True
    ) -> GestureActivationController:
        ctrl = GestureActivationController(
            command_map=_COMMAND_MAP,
            command_cooldown_ms=_COOLDOWN_MS,
            edge_trigger_enabled=edge_trigger_enabled,
        )
        assert ctrl.mode is RuntimeMode.ARMED
        return ctrl

    def test_single_swipe_dispatches_exactly_once(self):
        ctrl = self._make_armed_controller()
        t_base = _ACTIVATION_HOLD_MS + 50
        commands = []
        for i in range(8):
            t = t_base + i * 100
            with patch.object(type(ctrl), "_now_ms", staticmethod(lambda _t=t: _t)):
                cmd = ctrl.route(_pred("swiping_right"))
            if cmd is not None:
                commands.append(cmd)
        assert len(commands) == 1
        assert commands[0].command_id == "next_slide"

    def test_gesture_resets_after_unknown_then_redispatches(self):
        ctrl = self._make_armed_controller()
        t = _ACTIVATION_HOLD_MS + 50
        with patch.object(type(ctrl), "_now_ms", staticmethod(lambda: t)):
            cmd1 = ctrl.route(_pred("swiping_right"))
        assert cmd1 is not None

        t_unknown = t + _COOLDOWN_MS + 10
        with patch.object(type(ctrl), "_now_ms", staticmethod(lambda: t_unknown)):
            no_cmd = ctrl.route(_pred("unknown", accepted=False))
        assert no_cmd is None

        t2 = t_unknown + 50
        with patch.object(type(ctrl), "_now_ms", staticmethod(lambda: t2)):
            cmd2 = ctrl.route(_pred("swiping_right"))
        assert cmd2 is not None

    def test_same_swipe_redispatches_after_cooldown_expires(self):
        ctrl = self._make_armed_controller()
        t_base = _ACTIVATION_HOLD_MS + 50
        with patch.object(type(ctrl), "_now_ms", staticmethod(lambda: t_base)):
            cmd1 = ctrl.route(_pred("swiping_right"))
        assert cmd1 is not None

        t_after_cool = t_base + _COOLDOWN_MS + 10
        with patch.object(type(ctrl), "_now_ms", staticmethod(lambda: t_after_cool)):
            cmd2 = ctrl.route(_pred("swiping_right"))
        assert cmd2 is not None

    def test_different_gesture_after_cooldown_does_dispatch(self):
        ctrl = self._make_armed_controller()
        t_base = _ACTIVATION_HOLD_MS + 50
        with patch.object(type(ctrl), "_now_ms", staticmethod(lambda: t_base)):
            cmd1 = ctrl.route(_pred("swiping_right"))
        assert cmd1 is not None

        t_after = t_base + _COOLDOWN_MS + 10
        with patch.object(type(ctrl), "_now_ms", staticmethod(lambda: t_after)):
            cmd2 = ctrl.route(_pred("swiping_left"))
        assert cmd2 is not None
        assert cmd2.command_id == "previous_slide"

    def test_edge_trigger_disabled_falls_back_to_cooldown_only(self):
        ctrl = self._make_armed_controller(edge_trigger_enabled=False)
        t_base = _ACTIVATION_HOLD_MS + 50
        with patch.object(type(ctrl), "_now_ms", staticmethod(lambda: t_base)):
            cmd1 = ctrl.route(_pred("swiping_right"))
        assert cmd1 is not None

        t_after = t_base + _COOLDOWN_MS + 10
        with patch.object(type(ctrl), "_now_ms", staticmethod(lambda: t_after)):
            cmd2 = ctrl.route(_pred("swiping_right"))
        assert cmd2 is not None

    def test_full_pipeline_single_swipe_one_dispatch(self):
        ctrl = self._make_armed_controller()
        mock_backend = MagicMock(spec=OSKeyboardBackend)
        mock_backend.press_hotkey.return_value = True
        kb = KeyboardAdapter(backend=mock_backend)
        adapter = _make_adapter(kb)
        dispatcher = CommandDispatcher(
            adapter=adapter,
            allowed_commands=frozenset(_COMMAND_MAP.values()),
            rate_limit_per_window=10,
            rate_limit_window_ms=5000,
        )

        prediction_stream = [
            _pred("unknown", accepted=False),
            _pred("unknown", accepted=False),
            _pred("swiping_right"),
            _pred("swiping_right"),
            _pred("swiping_right"),
            _pred("swiping_right"),
            _pred("swiping_right"),
            _pred("unknown", accepted=False),
            _pred("unknown", accepted=False),
        ]

        dispatched = []
        t = _ACTIVATION_HOLD_MS + 50
        for pred in prediction_stream:
            with patch.object(type(ctrl), "_now_ms", staticmethod(lambda _t=t: _t)):
                cmd = ctrl.route(pred)
            if cmd is not None:
                result = dispatcher.dispatch(cmd)
                if result.success:
                    dispatched.append(cmd)
            t += 100

        assert len(dispatched) == 1
        assert dispatched[0].command_id == "next_slide"
        mock_backend.press_hotkey.assert_called_once_with("right")

    def test_temporal_voting_buildup_does_not_consume_rising_edge(self):
        ctrl = self._make_armed_controller()
        t_base = _ACTIVATION_HOLD_MS + 50

        with patch.object(type(ctrl), "_now_ms", staticmethod(lambda: t_base)):
            cmd1 = ctrl.route(_pred("swiping_right"))
        assert cmd1 is not None

        t_rest = t_base + _COOLDOWN_MS + 20
        with patch.object(type(ctrl), "_now_ms", staticmethod(lambda: t_rest)):
            ctrl.route(_pred("unknown", accepted=False))

        t1 = t_rest + 50
        with patch.object(type(ctrl), "_now_ms", staticmethod(lambda: t1)):
            no_cmd = ctrl.route(_pred("swiping_left", accepted=False))
        assert no_cmd is None

        t2 = t1 + 100
        with patch.object(type(ctrl), "_now_ms", staticmethod(lambda: t2)):
            no_cmd = ctrl.route(_pred("swiping_left", accepted=False))
        assert no_cmd is None

        t3 = t2 + 100
        with patch.object(type(ctrl), "_now_ms", staticmethod(lambda: t3)):
            cmd2 = ctrl.route(_pred("swiping_left", accepted=True))
        assert cmd2 is not None

    def test_sequential_swipe_right_then_left_then_right(self):
        ctrl = self._make_armed_controller()
        t = _ACTIVATION_HOLD_MS + 50

        def _route(label: str, accepted: bool = True) -> ExecutableCommand | None:
            nonlocal t
            with patch.object(type(ctrl), "_now_ms", staticmethod(lambda _t=t: _t)):
                cmd = ctrl.route(_pred(label, accepted))
            t += 150
            return cmd

        cmd1 = _route("swiping_right")
        assert cmd1 is not None

        for _ in range(8):
            _route("unknown", accepted=False)

        assert _route("swiping_left", accepted=False) is None
        assert _route("swiping_left", accepted=False) is None
        cmd2 = _route("swiping_left", accepted=True)
        assert cmd2 is not None

        for _ in range(8):
            _route("unknown", accepted=False)

        assert _route("swiping_right", accepted=False) is None
        assert _route("swiping_right", accepted=False) is None
        cmd3 = _route("swiping_right", accepted=True)
        assert cmd3 is not None
