from __future__ import annotations

from unittest.mock import patch

from maestro.domain.entities import StabilizedPrediction
from maestro.domain.enums import RuntimeMode
from maestro.pipeline.activation import GestureActivationController


def _pred(label: str, accepted: bool = True, confidence: float = 0.95) -> StabilizedPrediction:
    return StabilizedPrediction(label=label, confidence=confidence, accepted=accepted)


def _make_controller(
    command_map: dict[str, str] | None = None,
    idle_timeout_ms: int = 60_000,
    command_cooldown_ms: int = 50,
    edge_trigger_enabled: bool = True,
    continuous_labels: frozenset[str] = frozenset(),
    auto_idle_enabled: bool = True,
) -> GestureActivationController:
    return GestureActivationController(
        command_map=command_map or {"swiping_right": "next_slide"},
        idle_timeout_ms=idle_timeout_ms,
        command_cooldown_ms=command_cooldown_ms,
        edge_trigger_enabled=edge_trigger_enabled,
        continuous_labels=continuous_labels,
        auto_idle_enabled=auto_idle_enabled,
    )


def _make_armed_controller(
    command_map: dict[str, str] | None = None,
    idle_timeout_ms: int = 60_000,
    command_cooldown_ms: int = 50,
    edge_trigger_enabled: bool = True,
    continuous_labels: frozenset[str] = frozenset(),
    auto_idle_enabled: bool = True,
) -> GestureActivationController:
    ctrl = _make_controller(
        command_map=command_map,
        idle_timeout_ms=idle_timeout_ms,
        command_cooldown_ms=command_cooldown_ms,
        edge_trigger_enabled=edge_trigger_enabled,
        continuous_labels=continuous_labels,
        auto_idle_enabled=auto_idle_enabled,
    )
    ctrl.rearm()
    return ctrl


# ---------------------------------------------------------------------------
# Initial state
# ---------------------------------------------------------------------------


def test_controller_starts_in_idle_state() -> None:
    ctrl = _make_controller()
    assert ctrl.mode is RuntimeMode.IDLE


# ---------------------------------------------------------------------------
# ARMED → COOLDOWN: command dispatch
# ---------------------------------------------------------------------------


def test_armed_dispatches_known_gesture() -> None:
    ctrl = _make_armed_controller()
    with patch.object(type(ctrl), "_now_ms", staticmethod(lambda: 1000)):
        cmd = ctrl.route(_pred("swiping_right"))
    assert cmd is not None
    assert cmd.command_id == "next_slide"
    assert cmd.source_gesture == "swiping_right"
    assert ctrl.mode is RuntimeMode.COOLDOWN


def test_armed_does_not_dispatch_unknown_gesture() -> None:
    ctrl = _make_armed_controller()
    with patch.object(type(ctrl), "_now_ms", staticmethod(lambda: 1000)):
        cmd = ctrl.route(_pred("unknown_xyz"))
    assert cmd is None
    assert ctrl.mode is RuntimeMode.ARMED


def test_armed_does_not_dispatch_unaccepted_prediction() -> None:
    ctrl = _make_armed_controller()
    with patch.object(type(ctrl), "_now_ms", staticmethod(lambda: 1000)):
        cmd = ctrl.route(_pred("swiping_right", accepted=False))
    assert cmd is None
    assert ctrl.mode is RuntimeMode.ARMED


# ---------------------------------------------------------------------------
# COOLDOWN
# ---------------------------------------------------------------------------


def test_cooldown_suppresses_commands_during_window() -> None:
    ctrl = _make_armed_controller(command_cooldown_ms=500)
    with patch.object(type(ctrl), "_now_ms", staticmethod(lambda: 0)):
        ctrl.route(_pred("swiping_right"))
    assert ctrl.mode is RuntimeMode.COOLDOWN

    with patch.object(type(ctrl), "_now_ms", staticmethod(lambda: 400)):
        cmd = ctrl.route(_pred("swiping_right"))
    assert cmd is None
    assert ctrl.mode is RuntimeMode.COOLDOWN


def test_cooldown_expires_back_to_armed() -> None:
    ctrl = _make_armed_controller(command_cooldown_ms=500)
    with patch.object(type(ctrl), "_now_ms", staticmethod(lambda: 0)):
        ctrl.route(_pred("swiping_right"))

    with patch.object(type(ctrl), "_now_ms", staticmethod(lambda: 510)):
        cmd = ctrl.route(_pred("swiping_right", accepted=False))
    # Just expired, should be ARMED now, but swiping_right is the same label.
    # unknown resets edge trigger — verify ARMED at minimum.
    assert ctrl.mode is RuntimeMode.ARMED
    assert cmd is None  # edge-trigger: same label as last dispatch, unknown not yet seen


# ---------------------------------------------------------------------------
# ARMED idle timeout → IDLE
# ---------------------------------------------------------------------------


def test_idle_timeout_transitions_to_idle() -> None:
    # Patch _now_ms during construction AND rearm() so _last_active_ms starts at 0.
    with patch.object(GestureActivationController, "_now_ms", staticmethod(lambda: 0)):
        ctrl = _make_controller(idle_timeout_ms=1000)
        ctrl.rearm()  # IDLE → ARMED; _last_active_ms = 0 while still patched
    with patch.object(type(ctrl), "_now_ms", staticmethod(lambda: 2000)):
        cmd = ctrl.route(_pred("unknown", accepted=False))
    assert cmd is None
    assert ctrl.mode is RuntimeMode.IDLE


def test_auto_idle_disabled_never_transitions_to_idle_on_timeout() -> None:
    with patch.object(GestureActivationController, "_now_ms", staticmethod(lambda: 0)):
        ctrl = _make_controller(idle_timeout_ms=1000, auto_idle_enabled=False)
        ctrl.rearm()
    with patch.object(type(ctrl), "_now_ms", staticmethod(lambda: 2000)):
        cmd = ctrl.route(_pred("unknown", accepted=False))
    assert cmd is None
    assert ctrl.mode is RuntimeMode.ARMED


# ---------------------------------------------------------------------------
# force_idle
# ---------------------------------------------------------------------------


def test_force_idle_from_armed_resets_to_idle() -> None:
    ctrl = _make_controller()
    ctrl.rearm()  # IDLE → ARMED
    assert ctrl.mode is RuntimeMode.ARMED
    ctrl.force_idle()
    assert ctrl.mode is RuntimeMode.IDLE


def test_force_idle_from_cooldown_resets_to_idle() -> None:
    ctrl = _make_armed_controller()
    with patch.object(type(ctrl), "_now_ms", staticmethod(lambda: 0)):
        ctrl.route(_pred("swiping_right"))
    assert ctrl.mode is RuntimeMode.COOLDOWN
    ctrl.force_idle()
    assert ctrl.mode is RuntimeMode.IDLE


def test_force_idle_from_idle_is_idempotent() -> None:
    ctrl = _make_controller()
    ctrl.force_idle()
    ctrl.force_idle()  # must not raise
    assert ctrl.mode is RuntimeMode.IDLE


def test_force_idle_prevents_dispatch_for_all_gestures() -> None:
    """After force_idle(), route() returns None for all gesture labels."""
    ctrl = GestureActivationController(
        command_map={
            "swiping_right": "next_slide",
            "swiping_left": "previous_slide",
            "swiping_up": "start_presentation",
            "swiping_down": "exit_presentation",
            "zooming_in_full_hand": "toggle_pointer_mode",
            "zooming_out_full_hand": "raise_hand",
        },
        idle_timeout_ms=60_000,
        command_cooldown_ms=50,
    )
    ctrl.force_idle()

    t = 100
    for gesture in (
        "swiping_right",
        "swiping_left",
        "swiping_up",
        "swiping_down",
        "zooming_in_full_hand",
        "zooming_out_full_hand",
    ):
        with patch.object(type(ctrl), "_now_ms", staticmethod(lambda _t=t: _t)):
            cmd = ctrl.route(_pred(gesture))
        assert cmd is None, f"Expected no command for {gesture} when IDLE"
        t += 50


# ---------------------------------------------------------------------------
# rearm
# ---------------------------------------------------------------------------


def test_rearm_transitions_idle_to_armed() -> None:
    ctrl = _make_controller()
    ctrl.force_idle()
    assert ctrl.mode is RuntimeMode.IDLE
    ctrl.rearm()
    assert ctrl.mode is RuntimeMode.ARMED


def test_rearm_from_armed_is_noop() -> None:
    ctrl = _make_controller()
    ctrl.rearm()  # IDLE → ARMED
    assert ctrl.mode is RuntimeMode.ARMED
    ctrl.rearm()  # noop: already ARMED
    assert ctrl.mode is RuntimeMode.ARMED


def test_rearm_after_force_idle_restores_dispatch() -> None:
    """After force_idle() + rearm(), commands dispatch normally."""
    ctrl = _make_controller()
    ctrl.force_idle()
    ctrl.rearm()
    assert ctrl.mode is RuntimeMode.ARMED

    with patch.object(type(ctrl), "_now_ms", staticmethod(lambda: 100)):
        cmd = ctrl.route(_pred("swiping_right"))
    assert cmd is not None
    assert cmd.command_id == "next_slide"


def test_force_idle_clears_cooldown_timer_so_rearm_starts_fresh() -> None:
    """After dispatch + force_idle() + rearm(), commands dispatch immediately
    (no stale cooldown timer blocking the pipeline)."""
    ctrl = _make_armed_controller(command_cooldown_ms=500)
    with patch.object(type(ctrl), "_now_ms", staticmethod(lambda: 0)):
        cmd1 = ctrl.route(_pred("swiping_right"))
    assert cmd1 is not None
    assert ctrl.mode is RuntimeMode.COOLDOWN

    ctrl.force_idle()
    ctrl.rearm()
    assert ctrl.mode is RuntimeMode.ARMED

    # After rearm, commands must dispatch immediately (no stale cooldown).
    with patch.object(type(ctrl), "_now_ms", staticmethod(lambda: 100)):
        cmd2 = ctrl.route(_pred("swiping_right"))
    assert cmd2 is not None, "Expected dispatch after force_idle+rearm; old cooldown must not block"
    assert cmd2.command_id == "next_slide"


# ---------------------------------------------------------------------------
# Continuous label bypass (point_one / point_two → LaserPointerTracker only)
# ---------------------------------------------------------------------------


def test_continuous_label_returns_none_when_idle() -> None:
    """A continuous gesture in IDLE state must never produce a command."""
    ctrl = _make_controller(
        command_map={"swiping_right": "next_slide"},
        continuous_labels=frozenset({"point_one", "point_two"}),
    )
    assert ctrl.mode is RuntimeMode.IDLE
    cmd = ctrl.route(_pred("point_one"))
    assert cmd is None
    assert ctrl.mode is RuntimeMode.IDLE  # state must not change


def test_continuous_label_returns_none_when_armed() -> None:
    """A continuous gesture in ARMED state must bypass the state machine entirely."""
    ctrl = _make_armed_controller(
        command_map={"swiping_right": "next_slide"},
        continuous_labels=frozenset({"point_one", "point_two"}),
    )
    assert ctrl.mode is RuntimeMode.ARMED
    with patch.object(type(ctrl), "_now_ms", staticmethod(lambda: 1000)):
        cmd = ctrl.route(_pred("point_one"))
    assert cmd is None
    # ARMED → COOLDOWN transition must NOT have fired
    assert ctrl.mode is RuntimeMode.ARMED


def test_continuous_label_never_triggers_cooldown() -> None:
    """Repeated accepted continuous predictions must not consume the cooldown slot."""
    ctrl = _make_armed_controller(
        command_map={"swiping_right": "next_slide"},
        continuous_labels=frozenset({"point_one"}),
    )
    for t in range(0, 600, 100):
        with patch.object(type(ctrl), "_now_ms", staticmethod(lambda _t=t: _t)):
            cmd = ctrl.route(_pred("point_one"))
        assert cmd is None
    assert ctrl.mode is RuntimeMode.ARMED


def test_discrete_gesture_still_dispatches_when_continuous_labels_set() -> None:
    """Continuous label bypass must not affect discrete gesture dispatch."""
    ctrl = _make_armed_controller(
        command_map={"swiping_right": "next_slide"},
        continuous_labels=frozenset({"point_one", "point_two"}),
    )
    with patch.object(type(ctrl), "_now_ms", staticmethod(lambda: 1000)):
        cmd = ctrl.route(_pred("swiping_right"))
    assert cmd is not None
    assert cmd.command_id == "next_slide"
    assert ctrl.mode is RuntimeMode.COOLDOWN


def test_continuous_label_not_blocked_by_cooldown() -> None:
    """Even in COOLDOWN state, continuous gesture route() returns None immediately
    (not blocked — it exits before the COOLDOWN check matters)."""
    ctrl = _make_armed_controller(
        command_map={"swiping_right": "next_slide"},
        continuous_labels=frozenset({"point_one"}),
        command_cooldown_ms=5000,
    )
    # Dispatch a discrete gesture → enter COOLDOWN
    with patch.object(type(ctrl), "_now_ms", staticmethod(lambda: 0)):
        ctrl.route(_pred("swiping_right"))
    assert ctrl.mode is RuntimeMode.COOLDOWN

    # Continuous gesture during cooldown → None, mode stays COOLDOWN
    with patch.object(type(ctrl), "_now_ms", staticmethod(lambda: 100)):
        cmd = ctrl.route(_pred("point_one"))
    assert cmd is None
    assert ctrl.mode is RuntimeMode.COOLDOWN


# ---------------------------------------------------------------------------
# Duplicate dispatch regression: held gestures with edge_trigger_enabled=True
# ---------------------------------------------------------------------------


def test_held_discrete_gesture_dispatches_once_then_edge_blocked() -> None:
    """A gesture held across many accepted windows must produce exactly one command.

    This is the primary regression guard for the 'repeated swipe' bug:
    with edge_trigger_enabled=True (new production default), the second and
    subsequent accepted windows of the same label are suppressed by the
    _last_dispatched_label check.  The cooldown provides secondary protection.
    """
    ctrl = _make_armed_controller(
        command_map={"stop_sign": "zoom_out_totally"},
        command_cooldown_ms=1500,
        edge_trigger_enabled=True,
    )
    dispatch_count = 0
    # Simulate 20 accepted windows of the same gesture (≈2 seconds at 100ms/window)
    for t in range(0, 2000, 100):
        with patch.object(type(ctrl), "_now_ms", staticmethod(lambda _t=t: _t)):
            cmd = ctrl.route(_pred("stop_sign"))
        if cmd is not None:
            dispatch_count += 1

    assert dispatch_count == 1, (
        f"Expected exactly 1 dispatch for a held gesture, got {dispatch_count}"
    )


def test_held_gesture_redispatches_after_neutral_reset() -> None:
    """After a held gesture → unknown → same gesture again, exactly one more
    dispatch must occur (the unknown resets _last_dispatched_label)."""
    ctrl = _make_armed_controller(
        command_map={"swiping_right": "next_slide"},
        command_cooldown_ms=100,
        edge_trigger_enabled=True,
    )
    # First occurrence
    with patch.object(type(ctrl), "_now_ms", staticmethod(lambda: 0)):
        cmd1 = ctrl.route(_pred("swiping_right"))
    assert cmd1 is not None

    # Cooldown expires + hand goes neutral
    with patch.object(type(ctrl), "_now_ms", staticmethod(lambda: 200)):
        ctrl.route(_pred("unknown", accepted=False))  # resets _last_dispatched_label

    # Second occurrence of same gesture — must dispatch again
    with patch.object(type(ctrl), "_now_ms", staticmethod(lambda: 300)):
        cmd2 = ctrl.route(_pred("swiping_right"))
    assert cmd2 is not None, "Second swipe after neutral reset must dispatch"
    assert cmd2.command_id == "next_slide"
