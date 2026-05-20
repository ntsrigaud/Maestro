from __future__ import annotations

from unittest.mock import patch

from maestro.domain.entities import StabilizedPrediction
from maestro.domain.enums import RuntimeMode
from maestro.pipeline.activation import GestureActivationController


def _pred(label: str, accepted: bool = True, confidence: float = 0.95) -> StabilizedPrediction:
    return StabilizedPrediction(label=label, confidence=confidence, accepted=accepted)


def _make_controller(
    command_map: dict[str, str] | None = None,
    command_cooldown_ms: int = 50,
    edge_trigger_enabled: bool = True,
    continuous_labels: frozenset[str] = frozenset(),
) -> GestureActivationController:
    return GestureActivationController(
        command_map=command_map or {"swiping_right": "next_slide"},
        command_cooldown_ms=command_cooldown_ms,
        edge_trigger_enabled=edge_trigger_enabled,
        continuous_labels=continuous_labels,
    )


# ---------------------------------------------------------------------------
# Initial state
# ---------------------------------------------------------------------------


def test_controller_starts_in_armed_state() -> None:
    ctrl = _make_controller()
    assert ctrl.mode is RuntimeMode.ARMED


# ---------------------------------------------------------------------------
# ARMED → COOLDOWN: command dispatch
# ---------------------------------------------------------------------------


def test_armed_dispatches_known_gesture() -> None:
    ctrl = _make_controller()
    with patch.object(type(ctrl), "_now_ms", staticmethod(lambda: 1000)):
        cmd = ctrl.route(_pred("swiping_right"))
    assert cmd is not None
    assert cmd.command_id == "next_slide"
    assert cmd.source_gesture == "swiping_right"
    assert ctrl.mode is RuntimeMode.COOLDOWN


def test_armed_does_not_dispatch_unknown_gesture() -> None:
    ctrl = _make_controller()
    with patch.object(type(ctrl), "_now_ms", staticmethod(lambda: 1000)):
        cmd = ctrl.route(_pred("unknown_xyz"))
    assert cmd is None
    assert ctrl.mode is RuntimeMode.ARMED


def test_armed_does_not_dispatch_unaccepted_prediction() -> None:
    ctrl = _make_controller()
    with patch.object(type(ctrl), "_now_ms", staticmethod(lambda: 1000)):
        cmd = ctrl.route(_pred("swiping_right", accepted=False))
    assert cmd is None
    assert ctrl.mode is RuntimeMode.ARMED


# ---------------------------------------------------------------------------
# COOLDOWN
# ---------------------------------------------------------------------------


def test_cooldown_suppresses_commands_during_window() -> None:
    ctrl = _make_controller(command_cooldown_ms=500)
    with patch.object(type(ctrl), "_now_ms", staticmethod(lambda: 0)):
        ctrl.route(_pred("swiping_right"))
    assert ctrl.mode is RuntimeMode.COOLDOWN

    with patch.object(type(ctrl), "_now_ms", staticmethod(lambda: 400)):
        cmd = ctrl.route(_pred("swiping_right"))
    assert cmd is None
    assert ctrl.mode is RuntimeMode.COOLDOWN


def test_cooldown_expires_back_to_armed() -> None:
    ctrl = _make_controller(command_cooldown_ms=500)
    with patch.object(type(ctrl), "_now_ms", staticmethod(lambda: 0)):
        ctrl.route(_pred("swiping_right"))

    with patch.object(type(ctrl), "_now_ms", staticmethod(lambda: 510)):
        cmd = ctrl.route(_pred("swiping_right", accepted=False))
    # Cooldown expired → ARMED. _last_dispatched_label reset to unknown.
    # cmd is None because the prediction is not accepted, not due to edge-trigger.
    assert ctrl.mode is RuntimeMode.ARMED
    assert cmd is None  # not accepted


# ---------------------------------------------------------------------------
# Continuous label bypass (point_one / point_two → LaserPointerTracker only)
# ---------------------------------------------------------------------------


def test_continuous_label_returns_none_when_armed() -> None:
    """A continuous gesture in ARMED state must bypass the state machine entirely."""
    ctrl = _make_controller(
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
    ctrl = _make_controller(
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
    ctrl = _make_controller(
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
    ctrl = _make_controller(
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


def test_held_discrete_gesture_dispatches_once_per_cooldown_period() -> None:
    """A held gesture fires exactly once per cooldown period.

    After cooldown expires, _last_dispatched_label is reset, so the same gesture
    held continuously re-dispatches as soon as ARMED is re-entered.  For a 2-second
    window with a 1500 ms cooldown, exactly 2 dispatches are expected: one at the
    start and one when the cooldown expires at ~t=1500.
    """
    ctrl = _make_controller(
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

    assert dispatch_count == 2, (
        f"Expected 2 dispatches (one per cooldown period) for a held gesture, got {dispatch_count}"
    )


def test_held_gesture_redispatches_after_cooldown_without_neutral() -> None:
    """Same gesture re-dispatches immediately after cooldown, no neutral pose needed.

    _last_dispatched_label is reset at cooldown expiry (COOLDOWN → ARMED transition),
    so the same accepted gesture fires again as soon as the controller re-enters ARMED.
    """
    ctrl = _make_controller(
        command_map={"swiping_right": "next_slide"},
        command_cooldown_ms=100,
        edge_trigger_enabled=True,
    )
    # First occurrence
    with patch.object(type(ctrl), "_now_ms", staticmethod(lambda: 0)):
        cmd1 = ctrl.route(_pred("swiping_right"))
    assert cmd1 is not None
    assert ctrl.mode is RuntimeMode.COOLDOWN

    # Cooldown expires — dispatch memory reset automatically, no neutral needed
    with patch.object(type(ctrl), "_now_ms", staticmethod(lambda: 200)):
        cmd2 = ctrl.route(_pred("swiping_right"))
    assert cmd2 is not None, "Same gesture must re-dispatch immediately after cooldown"
    assert cmd2.command_id == "next_slide"


# ---------------------------------------------------------------------------
# Continuous labels (laser pointer, etc.)
# ---------------------------------------------------------------------------


def test_same_gesture_dispatches_again_after_cooldown_without_neutral() -> None:
    """Regression guard: same gesture must NOT require a neutral hand pose to re-arm.

    The only requirement for same-gesture re-dispatch is that the cooldown window
    has elapsed.  _last_dispatched_label is reset at cooldown expiry, not at
    the next unknown prediction.
    """
    ctrl = _make_controller(
        command_map={"swiping_right": "next_slide"},
        command_cooldown_ms=500,
        edge_trigger_enabled=True,
    )
    # Dispatch
    with patch.object(type(ctrl), "_now_ms", staticmethod(lambda: 0)):
        cmd1 = ctrl.route(_pred("swiping_right"))
    assert cmd1 is not None

    # Before cooldown — blocked
    with patch.object(type(ctrl), "_now_ms", staticmethod(lambda: 400)):
        cmd_blocked = ctrl.route(_pred("swiping_right"))
    assert cmd_blocked is None

    # After cooldown — dispatches again without any neutral frame
    with patch.object(type(ctrl), "_now_ms", staticmethod(lambda: 600)):
        cmd2 = ctrl.route(_pred("swiping_right"))
    assert cmd2 is not None, "Same gesture must re-dispatch after cooldown without neutral pose"
    assert cmd2.command_id == "next_slide"


def test_continuous_label_returns_none_in_armed() -> None:
    """A continuous gesture in ARMED state is excluded from routing."""
    ctrl = _make_controller(continuous_labels=frozenset(["point_one"]))
    assert ctrl.mode is RuntimeMode.ARMED
    with patch.object(type(ctrl), "_now_ms", staticmethod(lambda: 0)):
        cmd = ctrl.route(_pred("point_one"))
    assert cmd is None  # continuous gesture routed directly to laser pointer
    assert ctrl.mode is RuntimeMode.ARMED  # state must not change
