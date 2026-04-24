from __future__ import annotations

from unittest.mock import patch

from maestro.domain.entities import StabilizedPrediction
from maestro.domain.enums import RuntimeMode
from maestro.pipeline.activation import GestureActivationController


def _pred(
    label: str, accepted: bool = True, confidence: float = 0.95
) -> StabilizedPrediction:
    return StabilizedPrediction(label=label, confidence=confidence, accepted=accepted)


def _make_controller(
    command_map: dict[str, str] | None = None,
    idle_timeout_ms: int = 60_000,
    command_cooldown_ms: int = 50,
    edge_trigger_enabled: bool = True,
) -> GestureActivationController:
    return GestureActivationController(
        command_map=command_map or {"swipe_right": "next_slide"},
        idle_timeout_ms=idle_timeout_ms,
        command_cooldown_ms=command_cooldown_ms,
        edge_trigger_enabled=edge_trigger_enabled,
    )


def _make_armed_controller(
    command_map: dict[str, str] | None = None,
    idle_timeout_ms: int = 60_000,
    command_cooldown_ms: int = 50,
    edge_trigger_enabled: bool = True,
) -> GestureActivationController:
    ctrl = _make_controller(
        command_map=command_map,
        idle_timeout_ms=idle_timeout_ms,
        command_cooldown_ms=command_cooldown_ms,
        edge_trigger_enabled=edge_trigger_enabled,
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
        cmd = ctrl.route(_pred("swipe_right"))
    assert cmd is not None
    assert cmd.command_id == "next_slide"
    assert cmd.source_gesture == "swipe_right"
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
        cmd = ctrl.route(_pred("swipe_right", accepted=False))
    assert cmd is None
    assert ctrl.mode is RuntimeMode.ARMED


# ---------------------------------------------------------------------------
# COOLDOWN
# ---------------------------------------------------------------------------


def test_cooldown_suppresses_commands_during_window() -> None:
    ctrl = _make_armed_controller(command_cooldown_ms=500)
    with patch.object(type(ctrl), "_now_ms", staticmethod(lambda: 0)):
        ctrl.route(_pred("swipe_right"))
    assert ctrl.mode is RuntimeMode.COOLDOWN

    with patch.object(type(ctrl), "_now_ms", staticmethod(lambda: 400)):
        cmd = ctrl.route(_pred("swipe_right"))
    assert cmd is None
    assert ctrl.mode is RuntimeMode.COOLDOWN


def test_cooldown_expires_back_to_armed() -> None:
    ctrl = _make_armed_controller(command_cooldown_ms=500)
    with patch.object(type(ctrl), "_now_ms", staticmethod(lambda: 0)):
        ctrl.route(_pred("swipe_right"))

    with patch.object(type(ctrl), "_now_ms", staticmethod(lambda: 510)):
        cmd = ctrl.route(_pred("swipe_right", accepted=False))
    # Just expired, should be ARMED now, but swipe_right is the same label.
    # Unknown resets edge trigger — verify ARMED at minimum.
    assert ctrl.mode is RuntimeMode.ARMED
    assert (
        cmd is None
    )  # edge-trigger: same label as last dispatch, unknown not yet seen


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
        ctrl.route(_pred("swipe_right"))
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
            "swipe_right": "next_slide",
            "swipe_left": "previous_slide",
            "swipe_up": "start_presentation",
            "swipe_down": "exit_presentation",
            "zoom_in": "toggle_pointer_mode",
            "zoom_out": "raise_hand",
        },
        idle_timeout_ms=60_000,
        command_cooldown_ms=50,
    )
    ctrl.force_idle()

    t = 100
    for gesture in (
        "swipe_right",
        "swipe_left",
        "swipe_up",
        "swipe_down",
        "zoom_in",
        "zoom_out",
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
        cmd = ctrl.route(_pred("swipe_right"))
    assert cmd is not None
    assert cmd.command_id == "next_slide"


def test_force_idle_clears_cooldown_timer_so_rearm_starts_fresh() -> None:
    """After dispatch + force_idle() + rearm(), commands dispatch immediately
    (no stale cooldown timer blocking the pipeline)."""
    ctrl = _make_armed_controller(command_cooldown_ms=500)
    with patch.object(type(ctrl), "_now_ms", staticmethod(lambda: 0)):
        cmd1 = ctrl.route(_pred("swipe_right"))
    assert cmd1 is not None
    assert ctrl.mode is RuntimeMode.COOLDOWN

    ctrl.force_idle()
    ctrl.rearm()
    assert ctrl.mode is RuntimeMode.ARMED

    # After rearm, commands must dispatch immediately (no stale cooldown).
    with patch.object(type(ctrl), "_now_ms", staticmethod(lambda: 100)):
        cmd2 = ctrl.route(_pred("swipe_right"))
    assert (
        cmd2 is not None
    ), "Expected dispatch after force_idle+rearm; old cooldown must not block"
    assert cmd2.command_id == "next_slide"
