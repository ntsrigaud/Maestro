from __future__ import annotations

import time

from gesture_control.domain.entities import StabilizedPrediction
from gesture_control.domain.enums import RuntimeMode
from gesture_control.pipeline.activation import GestureActivationController


def _accepted(label: str, confidence: float = 0.95) -> StabilizedPrediction:
    return StabilizedPrediction(label=label, confidence=confidence, accepted=True)


def test_activation_controller_arms_after_hold() -> None:
    controller = GestureActivationController(
        activation_gesture="open_palm_hold",
        command_map={"swipe_right": "next_slide"},
        activation_hold_ms=10,
        idle_timeout_ms=1000,
        command_cooldown_ms=50,
    )

    assert controller.mode is RuntimeMode.IDLE
    assert controller.route(_accepted("open_palm_hold")) is None
    time.sleep(0.02)
    assert controller.route(_accepted("open_palm_hold")) is None
    assert controller.mode is RuntimeMode.ARMED


def test_activation_controller_emits_command_in_armed_mode() -> None:
    controller = GestureActivationController(
        activation_gesture="open_palm_hold",
        command_map={"swipe_right": "next_slide"},
        activation_hold_ms=1,
        idle_timeout_ms=1000,
        command_cooldown_ms=10,
    )

    controller.route(_accepted("open_palm_hold"))
    time.sleep(0.01)
    controller.route(_accepted("open_palm_hold"))
    assert controller.mode is RuntimeMode.ARMED

    command = controller.route(_accepted("swipe_right"))
    assert command is not None
    assert command.command_id == "next_slide"
    assert controller.mode is RuntimeMode.COOLDOWN


def _make_armed_controller() -> GestureActivationController:
    """Return a controller already in ARMED state."""
    controller = GestureActivationController(
        activation_gesture="open_palm_hold",
        command_map={"swipe_right": "next_slide"},
        activation_hold_ms=1,
        idle_timeout_ms=60_000,
        command_cooldown_ms=50,
    )
    controller.route(_accepted("open_palm_hold"))
    time.sleep(0.005)
    controller.route(_accepted("open_palm_hold"))
    assert controller.mode is RuntimeMode.ARMED
    return controller


def test_force_idle_from_armed_resets_to_idle() -> None:
    controller = _make_armed_controller()
    controller.force_idle()
    assert controller.mode is RuntimeMode.IDLE


def test_force_idle_from_cooldown_resets_to_idle() -> None:
    controller = _make_armed_controller()
    controller.route(_accepted("swipe_right"))  # ARMED → COOLDOWN
    assert controller.mode is RuntimeMode.COOLDOWN
    controller.force_idle()
    assert controller.mode is RuntimeMode.IDLE


def test_force_idle_clears_timers_so_reactivation_works() -> None:
    """After force_idle(), a fresh hold cycle must arm the controller again."""
    controller = _make_armed_controller()
    controller.force_idle()
    assert controller.mode is RuntimeMode.IDLE

    # A new hold cycle must succeed — if timers were not cleared, the stale
    # _activation_started_ms could make elapsed appear negative or erratic.
    controller.route(_accepted("open_palm_hold"))
    time.sleep(0.005)
    controller.route(_accepted("open_palm_hold"))
    assert controller.mode is RuntimeMode.ARMED


def test_force_idle_from_idle_is_idempotent() -> None:
    controller = GestureActivationController(
        activation_gesture="open_palm_hold",
        command_map={},
        activation_hold_ms=500,
        idle_timeout_ms=60_000,
        command_cooldown_ms=50,
    )
    assert controller.mode is RuntimeMode.IDLE
    controller.force_idle()  # must not raise
    assert controller.mode is RuntimeMode.IDLE
