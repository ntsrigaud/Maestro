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
