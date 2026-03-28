from __future__ import annotations

import time

from gesture_control.domain.entities import ExecutableCommand, StabilizedPrediction
from gesture_control.domain.enums import RuntimeMode


class GestureActivationController:
    """Controls runtime state transitions and emits executable commands safely."""

    def __init__(
        self,
        *,
        activation_gesture: str,
        command_map: dict[str, str],
        activation_hold_ms: int,
        idle_timeout_ms: int,
        command_cooldown_ms: int,
        unknown_label: str = "unknown",
    ) -> None:
        self._activation_gesture = activation_gesture
        self._command_map = dict(command_map)
        self._activation_hold_ms = activation_hold_ms
        self._idle_timeout_ms = idle_timeout_ms
        self._command_cooldown_ms = command_cooldown_ms
        self._unknown_label = unknown_label

        self._mode: RuntimeMode = RuntimeMode.IDLE
        self._activation_started_ms: int | None = None
        self._last_active_ms: int = self._now_ms()
        self._cooldown_started_ms: int | None = None

    @property
    def mode(self) -> RuntimeMode:
        return self._mode

    def route(self, prediction: StabilizedPrediction) -> ExecutableCommand | None:
        now = self._now_ms()

        if self._mode is RuntimeMode.COOLDOWN:
            if self._cooldown_started_ms is not None and (
                now - self._cooldown_started_ms >= self._command_cooldown_ms
            ):
                self._mode = RuntimeMode.ARMED
                self._cooldown_started_ms = None
            else:
                return None

        if not prediction.accepted or prediction.label == self._unknown_label:
            if self._mode is RuntimeMode.ARMED and (
                now - self._last_active_ms > self._idle_timeout_ms
            ):
                self._mode = RuntimeMode.IDLE
                self._activation_started_ms = None
            return None

        self._last_active_ms = now

        if self._mode is RuntimeMode.IDLE:
            if prediction.label != self._activation_gesture:
                self._activation_started_ms = None
                return None

            if self._activation_started_ms is None:
                self._activation_started_ms = now
                return None

            if now - self._activation_started_ms >= self._activation_hold_ms:
                self._mode = RuntimeMode.ARMED
                self._activation_started_ms = None
            return None

        if self._mode is RuntimeMode.ARMED:
            if prediction.label == self._activation_gesture:
                return None

            command_id = self._command_map.get(prediction.label)
            if command_id is None:
                return None

            self._mode = RuntimeMode.COOLDOWN
            self._cooldown_started_ms = now
            return ExecutableCommand(
                command_id=command_id,
                source_gesture=prediction.label,
                issued_at_ms=now,
            )

        return None

    @staticmethod
    def _now_ms() -> int:
        return int(time.time() * 1000)
