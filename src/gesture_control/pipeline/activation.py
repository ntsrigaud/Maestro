from __future__ import annotations

import logging
import time

from gesture_control.domain.entities import ExecutableCommand, StabilizedPrediction
from gesture_control.domain.enums import RuntimeMode

LOGGER = logging.getLogger(__name__)


class GestureActivationController:
    """Controls runtime state transitions and emits executable commands safely.

    State machine
    -------------
    IDLE     → ARMED    : activation gesture held for *activation_hold_ms*.
    ARMED    → COOLDOWN : command gesture recognised → ExecutableCommand emitted.
    ARMED    → IDLE     : no recognised gesture for *idle_timeout_ms*.
    COOLDOWN → ARMED    : cooldown window expires.

    Activation hold design
    ----------------------
    The hold timer begins as soon as the predicted label equals the activation
    gesture and the confidence threshold has been passed, regardless of whether
    the stabiliser has declared the prediction ``accepted``.  Waiting for
    ``accepted=True`` before starting the timer compounds the temporal-
    consistency window (~300 ms at 30 fps) on top of the hold requirement,
    making activation noticeably sluggish.  A label that passed the confidence
    threshold but has not yet accumulated enough temporal votes is still a
    valid hold signal; the hold duration itself provides the required
    robustness against transient misfires.

    Armed-state persistence
    -----------------------
    ``_last_active_ms`` is refreshed whenever any *non-unknown* label is seen,
    not only when an accepted command gesture is recognised.  Combined with a
    generous default ``idle_timeout_ms`` (5 minutes), this prevents the system
    from dropping back to IDLE between the natural pauses of a presentation.
    """

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

        # ------------------------------------------------------------------
        # COOLDOWN — wait for the inter-command window to expire.
        # ------------------------------------------------------------------
        if self._mode is RuntimeMode.COOLDOWN:
            if self._cooldown_started_ms is not None and (
                now - self._cooldown_started_ms >= self._command_cooldown_ms
            ):
                self._mode = RuntimeMode.ARMED
                self._cooldown_started_ms = None
                LOGGER.debug(
                    "activation: COOLDOWN → ARMED  (cooldown_ms=%d)",
                    self._command_cooldown_ms,
                )
            else:
                return None

        is_unknown = prediction.label == self._unknown_label
        is_activation = prediction.label == self._activation_gesture and not is_unknown

        # Refresh activity timestamp on any recognisable (non-unknown) gesture.
        # This keeps the ARMED idle timer alive during natural presentation pauses
        # where the user's hand may be visible but not performing a command gesture.
        if not is_unknown:
            self._last_active_ms = now
            LOGGER.debug(
                "activation: gesture seen  label=%r  conf=%.2f  accepted=%s  mode=%s",
                prediction.label,
                prediction.confidence,
                prediction.accepted,
                self._mode.value,
            )

        # ------------------------------------------------------------------
        # IDLE — track activation hold.
        # The timer starts on the first frame where the activation gesture
        # is recognised above the confidence threshold, even before temporal
        # consistency (accepted=True) is established.
        # ------------------------------------------------------------------
        if self._mode is RuntimeMode.IDLE:
            if not is_activation:
                if self._activation_started_ms is not None:
                    LOGGER.debug(
                        "activation: hold broken  label=%r — timer reset",
                        prediction.label,
                    )
                    self._activation_started_ms = None
                return None

            if self._activation_started_ms is None:
                self._activation_started_ms = now
                LOGGER.debug(
                    "activation: hold timer started  gesture=%r  required_ms=%d",
                    prediction.label,
                    self._activation_hold_ms,
                )
                return None

            elapsed = now - self._activation_started_ms
            LOGGER.debug(
                "activation: hold progress  elapsed_ms=%d / required_ms=%d",
                elapsed,
                self._activation_hold_ms,
            )
            if elapsed >= self._activation_hold_ms:
                self._mode = RuntimeMode.ARMED
                self._activation_started_ms = None
                LOGGER.info(
                    "activation: IDLE → ARMED  gesture=%r  held_ms=%d",
                    prediction.label,
                    elapsed,
                )
            return None

        # ------------------------------------------------------------------
        # ARMED — check idle timeout, then dispatch command gestures.
        # ------------------------------------------------------------------
        if self._mode is RuntimeMode.ARMED:
            if now - self._last_active_ms > self._idle_timeout_ms:
                self._mode = RuntimeMode.IDLE
                self._activation_started_ms = None
                LOGGER.info(
                    "activation: ARMED → IDLE  (no gesture for %d ms)",
                    self._idle_timeout_ms,
                )
                return None

            if is_unknown or is_activation:
                return None

            if not prediction.accepted:
                return None

            command_id = self._command_map.get(prediction.label)
            if command_id is None:
                LOGGER.debug(
                    "activation: ARMED  gesture=%r not in command_map — ignored",
                    prediction.label,
                )
                return None

            self._mode = RuntimeMode.COOLDOWN
            self._cooldown_started_ms = now
            LOGGER.info(
                "activation: ARMED → COOLDOWN  gesture=%r  command_id=%r",
                prediction.label,
                command_id,
            )
            return ExecutableCommand(
                command_id=command_id,
                source_gesture=prediction.label,
                issued_at_ms=now,
            )

        return None

    @staticmethod
    def _now_ms() -> int:
        return int(time.time() * 1000)
