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

    Edge-trigger dispatch
    ---------------------
    When ``edge_trigger_enabled=True`` (the default), a command gesture is only
    dispatched when the accepted label differs from ``_last_dispatched_label``
    — the label of the most recently executed command.  ``_last_dispatched_label``
    is reset to ``unknown`` whenever the prediction returns to the ``unknown``
    class (the hand goes neutral between gestures).

    **Why not track the previous frame's label?**  An earlier implementation
    advanced a ``_prev_label`` counter on *every* call to ``route()``, including
    frames where ``accepted=False``.  The temporal-voting window (``vote_window``
    = 3 by default) emits 2–3 unaccepted frames of the new gesture label before
    the first ``accepted=True`` frame.  By the time that eligible frame arrived,
    ``_prev_label`` already equalled the gesture label — so the edge check saw
    no change and suppressed every accepted prediction of every second gesture,
    making the system respond to only one gesture per activation.

    The ``_last_dispatched_label`` approach fixes this: unaccepted frames never
    advance the counter, so the first accepted prediction of a new gesture always
    sees a label change relative to the last dispatched command.

    The same-gesture-twice case is handled by the ``unknown`` reset: after the
    hand returns to neutral between two identical gestures, ``_last_dispatched_label``
    is cleared, allowing the second occurrence to fire.

    The cooldown remains as a secondary guard against rapid re-triggering in
    unusual conditions (e.g. alternating accepted/unaccepted despite short hold).
    """

    def __init__(
        self,
        *,
        activation_gesture: str,
        command_map: dict[str, str],
        activation_hold_ms: int,
        idle_timeout_ms: int,
        command_cooldown_ms: int,
        edge_trigger_enabled: bool = True,
        unknown_label: str = "unknown",
    ) -> None:
        self._activation_gesture = activation_gesture
        self._command_map = dict(command_map)
        self._activation_hold_ms = activation_hold_ms
        self._idle_timeout_ms = idle_timeout_ms
        self._command_cooldown_ms = command_cooldown_ms
        self._edge_trigger_enabled = edge_trigger_enabled
        self._unknown_label = unknown_label

        self._mode: RuntimeMode = RuntimeMode.IDLE
        self._activation_started_ms: int | None = None
        self._last_active_ms: int = self._now_ms()
        self._cooldown_started_ms: int | None = None
        # Label of the most recently dispatched command.  Reset to unknown_label
        # whenever the prediction returns to unknown (hand goes neutral).
        # See class docstring for the rationale over frame-by-frame label tracking.
        self._last_dispatched_label: str = unknown_label

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
                # Clear dispatch memory so that re-arming always starts fresh.
                self._last_dispatched_label = self._unknown_label
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

            if is_unknown:
                # Hand returned to neutral — clear the dispatch memory so that
                # performing the same gesture again (after a rest) fires normally.
                self._last_dispatched_label = self._unknown_label
                return None

            if is_activation:
                return None

            if not prediction.accepted:
                return None

            # Edge-trigger guard: dispatch only when the accepted label differs
            # from the last dispatched command label.  Unaccepted frames never
            # advance _last_dispatched_label, so the 2-3 unaccepted buildup frames
            # from temporal voting do not consume the rising edge.
            if (
                self._edge_trigger_enabled
                and prediction.label == self._last_dispatched_label
            ):
                LOGGER.debug(
                    "activation: edge_trigger suppressed  label=%r  (same as last dispatch)",
                    prediction.label,
                )
                return None

            command_id = self._command_map.get(prediction.label)
            if command_id is None:
                LOGGER.debug(
                    "activation: ARMED  gesture=%r not in command_map — ignored",
                    prediction.label,
                )
                return None

            self._last_dispatched_label = prediction.label
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
