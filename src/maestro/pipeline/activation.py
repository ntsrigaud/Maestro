from __future__ import annotations

import logging
import time

from maestro.domain.entities import ExecutableCommand, StabilizedPrediction
from maestro.domain.enums import RuntimeMode

LOGGER = logging.getLogger(__name__)


class GestureActivationController:
    """Controls runtime state transitions and emits executable commands safely.

    State machine
    -------------
    ARMED    -> COOLDOWN : command gesture recognised -> ExecutableCommand emitted.
    COOLDOWN -> ARMED    : cooldown window expires.

    Activation gate
    ---------------
    The controller starts in ARMED state and remains armed throughout the
    session lifetime.

    Armed-state persistence
    -----------------------
    ``_last_active_ms`` is refreshed whenever any *non-unknown* label is seen.
    This ensures the system remains continuously responsive throughout the session.

    Edge-trigger dispatch
    ---------------------
    When ``edge_trigger_enabled=True`` (the default), a command gesture is only
    dispatched when the accepted label differs from ``_last_dispatched_label``
    --- the label of the most recently executed command.
    ``_last_dispatched_label`` is reset to ``unknown`` on two occasions:

    1. **Cooldown expiry** (primary reset): when the COOLDOWN window expires and
       the controller transitions back to ARMED, ``_last_dispatched_label`` is
       cleared immediately.  This means the same gesture can be re-dispatched
       as soon as the cooldown ends, with no intermediate neutral hand pose
       required.  The cooldown is the sole anti-bounce protection.

    2. **Neutral prediction** (secondary reset): if the prediction returns to
       the ``unknown`` class while ARMED, ``_last_dispatched_label`` is also
       cleared.  This is now redundant with the cooldown reset in normal usage
       but is kept as an explicit signal that the hand has returned to neutral.

    **Why not track the previous frame's label?**  An earlier implementation
    advanced a ``_prev_label`` counter on *every* call to ``route()``, including
    frames where ``accepted=False``.  The temporal-voting window (``vote_window``
    = 3 by default) emits 2-3 unaccepted frames of the new gesture label before
    the first ``accepted=True`` frame.  By the time that eligible frame arrived,
    ``_prev_label`` already equalled the gesture label --- so the edge check saw
    no change and suppressed every accepted prediction of every second gesture,
    making the system respond to only one gesture per activation.

    The ``_last_dispatched_label`` approach fixes this: unaccepted frames never
    advance the counter, so the first accepted prediction of a new gesture always
    sees a label change relative to the last dispatched command.
    """

    def __init__(
        self,
        *,
        command_map: dict[str, str],
        command_cooldown_ms: int,
        edge_trigger_enabled: bool = True,
        continuous_labels: frozenset[str] = frozenset(),
        unknown_label: str = "unknown",
    ) -> None:
        self._command_map = dict(command_map)
        self._command_cooldown_ms = command_cooldown_ms
        self._edge_trigger_enabled = edge_trigger_enabled
        self._continuous_labels = continuous_labels
        self._unknown_label = unknown_label

        self._mode: RuntimeMode = RuntimeMode.ARMED
        self._cooldown_started_ms: int | None = None
        # Label of the most recently dispatched command.  Reset to unknown_label
        # whenever the prediction returns to unknown (hand goes neutral).
        # See class docstring for the rationale over frame-by-frame label tracking.
        self._last_dispatched_label: str = unknown_label

    @property
    def mode(self) -> RuntimeMode:
        return self._mode

    def route(self, prediction: StabilizedPrediction) -> ExecutableCommand | None:
        # Continuous gestures (e.g. point_one, point_two) are owned exclusively
        # by LaserPointerTracker and must never enter the ARMED/COOLDOWN state
        # machine.  Routing them here would cause CommandDispatcher to send a
        # deactivation keystroke on every accepted window, fighting LaserPointerTracker's
        # laser pointer activation.
        if prediction.label in self._continuous_labels:
            return None

        now = self._now_ms()

        # ------------------------------------------------------------------
        # COOLDOWN -- wait for the inter-command window to expire.
        # ------------------------------------------------------------------
        if self._mode is RuntimeMode.COOLDOWN:
            if self._cooldown_started_ms is not None and (
                now - self._cooldown_started_ms >= self._command_cooldown_ms
            ):
                self._mode = RuntimeMode.ARMED
                self._cooldown_started_ms = None
                # Reset dispatch memory so the same gesture can fire again
                # immediately without requiring a neutral hand pose first.
                self._last_dispatched_label = self._unknown_label
                LOGGER.debug(
                    "activation: COOLDOWN -> ARMED  (cooldown_ms=%d)",
                    self._command_cooldown_ms,
                )
            else:
                return None

        is_unknown = prediction.label == self._unknown_label

        # ------------------------------------------------------------------
        # ARMED -- dispatch command gestures.
        # ------------------------------------------------------------------
        if self._mode is RuntimeMode.ARMED:
            if is_unknown:
                # Hand returned to neutral -- clear the dispatch memory so that
                # performing the same gesture again (after a rest) fires normally.
                self._last_dispatched_label = self._unknown_label
                return None

            if not prediction.accepted:
                return None

            # Edge-trigger guard: dispatch only when the accepted label differs
            # from the last dispatched command label.  Unaccepted frames never
            # advance _last_dispatched_label, so the 2-3 unaccepted buildup frames
            # from temporal voting do not consume the rising edge.
            if self._edge_trigger_enabled and prediction.label == self._last_dispatched_label:
                LOGGER.debug(
                    "activation: edge_trigger suppressed  label=%r  (same as last dispatch)",
                    prediction.label,
                )
                return None

            command_id = self._command_map.get(prediction.label)
            if command_id is None:
                LOGGER.debug(
                    "activation: ARMED  gesture=%r not in command_map -- ignored",
                    prediction.label,
                )
                return None

            self._last_dispatched_label = prediction.label
            self._mode = RuntimeMode.COOLDOWN
            self._cooldown_started_ms = now
            LOGGER.info(
                "activation: ARMED -> COOLDOWN  gesture=%r  command_id=%r",
                prediction.label,
                command_id,
            )
            return ExecutableCommand(
                command_id=command_id,
                source_gesture=prediction.label,
                issued_at_ms=now,
            )

        return None

    def reset_dispatch_memory(self) -> None:
        """Reset dispatch memory without changing mode.

        Called at session boundaries or when re-initializing dispatch state.
        """
        self._last_dispatched_label = self._unknown_label
        self._cooldown_started_ms = None

    @staticmethod
    def _now_ms() -> int:
        return int(time.time() * 1000)
