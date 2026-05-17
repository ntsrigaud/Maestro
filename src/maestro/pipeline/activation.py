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
    ARMED    -> IDLE     : no recognised gesture for *idle_timeout_ms* when
                           ``auto_idle_enabled=True``.
    COOLDOWN -> ARMED    : cooldown window expires.
    IDLE     -> ARMED    : explicit ``rearm()`` call (gaze re-established).

    Activation gate
    ---------------
    The controller starts in IDLE state.  It transitions to ARMED only when
    ``rearm()`` is called, which the runtime loop invokes on the gaze-permitted
    rising edge (i.e. the first frame where ``gaze_permitted`` becomes ``True``
    after having been ``False``).  When the gaze grace period expires, the
    runtime calls ``force_idle()`` to transition the controller back to IDLE.

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
    --- the label of the most recently executed command.
    ``_last_dispatched_label`` is reset to ``unknown`` whenever the prediction
    returns to the ``unknown`` class (the hand goes neutral between gestures).

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

    The same-gesture-twice case is handled by the ``unknown`` reset: after the
    hand returns to neutral between two identical gestures,
    ``_last_dispatched_label`` is cleared, allowing the second occurrence to fire.

    The cooldown remains as a secondary guard against rapid re-triggering in
    unusual conditions (e.g. alternating accepted/unaccepted despite short hold).
    """

    def __init__(
        self,
        *,
        command_map: dict[str, str],
        idle_timeout_ms: int,
        command_cooldown_ms: int,
        edge_trigger_enabled: bool = True,
        continuous_labels: frozenset[str] = frozenset(),
        unknown_label: str = "unknown",
        auto_idle_enabled: bool = True,
    ) -> None:
        self._command_map = dict(command_map)
        self._idle_timeout_ms = idle_timeout_ms
        self._command_cooldown_ms = command_cooldown_ms
        self._edge_trigger_enabled = edge_trigger_enabled
        self._continuous_labels = continuous_labels
        self._unknown_label = unknown_label
        self._auto_idle_enabled = auto_idle_enabled

        self._mode: RuntimeMode = RuntimeMode.IDLE
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
        # Continuous gestures (e.g. point_one, point_two) are owned exclusively
        # by LaserPointerTracker and must never enter the IDLE/ARMED/COOLDOWN
        # state machine.  Routing them here would cause CommandDispatcher to
        # send a deactivation keystroke on every accepted window, fighting
        # LaserPointerTracker's laser pointer activation.
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
                LOGGER.debug(
                    "activation: COOLDOWN -> ARMED  (cooldown_ms=%d)",
                    self._command_cooldown_ms,
                )
            else:
                return None

        # ------------------------------------------------------------------
        # IDLE -- do not dispatch; wait for rearm() to be called externally.
        # ------------------------------------------------------------------
        if self._mode is RuntimeMode.IDLE:
            return None

        is_unknown = prediction.label == self._unknown_label

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
        # ARMED -- check idle timeout, then dispatch command gestures.
        # ------------------------------------------------------------------
        if self._mode is RuntimeMode.ARMED:
            if self._auto_idle_enabled and now - self._last_active_ms > self._idle_timeout_ms:
                self._mode = RuntimeMode.IDLE
                LOGGER.info(
                    "activation: ARMED -> IDLE  (no gesture for %d ms)",
                    self._idle_timeout_ms,
                )
                return None

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

    def rearm(self) -> None:
        """Transition from IDLE to ARMED.

        Called by the runtime when gaze is re-established after a
        ``force_idle()`` call.  Clears dispatch memory so that re-arming
        always starts fresh.
        """
        if self._mode is RuntimeMode.IDLE:
            self._mode = RuntimeMode.ARMED
            self._last_dispatched_label = self._unknown_label
            self._last_active_ms = self._now_ms()
            LOGGER.info("activation: IDLE -> ARMED  (gaze restored)")

    def force_idle(self) -> None:
        """Reset the controller to IDLE unconditionally.

        Called by the runtime loop when the gaze grace period expires while
        the controller is not already in IDLE state.  Clears all in-progress
        timers so that the next activation cycle starts from a clean state.
        """
        self._mode = RuntimeMode.IDLE
        self._cooldown_started_ms = None
        LOGGER.info("activation: forced -> IDLE  (gaze lost)")

    @staticmethod
    def _now_ms() -> int:
        return int(time.time() * 1000)
