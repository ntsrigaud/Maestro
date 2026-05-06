"""Laser pointer tracker for continuous pointing gestures.

Drives PowerPoint's built-in laser pointer (Ctrl+L) by mapping the index
fingertip landmark to screen coordinates every frame.  The tracker is stateful
and operates independently of the edge-trigger ``GestureActivationController``:
it monitors every frame rather than waiting for a cooldown window to expire.

Lifecycle
---------
- **Entry** (first frame where a pointing gesture is accepted):
  ``Ctrl+L`` is sent to activate the laser pointer in the running slideshow.
- **Sustained** (subsequent frames with the same pointing gesture):
  The OS mouse cursor is moved to the screen position of the index fingertip
  (landmark 8 in the MediaPipe 21-point hand model).  PowerPoint's laser dot
  follows the cursor in real-time.
- **Exit** (gesture returns to ``unknown`` or gaze is lost):
  ``Ctrl+A`` is sent to revert the pointer to the default arrow cursor.

Screen coordinate mapping
-------------------------
MediaPipe normalises landmarks to ``[0, 1]`` in the camera frame where
``x = 0`` is the left edge of the *captured* image.  Because webcam feeds are
typically mirrored for user comfort (``mirror_display: true`` in
``configs/app.yaml``), the physical left side of the scene maps to a *high*
x value in the raw image.  When ``mirror_x=True`` (the default), the tracker
applies ``screen_x = (1 - x_norm) * screen_width`` so that the laser dot
appears at the visually correct position on the presentation display.
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from maestro.infrastructure.dispatch.app_adapters.keyboard_adapter import (
        KeyboardAdapter,
    )

LOGGER = logging.getLogger(__name__)

# Index fingertip in the MediaPipe 21-point hand model.
_INDEX_FINGERTIP: int = 8

# Default set of gesture labels treated as "pointing".
_DEFAULT_POINTING_LABELS: frozenset[str] = frozenset({"point_one", "point_two"})


class LaserPointerTracker:
    """Continuous laser pointer controller driven by pointing gestures.

    Parameters
    ----------
    keyboard:
        :class:`~maestro.infrastructure.dispatch.app_adapters.keyboard_adapter.KeyboardAdapter`
        used to send keystrokes (``Ctrl+L`` / ``Ctrl+A``) and move the mouse.
    pointing_labels:
        Set of gesture label strings that should activate the laser pointer.
        Defaults to ``{"point_one", "point_two"}``.
    mirror_x:
        When ``True``, the normalised x landmark coordinate is horizontally
        flipped before being mapped to screen pixels.  Set this to match the
        ``mirror_display`` setting in ``app.yaml``.
    """

    def __init__(
        self,
        *,
        keyboard: KeyboardAdapter,
        pointing_labels: frozenset[str] = _DEFAULT_POINTING_LABELS,
        mirror_x: bool = True,
    ) -> None:
        self._keyboard = keyboard
        self._pointing_labels = pointing_labels
        self._mirror_x = mirror_x
        self._active: bool = False

    # ------------------------------------------------------------------
    # Public interface
    # ------------------------------------------------------------------

    @property
    def active(self) -> bool:
        """``True`` while the laser pointer is currently active."""
        return self._active

    def update(
        self,
        label: str,
        landmarks: list[list[float]] | None,
        valid: bool,
    ) -> None:
        """Process one frame.

        Parameters
        ----------
        label:
            The current stabilised gesture label (e.g. ``"point_one"`` or
            ``"unknown"``).
        landmarks:
            21-entry list of ``[x, y, z]`` normalised landmark coordinates, or
            ``None`` when no valid hand is detected.
        valid:
            Whether *landmarks* represent a valid hand detection.  When
            ``False`` the tracker treats the frame as "no hand present".
        """
        is_pointing = label in self._pointing_labels and valid and landmarks is not None

        if is_pointing and not self._active:
            self._activate(label)
        elif not is_pointing and self._active:
            self._deactivate(label)

        if is_pointing and self._active and landmarks is not None:
            self._move_to_fingertip(landmarks)

    def deactivate(self) -> None:
        """Unconditionally deactivate the laser pointer.

        Sends ``Ctrl+A`` to restore the arrow cursor if the pointer is
        currently active.  Safe to call from the runtime ``finally`` block or
        on gaze loss.
        """
        if self._active:
            self._deactivate("shutdown")

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _activate(self, label: str) -> None:
        try:
            self._keyboard.press_hotkey("ctrl", "l")
            self._active = True
            LOGGER.info("LaserPointerTracker: activated (gesture=%r)", label)
        except Exception:
            LOGGER.exception(
                "LaserPointerTracker: failed to activate laser pointer (gesture=%r)", label
            )

    def _deactivate(self, label: str) -> None:
        try:
            self._keyboard.press_hotkey("ctrl", "a")
            self._active = False
            LOGGER.info("LaserPointerTracker: deactivated (gesture=%r)", label)
        except Exception:
            LOGGER.exception(
                "LaserPointerTracker: failed to deactivate laser pointer (gesture=%r)", label
            )
            # Force the flag to False so the tracker doesn't get stuck active.
            self._active = False

    def _move_to_fingertip(self, landmarks: list[list[float]]) -> None:
        if len(landmarks) <= _INDEX_FINGERTIP:
            LOGGER.debug("LaserPointerTracker: not enough landmarks (%d)", len(landmarks))
            return

        x_norm: float = landmarks[_INDEX_FINGERTIP][0]
        y_norm: float = landmarks[_INDEX_FINGERTIP][1]

        screen_w, screen_h = self._keyboard.screen_size()
        screen_x = int((1.0 - x_norm if self._mirror_x else x_norm) * screen_w)
        screen_y = int(y_norm * screen_h)
        # Clamp to valid screen region.
        screen_x = max(0, min(screen_w - 1, screen_x))
        screen_y = max(0, min(screen_h - 1, screen_y))

        self._keyboard.move_mouse(screen_x, screen_y)
