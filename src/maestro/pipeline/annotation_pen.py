"""Annotation pen tracker for continuous pointing gestures.

Drives PowerPoint's built-in annotation pen by mapping the index fingertip
landmark to screen coordinates every frame.  The tracker is stateful and
operates independently of the edge-trigger ``GestureActivationController``:
it monitors every frame rather than waiting for a cooldown window to expire.

Platform-specific activation
-----------------------------
Microsoft PowerPoint uses different keyboard shortcuts per platform to toggle
the annotation pen on and off during a running slideshow:

- **macOS**: ``⌘+P`` (Command+P) activates; ``⌘+A`` (Command+A) reverts the
  cursor to the arrow pointer.
- **Windows / Linux**: ``Ctrl+P`` activates; ``Ctrl+A`` reverts.

The correct key tuple is selected automatically from ``sys.platform`` via the
module-level ``_PEN_ON_KEYS`` / ``_PEN_OFF_KEYS`` dicts.  Tests can override
the selection by passing ``pen_on_keys`` / ``pen_off_keys`` to the constructor.

Lifecycle
---------
- **Entry** (first frame where ``point_two`` is accepted):
  The pen-on hotkey (``⌘+P`` or ``Ctrl+P``) is pressed once to switch
  PowerPoint into annotation pen mode.  The cursor is then positioned at the
  index fingertip and the primary mouse button is pressed down, starting the
  pen stroke.
- **Sustained** (subsequent frames with the same pointing gesture):
  The OS mouse cursor is dragged (button held) to the screen position of the
  index fingertip each frame.  PowerPoint draws the annotation path in
  real-time while the button is held.
- **Exit** (gesture returns to ``unknown`` or runtime stops):
  The primary mouse button is released first (ending the current stroke), then
  the pen-off hotkey (``⌘+A`` or ``Ctrl+A``) is sent to revert the cursor to
  the default arrow pointer.

Screen coordinate mapping
-------------------------
MediaPipe normalises landmarks to ``[0, 1]`` in the camera frame where
``x = 0`` is the left edge of the *captured* image.  Because webcam feeds are
typically mirrored for user comfort (``mirror_display: true`` in
``configs/app.yaml``), the physical left side of the scene maps to a *high*
x value in the raw image.  When ``mirror_x=True`` (the default), the tracker
applies ``screen_x = (1 - x_norm) * screen_width`` so that the pen tip
appears at the visually correct position on the presentation display.

As with the laser pointer, users usually operate inside a narrower central
region of the camera frame. The tracker therefore applies a centre-relative
gain before mapping to screen pixels so smaller hand motions produce larger
on-screen pen displacement while keeping the midpoint stable.
"""

from __future__ import annotations

import logging
import sys
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from maestro.infrastructure.dispatch.app_adapters.keyboard_adapter import (
        KeyboardAdapter,
    )

LOGGER = logging.getLogger(__name__)

# Index fingertip in the MediaPipe 21-point hand model.
_INDEX_FINGERTIP: int = 8

# Default set of gesture labels that activate the annotation pen.
_DEFAULT_PEN_LABELS: frozenset[str] = frozenset({"point_two"})

# Hotkeys to start / stop the annotation pen per platform.
# macOS PowerPoint (slideshow mode): ⌘+P activates, ⌘+A reverts to arrow.
# Windows / Linux PowerPoint (slideshow mode): Ctrl+P activates, Ctrl+A reverts.
_PEN_ON_KEYS: dict[str, tuple[str, ...]] = {
    "darwin": ("command", "p"),
    "win32": ("ctrl", "p"),
    "linux": ("ctrl", "p"),
}
_PEN_OFF_KEYS: dict[str, tuple[str, ...]] = {
    "darwin": ("command", "a"),
    "win32": ("ctrl", "a"),
    "linux": ("ctrl", "a"),
}

_DEFAULT_POINTER_GAIN_X: float = 2.4
_DEFAULT_POINTER_GAIN_Y: float = 2.2


def _platform_key() -> str:
    """Return a normalised platform key for the _PEN_* dicts."""
    return sys.platform if sys.platform in ("darwin", "win32") else "linux"


class AnnotationPenTracker:
    """Continuous annotation pen controller driven by pointing gestures.

    Parameters
    ----------
    keyboard:
        :class:`~maestro.infrastructure.dispatch.app_adapters.keyboard_adapter.KeyboardAdapter`
        used to send keystrokes and move the mouse.
    pointing_labels:
        Set of gesture label strings that should activate the annotation pen.
        Defaults to ``{"point_two"}``.
    mirror_x:
        When ``True``, the normalised x landmark coordinate is horizontally
        flipped before being mapped to screen pixels.  Set this to match the
        ``mirror_display`` setting in ``app.yaml``.
    pen_on_keys:
        Hotkey tuple sent to activate the pen (e.g. ``("command", "p")`` on
        macOS or ``("ctrl", "p")`` on Windows).  Defaults to the correct value
        for the current platform.  Override in tests for cross-platform coverage.
    pen_off_keys:
        Hotkey tuple sent to deactivate the pen (e.g. ``("command", "a")`` on
        macOS or ``("ctrl", "a")`` on Windows).  Defaults to the current platform.
    pointer_gain_x:
        Centre-relative horizontal gain applied to fingertip movement before
        mapping to the screen. Values > 1 expand cursor travel.
    pointer_gain_y:
        Centre-relative vertical gain applied to fingertip movement before
        mapping to the screen. Values > 1 expand cursor travel.
    """

    def __init__(
        self,
        *,
        keyboard: KeyboardAdapter,
        pointing_labels: frozenset[str] = _DEFAULT_PEN_LABELS,
        mirror_x: bool = True,
        pen_on_keys: tuple[str, ...] | None = None,
        pen_off_keys: tuple[str, ...] | None = None,
        pointer_gain_x: float = _DEFAULT_POINTER_GAIN_X,
        pointer_gain_y: float = _DEFAULT_POINTER_GAIN_Y,
    ) -> None:
        self._keyboard = keyboard
        self._pointing_labels = pointing_labels
        self._mirror_x = mirror_x
        self._active: bool = False
        self._mouse_pressed: bool = False  # True while the primary button is held for drawing
        self._pointer_gain_x = max(1.0, float(pointer_gain_x))
        self._pointer_gain_y = max(1.0, float(pointer_gain_y))
        _p = _platform_key()
        self._pen_on_keys: tuple[str, ...] = (
            pen_on_keys if pen_on_keys is not None else _PEN_ON_KEYS[_p]
        )
        self._pen_off_keys: tuple[str, ...] = (
            pen_off_keys if pen_off_keys is not None else _PEN_OFF_KEYS[_p]
        )

    # ------------------------------------------------------------------
    # Public interface
    # ------------------------------------------------------------------

    @property
    def active(self) -> bool:
        """``True`` while the annotation pen is currently active."""
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
            The current stabilised gesture label (e.g. ``"point_two"`` or
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
        """Unconditionally deactivate the annotation pen.

        Sends the pen-off hotkey to restore the arrow cursor if the pen is
        currently active.  Safe to call from the runtime ``finally`` block or
        during control-state transitions.
        """
        if self._active:
            self._deactivate("shutdown")

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _activate(self, label: str) -> None:
        try:
            self._keyboard.press_hotkey(*self._pen_on_keys)
            self._active = True
            LOGGER.info("AnnotationPenTracker: activated (gesture=%r)", label)
        except Exception:
            LOGGER.exception(
                "AnnotationPenTracker: failed to activate annotation pen (gesture=%r)", label
            )

    def _deactivate(self, label: str) -> None:
        try:
            # Release the mouse button first so PowerPoint ends the current
            # pen stroke cleanly before the mode-switch hotkey is sent.
            if self._mouse_pressed:
                self._keyboard.mouse_up()
                self._mouse_pressed = False
            self._keyboard.press_hotkey(*self._pen_off_keys)
            self._active = False
            LOGGER.info("AnnotationPenTracker: deactivated (gesture=%r)", label)
        except Exception:
            LOGGER.exception(
                "AnnotationPenTracker: failed to deactivate annotation pen (gesture=%r)", label
            )
            # Force flags to False so the tracker doesn't get stuck active.
            self._active = False
            self._mouse_pressed = False

    def _move_to_fingertip(self, landmarks: list[list[float]]) -> None:
        if len(landmarks) <= _INDEX_FINGERTIP:
            LOGGER.debug("AnnotationPenTracker: not enough landmarks (%d)", len(landmarks))
            return

        x_norm: float = landmarks[_INDEX_FINGERTIP][0]
        y_norm: float = landmarks[_INDEX_FINGERTIP][1]

        x_norm = self._apply_pointer_gain(x_norm, self._pointer_gain_x)
        y_norm = self._apply_pointer_gain(y_norm, self._pointer_gain_y)

        screen_w, screen_h = self._keyboard.screen_size()
        screen_x = int((1.0 - x_norm if self._mirror_x else x_norm) * screen_w)
        screen_y = int(y_norm * screen_h)

        # Clamp to screen bounds.
        screen_x = max(0, min(screen_w - 1, screen_x))
        screen_y = max(0, min(screen_h - 1, screen_y))

        if not self._mouse_pressed:
            # First active frame: move to position AND press the button so that
            # PowerPoint registers the start of the pen stroke at the fingertip.
            self._keyboard.mouse_down(screen_x, screen_y)
            self._mouse_pressed = True
            LOGGER.debug("AnnotationPenTracker: stroke start at (%d, %d)", screen_x, screen_y)
        else:
            # Subsequent frames: send a drag event while the button is held.
            # On macOS, pyautogui.dragTo posts kCGEventLeftMouseDragged which
            # is what PowerPoint requires to actually draw the pen stroke.
            # pyautogui.moveTo posts kCGEventMouseMoved, which PowerPoint's pen
            # mode silently ignores — the reason plain move_mouse didn't draw.
            self._keyboard.mouse_drag_to(screen_x, screen_y)
            LOGGER.debug("AnnotationPenTracker: drag to (%d, %d)", screen_x, screen_y)

    @staticmethod
    def _apply_pointer_gain(value: float, gain: float) -> float:
        """Expand movement around the frame centre, then clamp to [0, 1]."""
        centered = 0.5 + ((value - 0.5) * gain)
        return max(0.0, min(1.0, centered))
