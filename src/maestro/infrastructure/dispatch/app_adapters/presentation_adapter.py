"""Presentation controller adapter.

Maps the eight IPN Hand gesture commands to universal input actions for
presentation software (PowerPoint, Keynote, Google Slides, LibreOffice Impress).
Input is delivered via OS-level keyboard and scroll simulation; no
application-process detection is required — the action is delivered to whichever
window holds focus.

All keyboard injection is delegated to :class:`KeyboardAdapter` so that every
simulated keystroke is validated against the safe-key allowlist before dispatch.

Zoom implementation
-------------------
Zoom is implemented via modifier+scroll (Cmd+Scroll on macOS, Ctrl+Scroll on
Windows and Linux) rather than keyboard shortcuts (Cmd/Ctrl+=/-).  This choice
provides correct behaviour across all target presentation applications:

- **PowerPoint (Windows / macOS)**: Has no reliable zoom keyboard shortcut;
  Ctrl/Cmd+Scroll is the standard zoom input accepted by PowerPoint's slide
  canvas in editing view.
- **Keynote (macOS)**: Accepts both Cmd+= and Cmd+Scroll; scroll is preferred
  for consistency.
- **Google Slides (browser)**: Ctrl/Cmd+Scroll triggers browser viewport zoom,
  equivalent to Ctrl/Cmd+= but more consistent across browsers.
- **LibreOffice Impress (all platforms)**: Has no Ctrl+= zoom shortcut;
  Ctrl+Scroll is the only reliable zoom mechanism.
"""

from __future__ import annotations

import logging
import sys

from maestro.domain.entities import DispatchResult
from maestro.infrastructure.dispatch.app_adapters.keyboard_adapter import (
    KeyboardAdapter,
)

LOGGER = logging.getLogger(__name__)

_TARGET_APP = "presentation"

# Platform modifier for view zoom (Cmd on macOS, Ctrl on Windows / Linux).
_ZOOM_MODIFIER: dict[str, str] = {
    "darwin": "command",
    "win32": "ctrl",
    "linux": "ctrl",
}

# Slideshow start / stop shortcut per platform.
# PowerPoint macOS: Cmd+Return; Keynote: Cmd+Return; Google Slides: Ctrl+F5.
# LibreOffice / PowerPoint Win/Linux: F5.
# Using F5 universally on Win/Linux covers the majority of software.
_SLIDESHOW_KEYS: dict[str, tuple[str, ...]] = {
    "darwin": ("command", "enter"),
    "win32": ("f5",),
    "linux": ("f5",),
}


class PresentationAdapter:
    """Maps gesture commands to presentation-controller keyboard shortcuts.

    All logical commands from ``command_map_presentation.yaml`` are handled.
    Unknown command IDs return a non-executed :class:`DispatchResult` rather
    than raising an exception, so the runtime loop is never interrupted.

    Canonical gesture-to-command mapping
    -------------------------------------
    swipe_up        → start_presentation
    swipe_down      → stop_presentation
    swipe_right     → next_slide
    swipe_left      → previous_slide
    zoom_in         → zoom_in_view
    zoom_out        → zoom_out_view
    open_palm_hold  → erase_annotations
    pinch           → activate_laser_pointer
    click           → mouse_click
    point_one       → mouse_pointer (handled by LaserPointerTracker)
    point_two       → mouse_pointer (handled by AnnotationPenTracker)
    unknown         → no_action
    """

    @property
    def target_app(self) -> str:
        return _TARGET_APP

    def __init__(self, keyboard: KeyboardAdapter | None = None) -> None:
        self._keyboard = keyboard or KeyboardAdapter()
        platform = sys.platform if sys.platform in ("darwin", "win32") else "linux"
        self._zoom_modifier = _ZOOM_MODIFIER[platform]
        self._slideshow_keys = _SLIDESHOW_KEYS[platform]

    def execute(self, command_id: str) -> DispatchResult:
        """Translate *command_id* into an OS-level keyboard action.

        Args:
            command_id: Logical command from the gesture-to-command map.

        Returns:
            :class:`DispatchResult` with ``success=True`` when a key was sent,
            ``success=False`` for ``no_action`` and unrecognised command IDs.
        """
        executed = False
        match command_id:
            case "start_presentation":
                # Start the slideshow: Cmd+Return (macOS) / F5 (Win/Linux).
                executed = self._keyboard.press_hotkey(*self._slideshow_keys)
            case "stop_presentation":
                # Stop / exit the slideshow universally.
                executed = self._keyboard.press_hotkey("escape")
            case "next_slide":
                executed = self._keyboard.press_hotkey("right")
            case "previous_slide":
                executed = self._keyboard.press_hotkey("left")
            case "zoom_in_view":
                executed = self._keyboard.zoom(3, modifier=self._zoom_modifier)
            case "zoom_out_view":
                executed = self._keyboard.zoom(-3, modifier=self._zoom_modifier)
            case "zoom_out_totally":
                # Zoom out to the minimum by scrolling at maximum magnitude.
                executed = self._keyboard.zoom(-5, modifier=self._zoom_modifier)
            case "erase_annotations":
                # Erase all pen / laser annotations on the current slide.
                # PowerPoint slideshow shortcut: E key (Windows and macOS).
                executed = self._keyboard.press_hotkey("e")
            case "toggle_black_screen":
                # Blank the slide canvas to black — universal pause gesture.
                # B key is supported by PowerPoint, Keynote, Google Slides,
                # and LibreOffice Impress during an active slideshow.
                executed = self._keyboard.press_hotkey("b")
            case "toggle_white_screen":
                # Blank the slide canvas to white.
                # W key is supported by PowerPoint, Keynote, Google Slides,
                # and LibreOffice Impress during an active slideshow.
                executed = self._keyboard.press_hotkey("w")
            case "go_to_first_slide":
                # Jump to the first slide of the presentation.
                # Home key is universally accepted by all target applications.
                executed = self._keyboard.press_hotkey("home")
            case "go_to_last_slide":
                # Jump to the last slide of the presentation.
                # End key is universally accepted by all target applications.
                executed = self._keyboard.press_hotkey("end")
            case "mouse_click":
                # Perform a left-click at the current cursor position.
                # Used to activate hyperlinks and trigger animations on slides.
                executed = self._keyboard.mouse_click()
            case "mouse_pointer":
                # Revert cursor to the default arrow pointer (Ctrl+A in
                # PowerPoint slideshow; no-op on apps without a laser mode).
                executed = self._keyboard.press_hotkey("ctrl", "a")
            # ── Legacy / extended commands (not in default gesture map) ────
            case "scroll_forward":
                executed = self._keyboard.press_hotkey("down")
            case "scroll_back":
                executed = self._keyboard.press_hotkey("up")
            case "start_stop_presentation":
                executed = self._keyboard.press_hotkey(*self._slideshow_keys)
            case "activate_laser_pointer":
                # Ctrl+L toggles the laser pointer on during a running slideshow.
                # Universal across PowerPoint (Windows / macOS).
                executed = self._keyboard.press_hotkey("ctrl", "l")
            case "deactivate_laser_pointer":
                # Ctrl+A reverts the pointer to the default arrow cursor.
                executed = self._keyboard.press_hotkey("ctrl", "a")
            case "no_action":
                return DispatchResult(
                    success=False,
                    command_id=command_id,
                    target_app=_TARGET_APP,
                    detail="no_action",
                )
            case _:
                LOGGER.warning("PresentationAdapter: unrecognised command %r", command_id)
                return DispatchResult(
                    success=False,
                    command_id=command_id,
                    target_app=_TARGET_APP,
                    detail=f"unknown_command:{command_id}",
                )

        if not executed:
            return DispatchResult(
                success=False,
                command_id=command_id,
                target_app=_TARGET_APP,
                detail="blocked_outside_presentation_scope",
            )

        LOGGER.debug("PresentationAdapter executed: %s", command_id)
        return DispatchResult(
            success=True,
            command_id=command_id,
            target_app=_TARGET_APP,
            detail=None,
        )
