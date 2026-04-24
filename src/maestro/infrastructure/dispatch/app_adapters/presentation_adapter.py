"""Presentation controller adapter.

Maps the eight IPN Hand gesture commands to universal keyboard shortcuts for
presentation software (PowerPoint, Keynote, Google Slides, LibreOffice Impress).
Commands are injected via OS-level keyboard simulation; no application-process
detection is required — the shortcut is delivered to whichever window holds focus.

All keyboard injection is delegated to :class:`KeyboardAdapter` so that every
simulated keystroke is validated against the safe-key allowlist before dispatch.
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

    All eight logical commands from ``command_map_presentation.yaml`` are
    handled.  Unknown command IDs return a non-executed :class:`DispatchResult`
    rather than raising an exception, so the runtime loop is never interrupted.
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
        match command_id:
            case "next_slide":
                self._keyboard.press_hotkey("right")
            case "previous_slide":
                self._keyboard.press_hotkey("left")
            case "scroll_forward":
                self._keyboard.press_hotkey("down")
            case "scroll_back":
                self._keyboard.press_hotkey("up")
            case "zoom_in_view":
                self._keyboard.press_hotkey(self._zoom_modifier, "equal")
            case "zoom_out_view":
                self._keyboard.press_hotkey(self._zoom_modifier, "minus")
            case "start_stop_presentation":
                self._keyboard.press_hotkey(*self._slideshow_keys)
            case "no_action":
                return DispatchResult(
                    success=False,
                    command_id=command_id,
                    target_app=_TARGET_APP,
                    detail="no_action",
                )
            case _:
                LOGGER.warning(
                    "PresentationAdapter: unrecognised command %r", command_id
                )
                return DispatchResult(
                    success=False,
                    command_id=command_id,
                    target_app=_TARGET_APP,
                    detail=f"unknown_command:{command_id}",
                )

        LOGGER.debug("PresentationAdapter executed: %s", command_id)
        return DispatchResult(
            success=True,
            command_id=command_id,
            target_app=_TARGET_APP,
            detail=None,
        )
