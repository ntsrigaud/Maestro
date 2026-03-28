from __future__ import annotations

import logging

from gesture_control.domain.entities import DispatchResult, ExecutableCommand

LOGGER = logging.getLogger(__name__)


class KeyboardCommandDispatcher:
    """Dispatches logical commands to app-specific shortcuts.

    Current implementation is intentionally safe-by-default and logs dispatch intent.
    Shortcut execution can be connected to pyautogui/pynput in a follow-up phase.
    """

    _SHORTCUTS: dict[str, dict[str, str]] = {
        "powerpoint": {
            "next_slide": "Right",
            "previous_slide": "Left",
            "mute_toggle": "Ctrl+Shift+M",
            "raise_hand": "Ctrl+Shift+K",
            "start_presentation": "F5",
            "toggle_presenter_tools": "Ctrl+Shift+F10",
        },
        "zoom": {
            "next_slide": "Right",
            "previous_slide": "Left",
            "mute_toggle": "Alt+A",
            "raise_hand": "Alt+Y",
            "start_presentation": "Alt+F5",
            "toggle_presenter_tools": "Alt+F1",
        },
    }

    def __init__(self, *, active_app: str) -> None:
        self._active_app = active_app.lower()

    def dispatch(self, command: ExecutableCommand) -> DispatchResult:
        app_shortcuts = self._SHORTCUTS.get(self._active_app, {})
        shortcut = app_shortcuts.get(command.command_id)
        if shortcut is None:
            detail = (
                f"No shortcut mapping for command={command.command_id} "
                f"in app={self._active_app}"
            )
            LOGGER.warning(detail)
            return DispatchResult(
                success=False,
                command_id=command.command_id,
                target_app=self._active_app,
                detail=detail,
            )

        LOGGER.info(
            "Dispatch command: command_id=%s gesture=%s app=%s shortcut=%s",
            command.command_id,
            command.source_gesture,
            self._active_app,
            shortcut,
        )
        return DispatchResult(
            success=True,
            command_id=command.command_id,
            target_app=self._active_app,
            detail=f"shortcut={shortcut}",
        )
