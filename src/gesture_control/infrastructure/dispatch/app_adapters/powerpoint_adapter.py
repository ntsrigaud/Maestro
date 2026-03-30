from __future__ import annotations

import logging

from gesture_control.domain.entities import DispatchResult
from gesture_control.infrastructure.dispatch.app_adapters.keyboard_adapter import (
    KeyboardAdapter,
)

LOGGER = logging.getLogger(__name__)

_TARGET_APP = "powerpoint"

# PowerPoint keyboard shortcuts for each supported command ID.
# All key names must be present in KeyboardAdapter._SAFE_KEY_NAMES.
_SHORTCUTS: dict[str, tuple[str, ...]] = {
    "next_slide": ("right",),
    "previous_slide": ("left",),
    "start_presentation": ("f5",),
    "mute_toggle": ("ctrl", "shift", "m"),
    "raise_hand": ("ctrl", "shift", "k"),
    "toggle_presenter_tools": ("ctrl", "shift", "f10"),
    "toggle_pointer_mode": ("ctrl", "l"),
    "exit_presentation": ("escape",),
}


class PowerPointAdapter:
    """Presentation adapter for Microsoft PowerPoint.

    Translates logical command IDs into PowerPoint keyboard shortcuts and
    delegates key delivery to :class:`KeyboardAdapter`.  All exceptions from
    the keyboard layer are caught and returned as a failed
    :class:`DispatchResult` so the caller always receives a predictable value.
    """

    @property
    def target_app(self) -> str:
        return _TARGET_APP

    def __init__(self, keyboard: KeyboardAdapter | None = None) -> None:
        self._keyboard = keyboard or KeyboardAdapter()

    def execute(self, command_id: str) -> DispatchResult:
        keys = _SHORTCUTS.get(command_id)
        if keys is None:
            detail = f"no_shortcut_for={command_id}"
            LOGGER.warning("PowerPoint adapter: %s", detail)
            return DispatchResult(
                success=False,
                command_id=command_id,
                target_app=_TARGET_APP,
                detail=detail,
            )

        LOGGER.debug(
            "PowerPoint adapter: executing command_id=%s keys=%s",
            command_id,
            keys,
        )
        try:
            self._keyboard.press_hotkey(*keys)
            shortcut_str = "+".join(keys)
            return DispatchResult(
                success=True,
                command_id=command_id,
                target_app=_TARGET_APP,
                detail=f"shortcut={shortcut_str}",
            )
        except Exception as exc:
            LOGGER.error(
                "PowerPoint keyboard execution failed: command_id=%s error=%s",
                command_id,
                exc,
            )
            return DispatchResult(
                success=False,
                command_id=command_id,
                target_app=_TARGET_APP,
                detail=str(exc),
            )
