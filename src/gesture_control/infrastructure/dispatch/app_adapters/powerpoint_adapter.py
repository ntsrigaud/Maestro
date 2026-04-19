from __future__ import annotations

import logging

from gesture_control.domain.entities import DispatchResult
from gesture_control.infrastructure.dispatch.app_adapters.keyboard_adapter import (
    KeyboardAdapter,
)

LOGGER = logging.getLogger(__name__)

_TARGET_APP = "powerpoint"

# Number of scroll clicks injected per scroll_forward / scroll_back command.
# Positive values scroll up (toward the top of the document); negative scroll
# down.  pyautogui.scroll() uses OS-native units, so 5 is a gentle scroll on
# most platforms.
_SCROLL_CLICKS = 5

# PowerPoint keyboard shortcuts for each supported command ID.
# All key names must be present in KeyboardAdapter._SAFE_KEY_NAMES.
_SHORTCUTS: dict[str, tuple[str, ...]] = {
    "next_slide": ("right",),
    "previous_slide": ("left",),
    "start_presentation": ("shift", "f5"),
    "raise_hand": ("ctrl", "shift", "k"),
    "toggle_pointer_mode": ("ctrl", "l"),
    "exit_presentation": ("escape",),
}

# Commands handled by scroll injection rather than keyboard shortcuts.
_SCROLL_COMMANDS: frozenset[str] = frozenset({"scroll_forward", "scroll_back"})


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
        # Universal no-op — always succeeds silently.
        if command_id == "no_action":
            return DispatchResult(
                success=True, command_id=command_id, target_app=_TARGET_APP, detail="no_action"
            )

        # Scroll injection is application-agnostic: pyautogui.scroll() targets
        # whichever window the OS cursor is over, regardless of active_app.
        if command_id in _SCROLL_COMMANDS:
            clicks = _SCROLL_CLICKS if command_id == "scroll_forward" else -_SCROLL_CLICKS
            try:
                self._keyboard.scroll(clicks)
                return DispatchResult(
                    success=True,
                    command_id=command_id,
                    target_app=_TARGET_APP,
                    detail=f"scroll_clicks={clicks}",
                )
            except Exception as exc:
                LOGGER.error(
                    "PowerPoint scroll failed: command_id=%s error=%s", command_id, exc
                )
                return DispatchResult(
                    success=False, command_id=command_id, target_app=_TARGET_APP, detail=str(exc)
                )

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
