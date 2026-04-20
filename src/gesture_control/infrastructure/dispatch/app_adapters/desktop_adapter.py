"""Desktop interaction adapter.

Covers four use cases for close-range (desk distance) gesture interaction:
  1. Universal document scrolling via pyautogui.scroll().
  2. View zoom via modifier+scroll (Cmd+scroll on macOS, Ctrl+scroll elsewhere).
  3. Workspace / tab navigation via per-platform keyboard shortcuts.
  4. OS overview (Mission Control / Task View / Activities) via platform hotkeys.

All OS-level interactions are delegated to KeyboardAdapter so that the scroll
and zoom injection paths remain centralised and safe.
"""

from __future__ import annotations

import logging
import sys

from gesture_control.domain.entities import DispatchResult
from gesture_control.infrastructure.dispatch.app_adapters.keyboard_adapter import (
    KeyboardAdapter,
)

LOGGER = logging.getLogger(__name__)

_TARGET_APP = "desktop"

# Navigation shortcuts keyed by (platform, navigation_mode).
# The platform key is normalised at construction time ("linux2" → "linux").
_PLATFORM_ZOOM_MODIFIER: dict[str, str] = {
    "darwin": "command",
    "win32": "ctrl",
    "linux": "ctrl",
}

_NAV_SHORTCUTS: dict[tuple[str, str], dict[str, tuple[str, ...]]] = {
    ("darwin", "workspace"): {
        "navigate_forward": ("ctrl", "right"),
        "navigate_back": ("ctrl", "left"),
    },
    ("win32", "workspace"): {
        "navigate_forward": ("ctrl", "win", "right"),
        "navigate_back": ("ctrl", "win", "left"),
    },
    ("linux", "workspace"): {
        "navigate_forward": ("ctrl", "alt", "right"),
        "navigate_back": ("ctrl", "alt", "left"),
    },
    ("darwin", "tabs"): {
        "navigate_forward": ("ctrl", "tab"),
        "navigate_back": ("ctrl", "shift", "tab"),
    },
    ("win32", "tabs"): {
        "navigate_forward": ("ctrl", "tab"),
        "navigate_back": ("ctrl", "shift", "tab"),
    },
    ("linux", "tabs"): {
        "navigate_forward": ("ctrl", "tab"),
        "navigate_back": ("ctrl", "shift", "tab"),
    },
}

# OS overview shortcuts keyed by platform.
# Linux defaults to GNOME (Super+A).  When linux_de="kde" is passed, the
# overview shortcut is overridden to Super+W at construction time.
_OVERVIEW_SHORTCUTS: dict[str, tuple[str, ...]] = {
    "darwin": ("ctrl", "up"),  # Mission Control (F3 is equivalent)
    "win32": ("win", "tab"),  # Task View
    "linux": ("super", "a"),  # GNOME Activities (default)
}

_OVERVIEW_SHORTCUTS_LINUX_DE: dict[str, tuple[str, ...]] = {
    "gnome": ("super", "a"),  # GNOME Activities
    "kde": ("super", "w"),  # KDE Present Windows
}

_KNOWN_PLATFORMS: frozenset[str] = frozenset({"darwin", "win32", "linux"})

# Module-level defaults.  Phase 3 replaces these with per-config values
# injected at construction time.
_SCROLL_CLICKS = 5  # pyautogui scroll clicks per gesture event
_ZOOM_CLICKS = 3  # modifier+scroll steps per zoom gesture event


class DesktopAdapter:
    """Desktop interaction adapter.

    Translates logical command IDs into OS-level actions via
    :class:`KeyboardAdapter`.  Platform normalisation and navigation mode
    selection happen at construction time; the :meth:`execute` method contains
    no runtime branching on platform or mode.
    """

    @property
    def target_app(self) -> str:
        return _TARGET_APP

    def __init__(
        self,
        keyboard: KeyboardAdapter | None = None,
        *,
        navigation_mode: str = "workspace",
        scroll_clicks: int = _SCROLL_CLICKS,
        zoom_clicks: int = _ZOOM_CLICKS,
        linux_de: str = "gnome",
        platform: str | None = None,
    ) -> None:
        self._keyboard = keyboard or KeyboardAdapter()
        self._navigation_mode = navigation_mode
        self._scroll_clicks = scroll_clicks
        self._zoom_clicks = zoom_clicks

        _platform = platform if platform is not None else sys.platform
        # Normalise Linux variants (e.g. "linux2" on older CPython builds).
        if _platform.startswith("linux"):
            _platform = "linux"
        if _platform not in _KNOWN_PLATFORMS:
            raise ValueError(
                f"DesktopAdapter: unsupported platform {_platform!r}. "
                "Supported: darwin, win32, linux."
            )
        self._platform = _platform

        # Resolve shortcuts eagerly to catch misconfiguration at construction.
        self._nav_shortcuts = _NAV_SHORTCUTS[(self._platform, self._navigation_mode)]
        # Linux overview: use DE-specific shortcut when known, else GNOME default.
        if self._platform == "linux":
            de_key = (linux_de or "gnome").lower()
            self._overview_keys = _OVERVIEW_SHORTCUTS_LINUX_DE.get(
                de_key, _OVERVIEW_SHORTCUTS["linux"]
            )
        else:
            self._overview_keys = _OVERVIEW_SHORTCUTS[self._platform]
        self._zoom_modifier = _PLATFORM_ZOOM_MODIFIER[self._platform]

    def execute(self, command_id: str) -> DispatchResult:
        """Execute the action corresponding to *command_id*.

        Returns a :class:`DispatchResult` describing success or failure without
        raising; all adapter-level exceptions are caught and logged here.
        """
        try:
            return self._dispatch(command_id)
        except Exception as exc:
            detail = f"adapter_error={exc!r}"
            LOGGER.exception("DesktopAdapter execute failed: command_id=%s", command_id)
            return DispatchResult(
                success=False,
                command_id=command_id,
                target_app=_TARGET_APP,
                detail=detail,
            )

    # ------------------------------------------------------------------
    # Internal dispatch (no exception handling — caller catches all)
    # ------------------------------------------------------------------

    def _dispatch(self, command_id: str) -> DispatchResult:
        if command_id == "scroll_forward":
            # Negative clicks advance the document (scroll view down).
            self._keyboard.scroll(-self._scroll_clicks)

        elif command_id == "scroll_back":
            # Positive clicks move view up (go back in document).
            self._keyboard.scroll(self._scroll_clicks)

        elif command_id == "zoom_in_view":
            self._keyboard.zoom(self._zoom_clicks, modifier=self._zoom_modifier)

        elif command_id == "zoom_out_view":
            self._keyboard.zoom(-self._zoom_clicks, modifier=self._zoom_modifier)

        elif command_id in ("navigate_forward", "navigate_back"):
            keys = self._nav_shortcuts[command_id]
            self._keyboard.press_hotkey(*keys)

        elif command_id == "mission_control":
            self._keyboard.press_hotkey(*self._overview_keys)

        elif command_id == "no_action":
            return DispatchResult(
                success=True,
                command_id=command_id,
                target_app=_TARGET_APP,
                detail="no_action",
            )

        else:
            detail = f"no_handler_for={command_id}"
            LOGGER.warning("DesktopAdapter: %s", detail)
            return DispatchResult(
                success=False,
                command_id=command_id,
                target_app=_TARGET_APP,
                detail=detail,
            )

        LOGGER.debug("DesktopAdapter: executed command_id=%s", command_id)
        return DispatchResult(
            success=True,
            command_id=command_id,
            target_app=_TARGET_APP,
        )
