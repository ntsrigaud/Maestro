from __future__ import annotations

import logging

LOGGER = logging.getLogger(__name__)

try:
    import pyautogui as pyautogui  # noqa: PLC0414  (re-exported for patching)
except ImportError:  # pragma: no cover
    pyautogui = None  # type: ignore[assignment]

# Allowlisted key names accepted by pyautogui.hotkey().
#
# Only keys required for presentation and meeting control workflows are
# included.  This prevents arbitrary key injection if a command_id string
# somehow arrives from an untrusted source: the adapter layer resolves it to a
# predefined tuple of key names, and every key in that tuple is validated here
# before pyautogui is invoked.
#
# Extend this set deliberately when adding new shortcuts; never pass user-
# supplied strings directly to pyautogui.
_SAFE_KEY_NAMES: frozenset[str] = frozenset(
    {
        # Modifier keys
        "ctrl",
        "shift",
        "alt",
        # Arrow keys
        "right",
        "left",
        "up",
        "down",
        # Function keys
        "f1",
        "f2",
        "f3",
        "f4",
        "f5",
        "f6",
        "f7",
        "f8",
        "f9",
        "f10",
        "f11",
        "f12",
        # Special keys
        "enter",
        "space",
        "escape",
        "tab",
        "backspace",
        # Letter keys used in presentation/meeting shortcuts
        "a",
        "f",
        "k",
        "l",
        "m",
        "n",
        "p",
        "s",
        "w",
        "y",
    }
)


class KeyboardAdapter:
    """Low-level key delivery service backed by *pyautogui*.

    Validates every key name against :data:`_SAFE_KEY_NAMES` before invoking
    ``pyautogui.hotkey``.  This is the only location in the codebase where
    simulated keystrokes are sent to the OS; all higher layers operate on
    logical command IDs, never on raw key strings.

    ``pyautogui`` is imported lazily so that the rest of the package can be
    imported and tested without the dependency present.  When the library is
    missing a warning is emitted and the call is silently skipped, enabling
    headless / CI environments to run without OS-level side effects.
    """

    def press_hotkey(self, *keys: str) -> None:
        """Validate *keys* and simulate a simultaneous keystroke.

        Args:
            *keys: One or more key names (e.g. ``"ctrl"``, ``"shift"``, ``"m"``).

        Raises:
            ValueError: If any key name is not in :data:`_SAFE_KEY_NAMES`.
        """
        if not keys:
            raise ValueError("press_hotkey requires at least one key name.")

        for key in keys:
            if key.lower() not in _SAFE_KEY_NAMES:
                raise ValueError(
                    f"Key {key!r} is not in the safe key allowlist. "
                    "Add it explicitly to keyboard_adapter._SAFE_KEY_NAMES "
                    "if the shortcut is intentional."
                )

        if pyautogui is None:  # pragma: no cover
            LOGGER.warning(
                "pyautogui is not installed — keyboard dispatch skipped. "
                "Install with `pip install pyautogui` to enable OS-level key delivery."
            )
            return

        pyautogui.hotkey(*[k.lower() for k in keys])
        LOGGER.debug("Pressed hotkey: %s", "+".join(k.lower() for k in keys))
