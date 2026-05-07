from __future__ import annotations

import logging

LOGGER = logging.getLogger(__name__)

try:
    import pyautogui as pyautogui  # noqa: PLC0414  (re-exported for patching)
except Exception:  # pragma: no cover
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
        "l",  # Ctrl+L → laser pointer (PowerPoint slideshow)
        "m",

        "n",
        "p",
        "s",
        "w",
        "y",
        # Presentation zoom and modifier keys
        "equal",  # Cmd/Ctrl+= to zoom in (pyautogui key name for the = key)
        "minus",  # Cmd/Ctrl+- to zoom out
        "command",  # macOS modifier for zoom and slideshow start/stop
        # Retained for completeness (workspace navigation removed but keys kept safe)
        "win",
        "super",
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

    def scroll(self, clicks: int) -> None:
        """Inject a native OS scroll event at the current cursor position.

        Positive clicks scroll up (view content higher in the document);
        negative clicks scroll down (advance through the document).  Clamped
        to [-20, 20] to prevent runaway scrolls from rapid-fire delivery.
        """
        if pyautogui is None:  # pragma: no cover
            LOGGER.warning("pyautogui is not installed — scroll skipped.")
            return
        pyautogui.scroll(max(-20, min(20, clicks)))
        LOGGER.debug("Scrolled: clicks=%d", clicks)

    def zoom(self, clicks: int, *, modifier: str) -> None:
        """Inject a modifier+scroll zoom event at the current cursor position.

        Uses Cmd+scroll on macOS and Ctrl+scroll on Windows / Linux to trigger
        in-application view zoom without activating OS-level zoom overlays.
        Positive clicks zoom in; negative zoom out.  Clamped to [-5, 5] to
        prevent runaway zoom from rapid-fire command delivery.
        """
        if pyautogui is None:  # pragma: no cover
            LOGGER.warning("pyautogui is not installed — zoom skipped.")
            return
        clamped = max(-5, min(5, clicks))
        pyautogui.keyDown(modifier)
        try:
            pyautogui.scroll(clamped)
        finally:
            # Always release the modifier — prevents a stuck Ctrl/Cmd key if
            # pyautogui.scroll() raises (e.g. display server unavailable).
            pyautogui.keyUp(modifier)
        LOGGER.debug("Zoomed: modifier=%s clicks=%d", modifier, clamped)

    def move_mouse(self, x: int, y: int) -> None:
        """Move the OS cursor to absolute screen coordinates (*x*, *y*).

        Coordinates are clamped to the screen boundary reported by pyautogui.
        Duration is 0 so that the cursor tracks the hand in real-time without
        visible lag.
        """
        if pyautogui is None:  # pragma: no cover
            LOGGER.warning("pyautogui is not installed — mouse move skipped.")
            return
        pyautogui.moveTo(x, y, duration=0.0)
        LOGGER.debug("Mouse moved to (%d, %d)", x, y)

    def screen_size(self) -> tuple[int, int]:
        """Return the primary monitor dimensions as *(width, height)* in pixels.

        Falls back to a safe 1920×1080 default when pyautogui is unavailable
        so that coordinate calculations in headless environments do not crash.
        """
        if pyautogui is None:  # pragma: no cover
            return (1920, 1080)
        w, h = pyautogui.size()
        return int(w), int(h)
