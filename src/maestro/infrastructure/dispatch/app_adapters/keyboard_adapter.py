from __future__ import annotations

import ctypes
import logging
import os
import shutil
import subprocess
import sys

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
        "b",  # B → toggle black screen (PowerPoint/Keynote/Slides/Impress slideshow)
        "e",  # E → erase all annotations (PowerPoint slideshow)
        "f",
        "k",
        "l",  # Ctrl+L (Windows) / Cmd+L (macOS) → laser pointer (PowerPoint slideshow)
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
        # Slide navigation jump keys
        "home",  # Home → go to first slide (universal)
        "end",  # End  → go to last slide  (universal)
        # Retained for completeness (workspace navigation removed but keys kept safe)
        "win",
        "super",
    }
)

_DEFAULT_SCOPE_TOKENS: tuple[str, ...] = (
    "microsoft powerpoint",
    "powerpnt",
    "keynote",
    "libreoffice",
    "soffice",
    "impress",
    "wps presentation",
)


def _scope_tokens() -> tuple[str, ...]:
    raw = os.getenv("MAESTRO_PRESENTATION_SCOPE_ALLOWLIST", "").strip()
    if not raw:
        return _DEFAULT_SCOPE_TOKENS
    items = tuple(token.strip().lower() for token in raw.split(",") if token.strip())
    return items if items else _DEFAULT_SCOPE_TOKENS


def _foreground_identity_mac() -> str | None:
    try:
        import AppKit  # type: ignore

        ws = AppKit.NSWorkspace.sharedWorkspace()
        app = ws.frontmostApplication()
        if app is None:
            return None
        bundle_id = str(app.bundleIdentifier() or "")
        app_name = str(app.localizedName() or "")
        return f"{bundle_id} {app_name}".strip().lower()
    except Exception:
        return None


def _foreground_identity_windows() -> str | None:
    try:
        user32 = ctypes.windll.user32  # type: ignore[attr-defined]
        kernel32 = ctypes.windll.kernel32  # type: ignore[attr-defined]

        hwnd = user32.GetForegroundWindow()
        if not hwnd:
            return None

        length = user32.GetWindowTextLengthW(hwnd)
        title_buf = ctypes.create_unicode_buffer(length + 1)
        user32.GetWindowTextW(hwnd, title_buf, length + 1)
        title = title_buf.value or ""

        pid = ctypes.c_ulong(0)
        user32.GetWindowThreadProcessId(hwnd, ctypes.byref(pid))
        if not pid.value:
            return title.lower() if title else None

        PROCESS_QUERY_LIMITED_INFORMATION = 0x1000
        handle = kernel32.OpenProcess(PROCESS_QUERY_LIMITED_INFORMATION, False, pid.value)
        if not handle:
            return title.lower() if title else None

        try:
            buf_len = ctypes.c_ulong(260)
            exe_buf = ctypes.create_unicode_buffer(260)
            if kernel32.QueryFullProcessImageNameW(handle, 0, exe_buf, ctypes.byref(buf_len)):
                exe_path = exe_buf.value or ""
            else:
                exe_path = ""
        finally:
            kernel32.CloseHandle(handle)

        return f"{exe_path} {title}".strip().lower()
    except Exception:
        return None


def _foreground_identity_linux() -> str | None:
    # Prefer xdotool (window id + pid), then map pid -> process command.
    if shutil.which("xdotool") is None:
        return None
    try:
        wid = subprocess.check_output(["xdotool", "getactivewindow"], text=True).strip()
        pid = subprocess.check_output(["xdotool", "getwindowpid", wid], text=True).strip()
        comm = subprocess.check_output(["ps", "-p", pid, "-o", "comm="], text=True).strip()
        title = subprocess.check_output(["xdotool", "getwindowname", wid], text=True).strip()
        return f"{comm} {title}".strip().lower()
    except Exception:
        return None


def _foreground_identity() -> str | None:
    if sys.platform == "darwin":
        return _foreground_identity_mac()
    if sys.platform == "win32":
        return _foreground_identity_windows()
    return _foreground_identity_linux()


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

    def _ensure_presentation_scope(self, action: str) -> bool:
        identity = _foreground_identity()
        if identity is None:
            LOGGER.debug(
                "Scope check unavailable for action=%s; allowing dispatch (best effort).",
                action,
            )
            return True

        tokens = _scope_tokens()
        if any(token in identity for token in tokens):
            return True

        LOGGER.warning(
            "Blocked action outside presentation scope: action=%s foreground=%r",
            action,
            identity,
        )
        return False

    def press_hotkey(self, *keys: str) -> bool:
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

        if not self._ensure_presentation_scope(action=f"hotkey:{'+'.join(keys)}"):
            return False

        if pyautogui is None:  # pragma: no cover
            LOGGER.warning(
                "pyautogui is not installed — keyboard dispatch skipped. "
                "Install with `pip install pyautogui` to enable OS-level key delivery."
            )
            return False

        pyautogui.hotkey(*[k.lower() for k in keys])
        LOGGER.debug("Pressed hotkey: %s", "+".join(k.lower() for k in keys))
        return True

    def scroll(self, clicks: int) -> bool:
        """Inject a native OS scroll event at the current cursor position.

        Positive clicks scroll up (view content higher in the document);
        negative clicks scroll down (advance through the document).  Clamped
        to [-20, 20] to prevent runaway scrolls from rapid-fire delivery.
        """
        if not self._ensure_presentation_scope(action=f"scroll:{clicks}"):
            return False

        if pyautogui is None:  # pragma: no cover
            LOGGER.warning("pyautogui is not installed — scroll skipped.")
            return False
        pyautogui.scroll(max(-20, min(20, clicks)))
        LOGGER.debug("Scrolled: clicks=%d", clicks)
        return True

    def zoom(self, clicks: int, *, modifier: str) -> bool:
        """Inject a modifier+scroll zoom event at the current cursor position.

        Uses Cmd+scroll on macOS and Ctrl+scroll on Windows / Linux to trigger
        in-application view zoom without activating OS-level zoom overlays.
        Positive clicks zoom in; negative zoom out.  Clamped to [-5, 5] to
        prevent runaway zoom from rapid-fire command delivery.
        """
        if not self._ensure_presentation_scope(action=f"zoom:{modifier}:{clicks}"):
            return False

        if pyautogui is None:  # pragma: no cover
            LOGGER.warning("pyautogui is not installed — zoom skipped.")
            return False
        clamped = max(-5, min(5, clicks))
        pyautogui.keyDown(modifier)
        try:
            pyautogui.scroll(clamped)
        finally:
            # Always release the modifier — prevents a stuck Ctrl/Cmd key if
            # pyautogui.scroll() raises (e.g. display server unavailable).
            pyautogui.keyUp(modifier)
        LOGGER.debug("Zoomed: modifier=%s clicks=%d", modifier, clamped)
        return True

    def move_mouse(self, x: int, y: int) -> bool:
        """Move the OS cursor to absolute screen coordinates (*x*, *y*).

        Coordinates are clamped to the screen boundary reported by pyautogui.
        Duration is 0 so that the cursor tracks the hand in real-time without
        visible lag.
        """
        if not self._ensure_presentation_scope(action=f"move_mouse:{x}:{y}"):
            return False

        if pyautogui is None:  # pragma: no cover
            LOGGER.warning("pyautogui is not installed — mouse move skipped.")
            return False
        pyautogui.moveTo(x, y, duration=0.0)
        LOGGER.debug("Mouse moved to (%d, %d)", x, y)
        return True

    def mouse_drag_to(self, x: int, y: int) -> bool:
        """Drag the mouse cursor to (*x*, *y*) while the primary button is held.

        On macOS, pyautogui uses ``kCGEventLeftMouseDragged`` for this call,
        which is the correct event type required by drawing applications such
        as PowerPoint.  A plain ``moveTo`` call (``kCGEventMouseMoved``) is
        silently ignored by PowerPoint's pen mode even when the button is
        already pressed.

        On Windows/Linux, a drag event is equivalent to a move while the button
        is held, so this delegates to the same underlying Quartz/Win32 drag
        path via ``pyautogui.dragTo(mouseDownUp=False)``.

        Must only be called after :meth:`mouse_down` and before
        :meth:`mouse_up`.
        """
        if not self._ensure_presentation_scope(action=f"mouse_drag_to:{x}:{y}"):
            return False

        if pyautogui is None:  # pragma: no cover
            LOGGER.warning("pyautogui is not installed — mouse_drag_to skipped.")
            return False
        pyautogui.dragTo(x, y, duration=0.0, button="left", mouseDownUp=False, _pause=False)
        LOGGER.debug("Mouse dragged to (%d, %d)", x, y)
        return True

    def mouse_down(self, x: int, y: int) -> bool:
        """Move the OS cursor to (*x*, *y*) and press the primary mouse button.

        Used by :class:`~maestro.pipeline.annotation_pen.AnnotationPenTracker`
        to begin a pen stroke: the cursor is positioned at the fingertip before
        the button is pressed so that PowerPoint registers the stroke origin at
        the correct location.
        """
        if not self._ensure_presentation_scope(action=f"mouse_down:{x}:{y}"):
            return False

        if pyautogui is None:  # pragma: no cover
            LOGGER.warning("pyautogui is not installed — mouse_down skipped.")
            return False
        pyautogui.mouseDown(x, y, button="left")
        LOGGER.debug("Mouse down at (%d, %d)", x, y)
        return True

    def mouse_up(self) -> bool:
        """Release the primary mouse button at the current cursor position.

        Used by :class:`~maestro.pipeline.annotation_pen.AnnotationPenTracker`
        to end a pen stroke before sending the revert-to-arrow hotkey.
        """
        if not self._ensure_presentation_scope(action="mouse_up"):
            return False

        if pyautogui is None:  # pragma: no cover
            LOGGER.warning("pyautogui is not installed — mouse_up skipped.")
            return False
        pyautogui.mouseUp(button="left")
        LOGGER.debug("Mouse up")
        return True

    def mouse_click(self) -> bool:
        """Perform a left-click at the current cursor position.

        Used to activate hyperlinks and trigger animations in presentation
        slides without moving the cursor.  The click is delivered at whatever
        position the OS cursor currently occupies, which is set by
        :meth:`move_mouse` (e.g. from the laser-pointer or annotation tracker).
        """
        if not self._ensure_presentation_scope(action="mouse_click"):
            return False

        if pyautogui is None:  # pragma: no cover
            LOGGER.warning("pyautogui is not installed — mouse_click skipped.")
            return False
        pyautogui.click()
        LOGGER.debug("Mouse clicked")
        return True

    def screen_size(self) -> tuple[int, int]:
        """Return the primary monitor dimensions as *(width, height)* in pixels.

        Falls back to a safe 1920×1080 default when pyautogui is unavailable
        so that coordinate calculations in headless environments do not crash.
        """
        if pyautogui is None:  # pragma: no cover
            return (1920, 1080)
        w, h = pyautogui.size()
        return int(w), int(h)
