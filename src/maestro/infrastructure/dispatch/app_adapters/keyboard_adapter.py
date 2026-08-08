from __future__ import annotations

import ctypes
import logging
import os
import shutil
import subprocess
import sys
import typing

LOGGER = logging.getLogger(__name__)

try:
    import pyautogui as pyautogui  # noqa: PLC0414  (re-exported for patching)

    # Disable PyAutoGUI's built-in fail-safe (raises FailSafeException when the
    # cursor reaches a screen corner).  The fail-safe is designed as a developer
    # escape hatch for automated scripts; in Maestro the cursor is driven by
    # intentional hand gestures, and a corner position is a legitimate pointer
    # destination during presentations.  Leaving it enabled causes the runtime
    # worker thread to crash whenever the user points at a screen corner.
    pyautogui.FAILSAFE = False
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

# Win32 mouse_event() dwFlags constants used in mouse_down() and
# mouse_drag_to() on Windows.
#
# pyautogui's _moveTo() uses SetCursorPos() which moves the OS cursor
# silently WITHOUT generating WM_MOUSEMOVE.  PowerPoint's annotation pen only
# draws ink when it receives WM_MOUSEMOVE (with MK_LBUTTON set in wParam).
# Calling mouse_event(MOVE|ABSOLUTE, ...) directly generates the required
# WM_MOUSEMOVE so PowerPoint can track the pen stroke.
_W32_MOUSEEVENTF_MOVE: int = 0x0001  # synthesise a WM_MOUSEMOVE
_W32_MOUSEEVENTF_LEFTDOWN: int = 0x0002  # WM_LBUTTONDOWN
_W32_MOUSEEVENTF_ABSOLUTE: int = 0x8000  # dx/dy are normalised absolute (0-65535)

# Scope restricted to Microsoft PowerPoint only.
# Sending keystrokes to other presentation apps risks triggering unintended
# OS-level shortcuts (e.g. 'E' in a macOS text field opens the emoji picker).
# Override at runtime via the MAESTRO_PRESENTATION_SCOPE_ALLOWLIST env var.
_DEFAULT_SCOPE_TOKENS: tuple[str, ...] = (
    "microsoft powerpoint",  # macOS localised app name / Windows window title
    "powerpnt",  # Windows process name (POWERPNT.EXE)
    "com.microsoft.powerpoint",  # macOS bundle ID
)


def _scope_tokens() -> tuple[str, ...]:
    raw = os.getenv("MAESTRO_PRESENTATION_SCOPE_ALLOWLIST", "").strip()
    if not raw:
        return _DEFAULT_SCOPE_TOKENS
    items = tuple(token.strip().lower() for token in raw.split(",") if token.strip())
    return items if items else _DEFAULT_SCOPE_TOKENS


def _foreground_identity_mac() -> str | None:
    try:
        from maestro.infrastructure.dispatch.app_adapters.mac_keyboard_adapter import (
            MacKeyboardAdapter,
        )

        return MacKeyboardAdapter._foreground_identity_mac()
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


class OSKeyboardBackend(typing.Protocol):
    """Protocol for OS-specific keyboard and mouse execution backends."""

    def press_hotkey(self, *keys: str) -> bool: ...
    def scroll(self, clicks: int) -> bool: ...
    def zoom(self, clicks: int, *, modifier: str, max_abs_clicks: int) -> bool: ...
    def move_mouse(self, x: int, y: int) -> bool: ...
    def mouse_drag_to(self, x: int, y: int) -> bool: ...
    def mouse_down(self, x: int, y: int) -> bool: ...
    def mouse_up(self) -> bool: ...
    def mouse_click(self) -> bool: ...
    def screen_size(self) -> tuple[int, int]: ...


class PyAutoGUIKeyboardAdapter:
    """Linux/Cross-platform backend using pyautogui exclusively."""

    def press_hotkey(self, *keys: str) -> bool:
        if pyautogui is None:
            return False
        pyautogui.hotkey(*[k.lower() for k in keys])
        return True

    def scroll(self, clicks: int) -> bool:
        if pyautogui is None:
            return False
        pyautogui.scroll(clicks)
        return True

    def zoom(self, clicks: int, *, modifier: str, max_abs_clicks: int) -> bool:
        if pyautogui is None:
            return False
        clamped = max(-max_abs_clicks, min(max_abs_clicks, clicks))
        pyautogui.keyDown(modifier)
        try:
            pyautogui.scroll(clamped)
        finally:
            pyautogui.keyUp(modifier)
        return True

    def move_mouse(self, x: int, y: int) -> bool:
        if pyautogui is None:
            return False
        pyautogui.moveTo(x, y, duration=0.0)
        return True

    def mouse_drag_to(self, x: int, y: int) -> bool:
        if pyautogui is None:
            return False
        pyautogui.dragTo(x, y, duration=0.0, button="left", mouseDownUp=False, _pause=False)
        return True

    def mouse_down(self, x: int, y: int) -> bool:
        if pyautogui is None:
            return False
        pyautogui.mouseDown(x, y, button="left")
        return True

    def mouse_up(self) -> bool:
        if pyautogui is None:
            return False
        pyautogui.mouseUp(button="left")
        return True

    def mouse_click(self) -> bool:
        if pyautogui is None:
            return False
        pyautogui.click()
        return True

    def screen_size(self) -> tuple[int, int]:
        if pyautogui is None:
            return (1920, 1080)
        w, h = pyautogui.size()
        return int(w), int(h)


class WindowsKeyboardAdapter(PyAutoGUIKeyboardAdapter):
    """Windows-specific backend using ctypes for mouse movement."""

    def mouse_drag_to(self, x: int, y: int) -> bool:
        if pyautogui is None:
            return False
        user32 = ctypes.windll.user32  # type: ignore[attr-defined]
        sw = user32.GetSystemMetrics(0)
        sh = user32.GetSystemMetrics(1)
        user32.mouse_event(
            _W32_MOUSEEVENTF_MOVE | _W32_MOUSEEVENTF_ABSOLUTE,
            65536 * x // sw + 1,
            65536 * y // sh + 1,
            0,
            0,
        )
        return True

    def mouse_down(self, x: int, y: int) -> bool:
        if pyautogui is None:
            return False
        user32 = ctypes.windll.user32  # type: ignore[attr-defined]
        sw = user32.GetSystemMetrics(0)
        sh = user32.GetSystemMetrics(1)
        dx = 65536 * x // sw + 1
        dy = 65536 * y // sh + 1
        user32.mouse_event(_W32_MOUSEEVENTF_MOVE | _W32_MOUSEEVENTF_ABSOLUTE, dx, dy, 0, 0)
        user32.mouse_event(_W32_MOUSEEVENTF_LEFTDOWN, 0, 0, 0, 0)
        return True


class KeyboardAdapter:
    """Low-level key delivery service facade.

    Validates every key name against :data:`_SAFE_KEY_NAMES` before invoking
    the OS-specific backend. This is the only location in the codebase where
    simulated keystrokes are sent to the OS; all higher layers operate on
    logical command IDs, never on raw key strings.
    """

    def __init__(self, backend: OSKeyboardBackend | None = None):
        if backend is not None:
            self._backend = backend
        elif sys.platform == "darwin":
            from maestro.infrastructure.dispatch.app_adapters.mac_keyboard_adapter import (
                MacKeyboardAdapter,
            )

            self._backend = MacKeyboardAdapter()
        elif sys.platform == "win32":
            self._backend = WindowsKeyboardAdapter()
        else:
            self._backend = PyAutoGUIKeyboardAdapter()

    def is_target_app_focused(self) -> bool:
        """Return True if a supported target application (e.g. PowerPoint) is currently focused."""
        return self._ensure_presentation_scope(action="focus_check", silent=True)

    def _ensure_presentation_scope(self, action: str, *, silent: bool = False) -> bool:
        identity = _foreground_identity()
        if identity is None:
            if not silent:
                LOGGER.warning(
                    "Scope check failed for action=%s: foreground application could not be "
                    "identified. Blocking dispatch to prevent accidental key injection.",
                    action,
                )
            return False

        tokens = _scope_tokens()
        if any(token in identity for token in tokens):
            return True

        if not silent:
            LOGGER.warning(
                "Blocked action outside presentation scope: action=%s foreground=%r",
                action,
                identity,
            )
        return False

    def press_hotkey(self, *keys: str) -> bool:
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

        result = self._backend.press_hotkey(*keys)
        if result:
            LOGGER.debug("Pressed hotkey: %s", "+".join(k.lower() for k in keys))
        elif pyautogui is None and not isinstance(
            self._backend, WindowsKeyboardAdapter
        ):  # pragma: no cover
            LOGGER.warning("pyautogui is not installed — keyboard dispatch skipped.")
        return result

    def scroll(self, clicks: int) -> bool:
        if not self._ensure_presentation_scope(action=f"scroll:{clicks}"):
            return False

        clamped = max(-20, min(20, clicks))
        result = self._backend.scroll(clamped)
        if result:
            LOGGER.debug("Scrolled: clicks=%d", clamped)
        return result

    def zoom(self, clicks: int, *, modifier: str, max_abs_clicks: int) -> bool:
        if not self._ensure_presentation_scope(action=f"zoom:{modifier}:{clicks}"):
            return False

        if max_abs_clicks < 1:
            raise ValueError("max_abs_clicks must be >= 1")

        result = self._backend.zoom(clicks, modifier=modifier, max_abs_clicks=max_abs_clicks)
        if result:
            LOGGER.debug("Zoomed: modifier=%s clicks=%d", modifier, clicks)
        return result

    def move_mouse(self, x: int, y: int) -> bool:
        if not self._ensure_presentation_scope(action=f"move_mouse:{x}:{y}"):
            return False

        result = self._backend.move_mouse(x, y)
        if result:
            LOGGER.debug("Mouse moved to (%d, %d)", x, y)
        return result

    def mouse_drag_to(self, x: int, y: int) -> bool:
        if not self._ensure_presentation_scope(action=f"mouse_drag_to:{x}:{y}"):
            return False

        result = self._backend.mouse_drag_to(x, y)
        if result:
            LOGGER.debug("Mouse dragged to (%d, %d)", x, y)
        return result

    def mouse_down(self, x: int, y: int) -> bool:
        if not self._ensure_presentation_scope(action=f"mouse_down:{x}:{y}"):
            return False

        result = self._backend.mouse_down(x, y)
        if result:
            LOGGER.debug("Mouse down at (%d, %d)", x, y)
        return result

    def mouse_up(self) -> bool:
        if not self._ensure_presentation_scope(action="mouse_up"):
            return False

        result = self._backend.mouse_up()
        if result:
            LOGGER.debug("Mouse up")
        return result

    def mouse_click(self) -> bool:
        if not self._ensure_presentation_scope(action="mouse_click"):
            return False

        result = self._backend.mouse_click()
        if result:
            LOGGER.debug("Mouse clicked")
        return result

    def screen_size(self) -> tuple[int, int]:
        return self._backend.screen_size()
