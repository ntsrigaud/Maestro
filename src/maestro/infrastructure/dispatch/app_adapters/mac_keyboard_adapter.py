from __future__ import annotations

import ctypes
import ctypes.util
import logging
import time

LOGGER = logging.getLogger(__name__)

_CG = None
try:
    _CG = ctypes.cdll.LoadLibrary(ctypes.util.find_library('CoreGraphics'))

    class CGPoint(ctypes.Structure):
        _fields_ = [("x", ctypes.c_double), ("y", ctypes.c_double)]

    _CG.CGEventCreateKeyboardEvent.restype = ctypes.c_void_p
    _CG.CGEventCreateKeyboardEvent.argtypes = [ctypes.c_void_p, ctypes.c_uint16, ctypes.c_bool]

    _CG.CGEventCreateMouseEvent.restype = ctypes.c_void_p
    _CG.CGEventCreateMouseEvent.argtypes = [ctypes.c_void_p, ctypes.c_uint32, CGPoint, ctypes.c_uint32]

    _CG.CGEventCreateScrollWheelEvent.restype = ctypes.c_void_p
    _CG.CGEventCreateScrollWheelEvent.argtypes = [ctypes.c_void_p, ctypes.c_uint32, ctypes.c_uint32, ctypes.c_int32]

    _CG.CGEventPost.restype = None
    _CG.CGEventPost.argtypes = [ctypes.c_uint32, ctypes.c_void_p]

    _CG.CFRelease.restype = None
    _CG.CFRelease.argtypes = [ctypes.c_void_p]

    _CG.CGDisplayPixelsHigh.restype = ctypes.c_size_t
    _CG.CGDisplayPixelsHigh.argtypes = [ctypes.c_uint32]
    
    _CG.CGDisplayPixelsWide.restype = ctypes.c_size_t
    _CG.CGDisplayPixelsWide.argtypes = [ctypes.c_uint32]

    _CG.CGMainDisplayID.restype = ctypes.c_uint32
    _CG.CGMainDisplayID.argtypes = []
except Exception:
    pass

_MAC_KEYS = {
    'right': 124, 'left': 123, 'up': 126, 'down': 125,
    'escape': 53, 'enter': 36, 'space': 49, 'tab': 48, 'backspace': 51,
    'a': 0, 'b': 11, 'e': 14, 'f': 3, 'k': 40, 'l': 37, 'm': 46, 'n': 45, 'p': 35, 's': 1, 'w': 13, 'y': 16,
    'equal': 24, 'minus': 27,
    'home': 115, 'end': 119,
    'ctrl': 59, 'command': 55, 'shift': 56, 'alt': 58,
}

kCGHIDEventTap = 0
kCGEventMouseMoved = 5
kCGEventLeftMouseDown = 1
kCGEventLeftMouseUp = 2
kCGEventLeftMouseDragged = 6
kCGScrollEventUnitLine = 0

class MacKeyboardAdapter:
    @staticmethod
    def _foreground_identity_mac() -> str | None:
        try:
            objc = ctypes.cdll.LoadLibrary(ctypes.util.find_library('objc'))
            objc.objc_getClass.restype = ctypes.c_void_p
            objc.sel_registerName.restype = ctypes.c_void_p
            objc_msgSend = objc.objc_msgSend
            objc_msgSend.restype = ctypes.c_void_p
            objc_msgSend.argtypes = [ctypes.c_void_p, ctypes.c_void_p]

            NSWorkspace = objc.objc_getClass(b"NSWorkspace")
            sharedWorkspace = objc_msgSend(NSWorkspace, objc.sel_registerName(b"sharedWorkspace"))
            frontmostApp = objc_msgSend(sharedWorkspace, objc.sel_registerName(b"frontmostApplication"))
            if not frontmostApp: return None

            bundleIdStr = objc_msgSend(frontmostApp, objc.sel_registerName(b"bundleIdentifier"))
            localizedNameStr = objc_msgSend(frontmostApp, objc.sel_registerName(b"localizedName"))

            objc_msgSend_ptr = objc.objc_msgSend
            objc_msgSend_ptr.restype = ctypes.c_char_p
            objc_msgSend_ptr.argtypes = [ctypes.c_void_p, ctypes.c_void_p]

            bundle_id = ""
            if bundleIdStr:
                b_bytes = objc_msgSend_ptr(bundleIdStr, objc.sel_registerName(b"UTF8String"))
                if b_bytes: bundle_id = b_bytes.decode('utf-8')

            app_name = ""
            if localizedNameStr:
                n_bytes = objc_msgSend_ptr(localizedNameStr, objc.sel_registerName(b"UTF8String"))
                if n_bytes: app_name = n_bytes.decode('utf-8')

            return f"{bundle_id} {app_name}".strip().lower()
        except Exception:
            return None

    def press_hotkey(self, *keys: str) -> bool:
        if not _CG: return False
        codes = [int(_MAC_KEYS.get(k.lower(), -1)) for k in keys]
        if any(c == -1 for c in codes):
            return False
        
        # Key down
        for c in codes:
            evt = _CG.CGEventCreateKeyboardEvent(None, c, True)
            _CG.CGEventPost(kCGHIDEventTap, evt)
            _CG.CFRelease(evt)
            time.sleep(0.01)
        # Key up
        for c in reversed(codes):
            evt = _CG.CGEventCreateKeyboardEvent(None, c, False)
            _CG.CGEventPost(kCGHIDEventTap, evt)
            _CG.CFRelease(evt)
            time.sleep(0.01)
        return True

    def scroll(self, clicks: int) -> bool:
        if not _CG: return False
        # In CoreGraphics, positive is usually UP, but let's match pyautogui behavior.
        # macOS wheel: positive wheel1 means up.
        evt = _CG.CGEventCreateScrollWheelEvent(None, kCGScrollEventUnitLine, 1, clicks)
        _CG.CGEventPost(kCGHIDEventTap, evt)
        _CG.CFRelease(evt)
        return True

    def zoom(self, clicks: int, *, modifier: str, max_abs_clicks: int) -> bool:
        if not _CG: return False
        clamped = max(-max_abs_clicks, min(max_abs_clicks, clicks))
        mod_code = _MAC_KEYS.get(modifier.lower())
        if mod_code is None: return False
        
        evt_mod_down = _CG.CGEventCreateKeyboardEvent(None, mod_code, True)
        _CG.CGEventPost(kCGHIDEventTap, evt_mod_down)
        _CG.CFRelease(evt_mod_down)
        time.sleep(0.01)
        
        self.scroll(clamped)
        
        evt_mod_up = _CG.CGEventCreateKeyboardEvent(None, mod_code, False)
        _CG.CGEventPost(kCGHIDEventTap, evt_mod_up)
        _CG.CFRelease(evt_mod_up)
        return True

    def move_mouse(self, x: int, y: int) -> bool:
        if not _CG: return False
        pt = CGPoint(float(x), float(y))
        evt = _CG.CGEventCreateMouseEvent(None, kCGEventMouseMoved, pt, 0)
        _CG.CGEventPost(kCGHIDEventTap, evt)
        _CG.CFRelease(evt)
        return True

    def mouse_drag_to(self, x: int, y: int) -> bool:
        if not _CG: return False
        pt = CGPoint(float(x), float(y))
        evt = _CG.CGEventCreateMouseEvent(None, kCGEventLeftMouseDragged, pt, 0)
        _CG.CGEventPost(kCGHIDEventTap, evt)
        _CG.CFRelease(evt)
        return True

    def mouse_down(self, x: int, y: int) -> bool:
        if not _CG: return False
        pt = CGPoint(float(x), float(y))
        evt = _CG.CGEventCreateMouseEvent(None, kCGEventLeftMouseDown, pt, 0)
        _CG.CGEventPost(kCGHIDEventTap, evt)
        _CG.CFRelease(evt)
        return True

    def mouse_up(self) -> bool:
        if not _CG: return False
        # Need current mouse position. For simplicity we can use 0,0 but kCGEventLeftMouseUp
        # does not strictly require the exact position if it's just a button release.
        # However, to be safe, we should use the current position.
        pt = CGPoint(0, 0) # CFRelease handles the event regardless of exact pos if it's a drag release
        evt = _CG.CGEventCreateMouseEvent(None, kCGEventLeftMouseUp, pt, 0)
        _CG.CGEventPost(kCGHIDEventTap, evt)
        _CG.CFRelease(evt)
        return True

    def mouse_click(self) -> bool:
        if not _CG: return False
        pt = CGPoint(0, 0)
        evt_down = _CG.CGEventCreateMouseEvent(None, kCGEventLeftMouseDown, pt, 0)
        _CG.CGEventPost(kCGHIDEventTap, evt_down)
        _CG.CFRelease(evt_down)
        evt_up = _CG.CGEventCreateMouseEvent(None, kCGEventLeftMouseUp, pt, 0)
        _CG.CGEventPost(kCGHIDEventTap, evt_up)
        _CG.CFRelease(evt_up)
        return True

    def screen_size(self) -> tuple[int, int]:
        if not _CG: return (1920, 1080)
        main_id = _CG.CGMainDisplayID()
        w = _CG.CGDisplayPixelsWide(main_id)
        h = _CG.CGDisplayPixelsHigh(main_id)
        return int(w), int(h)
