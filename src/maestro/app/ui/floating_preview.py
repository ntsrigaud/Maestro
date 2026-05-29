from __future__ import annotations

import sys
from typing import Any

from PySide6.QtCore import QPoint, Qt, QTimer, Slot
from PySide6.QtGui import QGuiApplication, QMouseEvent
from PySide6.QtWidgets import QLabel, QVBoxLayout, QWidget

from maestro.app.ui.preview_widget import PreviewWidget
from maestro.domain.entities import RuntimeState

try:
    import AppKit
except ImportError:  # pragma: no cover - only unavailable off macOS/PyObjC envs
    AppKit = None  # type: ignore[assignment]

_CORNER_MARGIN = 24
_MAX_PREVIEW_WIDTH = 280
_STATUS_BADGE_MARGIN = 8

# Keep badge theme aligned with overlay border state priority.
_BADGE_ACQUIRING = "rgba(217, 119, 6, 220)"
_BADGE_ARMED = "rgba(5, 150, 105, 215)"
_BADGE_LASER = "rgba(37, 99, 235, 225)"
_BADGE_PEN = "rgba(124, 58, 237, 225)"

# Delay (ms) after show() before resolving the NSWindow handle.
# Qt requires a brief moment to register the OS-level window with NSApp.
_NATIVE_LEVEL_DELAY_MS = 300


class FloatingPreviewWindow(QWidget):
    """Always-on-top draggable preview overlay used by the tray UI.

    Thread safety
    -------------
    All methods must be called from the main Qt thread.  State delivery from
    the worker thread arrives via Qt's queued-signal mechanism and is already
    on the main thread when ``on_state`` is invoked.

    Performance notes
    -----------------
    * Frame rendering (QImage construction) is skipped when the window is
      hidden, saving ~0.5 ms per frame at 10 Hz.
    * The macOS NSWindow level is applied exactly once per show via a deferred
      timer rather than being retried on every frame, eliminating repeated
      NSApp window traversal.
    * Native level is re-armed on each ``show_preview`` call so it survives
      hide/show cycles (e.g. user toggles the Preview tray action).
    """

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self._drag_offset: QPoint | None = None
        self._default_positioned = False
        self._user_moved = False
        self._has_frame = False
        self._native_level_applied = False

        self.setWindowTitle("Maestro Preview")
        self.setWindowFlags(
            Qt.WindowType.Window
            | Qt.WindowType.FramelessWindowHint
            | Qt.WindowType.WindowStaysOnTopHint
        )
        self.setAttribute(Qt.WidgetAttribute.WA_ShowWithoutActivating, True)
        self.setAttribute(Qt.WidgetAttribute.WA_TranslucentBackground, True)
        self.setWindowFlag(Qt.WindowType.WindowDoesNotAcceptFocus, True)
        self.setStyleSheet("background: transparent; border: none;")

        self._preview = PreviewWidget(self, max_display_width=_MAX_PREVIEW_WIDTH)
        self._preview.setStyleSheet("background: transparent; border: none;")

        self._state_badge = QLabel("STATE: STARTING", self)
        self._state_badge.setAttribute(Qt.WidgetAttribute.WA_TransparentForMouseEvents, True)
        self._state_badge.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self._state_badge.setStyleSheet(
            ""
            "QLabel {"
            " color: #F9FAFB;"
            " background-color: rgba(17, 24, 39, 190);"
            " border: 1px solid rgba(249, 250, 251, 120);"
            " border-radius: 10px;"
            " padding: 4px 8px;"
            " font-size: 11px;"
            " font-weight: 700;"
            " letter-spacing: 0.4px;"
            "}"
        )
        self._state_badge.hide()

        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(0)
        layout.addWidget(self._preview)

        self.hide()

    def show_preview(self) -> None:
        """Show the overlay and schedule macOS native window-level application."""
        if not self._default_positioned:
            self._move_to_default_corner()
            self._default_positioned = True
        self.show()
        self.raise_()
        # Re-arm native level on every show so it is correctly reapplied after
        # hide/show cycles.  The timer fires after Qt has registered the OS
        # window handle with NSApp; calling it synchronously here would fail
        # because the NSWindow may not yet exist.
        if sys.platform == "darwin" and AppKit is not None:
            self._native_level_applied = False
            QTimer.singleShot(_NATIVE_LEVEL_DELAY_MS, self._apply_native_overlay_level)

    def reset(self) -> None:
        """Reset frame-tracking state when a new session begins.

        Releases the fixed-size constraint so the window can resize to the
        new session's video frame dimensions on the first received frame.
        """
        self._has_frame = False
        self.setMinimumSize(0, 0)
        self.setMaximumSize(16_777_215, 16_777_215)

    @Slot(object)
    def on_state(self, state: RuntimeState) -> None:
        """Receive a frame from the worker thread (via queued signal)."""
        # Skip expensive QImage construction while hidden — saves ~0.5 ms/frame.
        if not self.isVisible():
            return
        self._update_state_badge(state)
        first_frame = not self._has_frame and state.preview_frame is not None
        if first_frame:
            self._has_frame = True
        self._preview.on_state(state)
        if first_frame:
            # Lock the window exactly to the video frame size — no padding.
            self.adjustSize()
            self.setFixedSize(self.sizeHint())
            if not self._user_moved:
                self._move_to_default_corner()
            self._reposition_state_badge()

    def mousePressEvent(self, event: QMouseEvent) -> None:  # type: ignore[override]
        if event.button() == Qt.MouseButton.LeftButton:
            self._drag_offset = event.globalPosition().toPoint() - self.frameGeometry().topLeft()
            event.accept()
            return
        super().mousePressEvent(event)

    def mouseMoveEvent(self, event: QMouseEvent) -> None:  # type: ignore[override]
        if self._drag_offset is not None and event.buttons() & Qt.MouseButton.LeftButton:
            self._user_moved = True
            self.move(event.globalPosition().toPoint() - self._drag_offset)
            event.accept()
            return
        super().mouseMoveEvent(event)

    def mouseReleaseEvent(self, event: QMouseEvent) -> None:  # type: ignore[override]
        if event.button() == Qt.MouseButton.LeftButton:
            self._drag_offset = None
            event.accept()
            return
        super().mouseReleaseEvent(event)

    def mouseDoubleClickEvent(self, event: QMouseEvent) -> None:  # type: ignore[override]
        """Snap back to the default screen corner on double-click."""
        if event.button() == Qt.MouseButton.LeftButton:
            self._user_moved = False
            self._move_to_default_corner()
            event.accept()
            return
        super().mouseDoubleClickEvent(event)

    def _move_to_default_corner(self) -> None:
        screen = QGuiApplication.primaryScreen()
        if screen is None:
            self.move(_CORNER_MARGIN, _CORNER_MARGIN)
            return

        geometry = screen.availableGeometry()
        x = geometry.right() - self.width() - _CORNER_MARGIN
        y = geometry.top() + _CORNER_MARGIN
        self.move(max(geometry.left() + _CORNER_MARGIN, x), y)

    def resizeEvent(self, event) -> None:  # type: ignore[override]
        super().resizeEvent(event)
        self._reposition_state_badge()

    def _reposition_state_badge(self) -> None:
        if not self._state_badge.isVisible():
            return
        self._state_badge.adjustSize()
        badge_w = self._state_badge.width()
        x = max(_STATUS_BADGE_MARGIN, self.width() - badge_w - _STATUS_BADGE_MARGIN)
        y = _STATUS_BADGE_MARGIN
        self._state_badge.move(x, y)

    def _update_state_badge(self, state: RuntimeState) -> None:
        laser_active = bool(state.info.get("laser_active", False))
        pen_active = bool(state.info.get("pen_active", False))
        label = state.last_label or "unknown"
        label_display = label.replace("_", " ").upper() if label != "unknown" else ""
        color = self._badge_theme_color(state)

        if laser_active:
            text = "LASER POINTER ACTIVE"
        elif pen_active:
            text = "ANNOTATION PEN ACTIVE"
        elif state.mode == "cooldown":
            suffix = f": {label_display}" if label_display else ""
            text = f"DISPATCHED{suffix}"
        else:  # armed
            text = f"ARMED | {label_display}" if label_display else "ARMED | READY"

        self._state_badge.setText(text)
        self._state_badge.setStyleSheet(
            ""
            "QLabel {"
            " color: #F9FAFB;"
            f" background-color: {color};"
            " border: 1px solid rgba(249, 250, 251, 120);"
            " border-radius: 10px;"
            " padding: 4px 8px;"
            " font-size: 11px;"
            " font-weight: 700;"
            " letter-spacing: 0.4px;"
            "}"
        )
        if not self._state_badge.isVisible():
            self._state_badge.show()
        self._state_badge.raise_()
        self._reposition_state_badge()

    def _badge_theme_color(self, state: RuntimeState) -> str:
        """Return a badge background color matching overlay border state priority."""
        if state.info.get("laser_active"):
            return _BADGE_LASER
        if state.info.get("pen_active"):
            return _BADGE_PEN
        if state.mode == "armed":
            return _BADGE_ARMED
        return _BADGE_ACQUIRING

    def _apply_native_overlay_level(self) -> None:
        """Apply macOS NSWindow overlay level once after the window is shown.

        Invoked via ``QTimer.singleShot`` from ``show_preview``; never called
        directly on every frame.  Retries once if the NSWindow is not yet
        registered with NSApp (can happen if Qt is still compositing the window).
        """
        if self._native_level_applied:
            return
        if sys.platform != "darwin" or AppKit is None:
            return

        ns_window = self._find_native_window()
        if ns_window is None:
            # NSWindow not yet visible; schedule a single retry.
            QTimer.singleShot(_NATIVE_LEVEL_DELAY_MS, self._apply_native_overlay_level)
            return

        behavior = (
            AppKit.NSWindowCollectionBehaviorCanJoinAllSpaces
            | AppKit.NSWindowCollectionBehaviorFullScreenAuxiliary
            | AppKit.NSWindowCollectionBehaviorStationary
        )
        ns_window.setCollectionBehavior_(behavior)
        ns_window.setLevel_(AppKit.NSStatusWindowLevel)
        ns_window.setHidesOnDeactivate_(False)
        ns_window.setExcludedFromWindowsMenu_(True)
        ns_window.orderFrontRegardless()
        self._native_level_applied = True

    def _find_native_window(self) -> Any:
        ns_app = getattr(AppKit, "NSApp", None)
        if ns_app is None:
            try:
                ns_app = AppKit.NSApplication.sharedApplication()
            except Exception:
                return None
        if ns_app is None:
            return None

        try:
            windows = ns_app.windows()
        except Exception:
            return None

        for ns_window in windows:
            try:
                if str(ns_window.title()) == self.windowTitle():
                    return ns_window
            except Exception:
                continue
        return None
