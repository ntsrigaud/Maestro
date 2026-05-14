from __future__ import annotations

import sys

from PySide6.QtCore import QPoint, Qt, Slot
from PySide6.QtGui import QGuiApplication, QMouseEvent
from PySide6.QtWidgets import QVBoxLayout, QWidget

from maestro.app.ui.preview_widget import PreviewWidget
from maestro.domain.entities import RuntimeState

try:
    import AppKit
except ImportError:  # pragma: no cover - only unavailable off macOS/PyObjC envs
    AppKit = None  # type: ignore[assignment]

_CORNER_MARGIN = 24
_MAX_PREVIEW_WIDTH = 280


class FloatingPreviewWindow(QWidget):
    """Always-on-top draggable preview overlay used by the tray UI."""

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

        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(0)
        layout.addWidget(self._preview)

        self.hide()

    def show_preview(self) -> None:
        if not self._default_positioned:
            self._move_to_default_corner()
            self._default_positioned = True
        self.show()
        self.raise_()
        self._apply_native_overlay_level()

    @Slot(object)
    def on_state(self, state: RuntimeState) -> None:
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
        self._apply_native_overlay_level()

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

    def _move_to_default_corner(self) -> None:
        screen = QGuiApplication.primaryScreen()
        if screen is None:
            self.move(_CORNER_MARGIN, _CORNER_MARGIN)
            return

        geometry = screen.availableGeometry()
        x = geometry.right() - self.width() - _CORNER_MARGIN
        y = geometry.top() + _CORNER_MARGIN
        self.move(max(geometry.left() + _CORNER_MARGIN, x), y)

    def _apply_native_overlay_level(self) -> None:
        if self._native_level_applied:
            return
        if sys.platform != "darwin" or AppKit is None:
            return

        ns_window = self._find_native_window()
        if ns_window is None:
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

    def _find_native_window(self):
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