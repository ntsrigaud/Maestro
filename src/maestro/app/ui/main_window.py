from __future__ import annotations

from PySide6.QtCore import QCoreApplication, QSignalBlocker, Qt
from PySide6.QtGui import QAction, QColor, QIcon, QPainter, QPen, QPixmap
from PySide6.QtWidgets import (
    QHBoxLayout,
    QLabel,
    QMenu,
    QSystemTrayIcon,
    QToolButton,
    QVBoxLayout,
    QWidget,
)

from maestro.app._paths import get_app_icon, get_config_dir
from maestro.app.ui.controller import RuntimeController
from maestro.app.ui.floating_preview import FloatingPreviewWindow
from maestro.domain.entities import RuntimeState


def _make_tray_icon(colour: str) -> QIcon:
    """Create a 16×16 solid-colour tray icon."""
    px = QPixmap(16, 16)
    px.fill(QColor(colour))
    return QIcon(px)


def _make_state_tray_icon(base_icon: QIcon | None, colour: str) -> QIcon:
    """Create a tray icon that preserves the app glyph with a state-colour accent."""
    if base_icon is None or base_icon.isNull():
        return _make_tray_icon(colour)

    size = 16
    composed = QPixmap(size, size)
    composed.fill(Qt.GlobalColor.transparent)

    painter = QPainter(composed)
    painter.setRenderHint(QPainter.RenderHint.Antialiasing, True)

    # Draw a coloured ring to communicate runtime state while keeping the
    # original app icon visible in the tray.
    pen = QPen(QColor(colour))
    pen.setWidth(2)
    painter.setPen(pen)
    painter.setBrush(Qt.BrushStyle.NoBrush)
    painter.drawEllipse(1, 1, size - 3, size - 3)

    inner = base_icon.pixmap(12, 12)
    if not inner.isNull():
        painter.drawPixmap((size - 12) // 2, (size - 12) // 2, inner)

    # Small state dot improves visibility on macOS menu bar where icons are tiny.
    painter.setPen(Qt.PenStyle.NoPen)
    painter.setBrush(QColor(colour))
    painter.drawEllipse(size - 6, size - 6, 4, 4)
    painter.end()

    return QIcon(composed)


class MainWindow(QWidget):
    """Cross-platform runtime manager with tray and taskbar-accessible controls."""

    def __init__(self, config_path: str | None = None) -> None:
        super().__init__()
        self._config_path = config_path or str(get_config_dir())
        self._preview_window = FloatingPreviewWindow()
        self._controller = RuntimeController(parent=self)

        self.setWindowTitle("Maestro")
        self.setMinimumSize(380, 180)

        # Inactive tray icon — the real app icon when available, grey dot otherwise.
        icon_path = get_app_icon()
        self._base_icon = QIcon(str(icon_path)) if icon_path else None
        self._tray_icons_cache: dict[str, QIcon] = {}
        self._inactive_icon = self._tray_icon_for_colour("#6B7280")

        self._status_label = QLabel("Ready", self)
        self._status_label.setWordWrap(True)
        self._status_label.setObjectName("statusLabel")

        self._controls_menu = QMenu(self)

        self._active_action = QAction("Active", self)
        self._active_action.setCheckable(True)
        self._active_action.toggled.connect(self._on_active_toggled)

        self._preview_action = QAction("Preview", self)
        self._preview_action.setCheckable(True)
        self._preview_action.toggled.connect(self._on_preview_toggled)

        self._controls_menu.addAction(self._active_action)
        self._controls_menu.addAction(self._preview_action)
        self._controls_menu.addSeparator()
        self._controls_menu.addAction("Quit", self.close)

        self._menu_button = QToolButton(self)
        self._menu_button.setText("Maestro")
        self._menu_button.setIcon(self._inactive_icon)
        self._menu_button.setToolButtonStyle(Qt.ToolButtonStyle.ToolButtonTextBesideIcon)
        self._menu_button.setPopupMode(QToolButton.ToolButtonPopupMode.InstantPopup)
        self._menu_button.setMenu(self._controls_menu)

        self._tray_available = QSystemTrayIcon.isSystemTrayAvailable()
        self._tray: QSystemTrayIcon | None = None
        if self._tray_available:
            self._tray = QSystemTrayIcon(self)
            self._tray.setIcon(self._inactive_icon)
            self._tray.setToolTip("Maestro: inactive")
            self._tray.setContextMenu(self._controls_menu)
            self._tray.show()

        header = QHBoxLayout()
        header.addWidget(self._menu_button, 0)
        header.addWidget(self._status_label, 1)

        hint = QLabel(
            "Use the menu button or tray icon to start/stop the runtime and toggle preview.",
            self,
        )
        hint.setWordWrap(True)
        hint.setObjectName("hintLabel")

        layout = QVBoxLayout(self)
        layout.setContentsMargins(16, 16, 16, 16)
        layout.setSpacing(12)
        layout.addLayout(header)
        layout.addWidget(hint)

        self.setWindowIcon(self._inactive_icon)
        self.show()

        self._controller.runtime_started.connect(self._on_runtime_started)
        self._controller.runtime_stopped.connect(self._on_runtime_stopped)
        self._controller.error_occurred.connect(self._on_error)
        self._controller.state_updated.connect(self._preview_window.on_state)
        self._controller.state_updated.connect(self._on_tray_update)

    def _on_active_toggled(self, active: bool) -> None:
        if active:
            if self._controller.is_running:
                return
            self._set_runtime_message("Maestro: starting")
            self._controller.start(
                self._config_path,
                show_window=True,
                minimal_preview=True,
            )
            return

        if self._controller.is_running:
            self._set_runtime_message("Maestro: stopping")
            self._controller.stop()
        self._sync_preview_visibility()

    def _on_preview_toggled(self, _enabled: bool) -> None:
        self._sync_preview_visibility()

    def _on_runtime_started(self) -> None:
        self._set_runtime_message("Maestro: active")
        self._preview_window.reset()
        self._sync_preview_visibility()

    def _on_runtime_stopped(self) -> None:
        self._set_runtime_message("Maestro: inactive")
        self._apply_state_icon(self._inactive_icon)
        self._preview_window.hide()
        with QSignalBlocker(self._active_action):
            self._active_action.setChecked(False)

    def _on_error(self, message: str) -> None:
        self._apply_state_icon(self._tray_icon_for_colour("#DC2626"))
        self._set_runtime_message("Maestro: error")
        self._preview_window.hide()
        with QSignalBlocker(self._active_action):
            self._active_action.setChecked(False)
        self._show_notification("Maestro", message, QSystemTrayIcon.MessageIcon.Critical, 2000)

    def _on_tray_update(self, state: RuntimeState) -> None:
        colour = self._tray_colour_for_state(state)
        self._apply_state_icon(self._tray_icon_for_colour(colour))
        self._status_label.setText(self._state_text_for_state(state))

        if state.info.get("accepted"):
            self._show_notification(
                "Gesture", str(state.last_label), QSystemTrayIcon.MessageIcon.Information, 800
            )

    def _tray_colour_for_state(self, state: RuntimeState) -> str:
        """Map runtime state to the same colour theme used by floating preview badge."""
        laser_active = bool(state.info.get("laser_active", False))
        pen_active = bool(state.info.get("pen_active", False))

        if laser_active:
            return "#2563EB"  # blue — laser pointer active
        if pen_active:
            return "#7C3AED"  # violet — annotation pen active
        if state.mode == "armed":
            return "#059669"  # green — armed
        return "#D97706"  # amber — cooldown

    def _tray_icon_for_colour(self, colour: str) -> QIcon:
        cached = self._tray_icons_cache.get(colour)
        if cached is not None:
            return cached
        icon = _make_state_tray_icon(self._base_icon, colour)
        self._tray_icons_cache[colour] = icon
        return icon

    def _apply_state_icon(self, icon: QIcon) -> None:
        self.setWindowIcon(icon)
        self._menu_button.setIcon(icon)
        if self._tray is not None:
            self._tray.setIcon(icon)

    def _set_runtime_message(self, text: str) -> None:
        self._status_label.setText(text)
        if self._tray is not None:
            self._tray.setToolTip(text)
        self.setToolTip(text)

    def _show_notification(
        self,
        title: str,
        message: str,
        icon: QSystemTrayIcon.MessageIcon,
        timeout_ms: int,
    ) -> None:
        if self._tray is not None:
            self._tray.showMessage(title, message, icon, timeout_ms)
        else:
            self._status_label.setText(message)

    def _state_text_for_state(self, state: RuntimeState) -> str:
        if bool(state.info.get("laser_active", False)):
            return "Laser pointer active"
        if bool(state.info.get("pen_active", False)):
            return "Annotation pen active"
        if state.mode == "armed":
            return (
                f"Armed: {state.last_label.replace('_', ' ').upper()}"
                if state.last_label != "unknown"
                else "Armed"
            )
        else:  # cooldown
            return (
                f"Cooldown: {state.last_label.replace('_', ' ').upper()}"
                if state.last_label != "unknown"
                else "Cooldown"
            )

    def _sync_preview_visibility(self) -> None:
        if self._preview_action.isChecked() and self._controller.is_running:
            self._preview_window.show_preview()
            return
        self._preview_window.hide()

    def closeEvent(self, event) -> None:  # type: ignore[override]
        self._controller.stop()
        self._controller.wait(timeout_ms=3_000)
        self._preview_window.close()
        if self._tray is not None:
            self._tray.hide()
        super().closeEvent(event)
        QCoreApplication.quit()
