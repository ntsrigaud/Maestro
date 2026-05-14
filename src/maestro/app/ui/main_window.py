from __future__ import annotations

from PySide6.QtCore import QSignalBlocker, QTimer
from PySide6.QtGui import QAction, QColor, QIcon, QPixmap
from PySide6.QtWidgets import QMenu, QSystemTrayIcon, QWidget

from maestro.app._paths import get_config_dir
from maestro.app.ui.controller import RuntimeController
from maestro.app.ui.floating_preview import FloatingPreviewWindow
from maestro.domain.entities import RuntimeState


def _make_tray_icon(colour: str) -> QIcon:
    """Create a 16×16 solid-colour tray icon."""
    px = QPixmap(16, 16)
    px.fill(QColor(colour))
    return QIcon(px)


class MainWindow(QWidget):
    """Tray-only runtime manager with a floating preview overlay."""

    def __init__(self, config_path: str | None = None) -> None:
        super().__init__()
        self._config_path = config_path or str(get_config_dir())
        self._preview_window = FloatingPreviewWindow()
        self._controller = RuntimeController(parent=self)

        self.setWindowTitle("Maestro")
        self.hide()

        self._tray = QSystemTrayIcon(self)
        self._tray.setIcon(_make_tray_icon("#6B7280"))
        self._tray.setToolTip("Maestro: inactive")

        self._active_action = QAction("Active", self)
        self._active_action.setCheckable(True)
        self._active_action.toggled.connect(self._on_active_toggled)

        self._preview_action = QAction("Preview", self)
        self._preview_action.setCheckable(True)
        self._preview_action.toggled.connect(self._on_preview_toggled)

        tray_menu = QMenu()
        tray_menu.addAction(self._active_action)
        tray_menu.addAction(self._preview_action)
        tray_menu.addSeparator()
        tray_menu.addAction("Quit", self.close)
        self._tray.setContextMenu(tray_menu)
        self._tray.show()

        self._controller.runtime_started.connect(self._on_runtime_started)
        self._controller.runtime_stopped.connect(self._on_runtime_stopped)
        self._controller.runtime_stopped.connect(lambda: self._tray.setIcon(_make_tray_icon("#6B7280")))
        self._controller.error_occurred.connect(self._on_error)
        self._controller.state_updated.connect(self._preview_window.on_state)
        self._controller.state_updated.connect(self._on_tray_update)

    def _on_active_toggled(self, active: bool) -> None:
        if active:
            if self._controller.is_running:
                return
            self._tray.setToolTip("Maestro: starting")
            self._controller.start(
                self._config_path,
                show_window=True,
                minimal_preview=True,
            )
            return

        if self._controller.is_running:
            self._tray.setToolTip("Maestro: stopping")
            self._controller.stop()
        self._sync_preview_visibility()

    def _on_preview_toggled(self, _enabled: bool) -> None:
        self._sync_preview_visibility()

    def _on_runtime_started(self) -> None:
        self._tray.setToolTip("Maestro: active")
        self._sync_preview_visibility()

    def _on_runtime_stopped(self) -> None:
        self._tray.setToolTip("Maestro: inactive")
        self._preview_window.hide()
        with QSignalBlocker(self._active_action):
            self._active_action.setChecked(False)

    def _on_error(self, message: str) -> None:
        self._tray.setIcon(_make_tray_icon("#DC2626"))
        self._tray.setToolTip("Maestro: error")
        self._preview_window.hide()
        with QSignalBlocker(self._active_action):
            self._active_action.setChecked(False)
        self._tray.showMessage(
            "Maestro",
            message,
            QSystemTrayIcon.MessageIcon.Critical,
            2000,
        )

    def _on_tray_update(self, state: RuntimeState) -> None:
        if not state.gaze_permitted:
            self._tray.setIcon(_make_tray_icon("#6B7280"))
            return
        if state.info.get("accepted"):
            self._tray.setIcon(_make_tray_icon("#D97706"))
            self._tray.showMessage(
                "Gesture",
                state.last_label,
                QSystemTrayIcon.MessageIcon.Information,
                800,
            )
            QTimer.singleShot(300, lambda: self._tray.setIcon(_make_tray_icon("#059669")))
            return
        self._tray.setIcon(_make_tray_icon("#059669"))

    def _sync_preview_visibility(self) -> None:
        if self._preview_action.isChecked() and self._controller.is_running:
            self._preview_window.show_preview()
            return
        self._preview_window.hide()

    def closeEvent(self, event) -> None:  # type: ignore[override]
        self._controller.stop()
        self._controller.wait(timeout_ms=3_000)
        self._preview_window.close()
        self._tray.hide()
        super().closeEvent(event)
