from __future__ import annotations

import datetime
from pathlib import Path

from PySide6.QtCore import QEvent, QTimer
from PySide6.QtGui import QColor, QIcon, QPixmap
from PySide6.QtWidgets import (
    QMainWindow,
    QMenu,
    QSystemTrayIcon,
    QTextEdit,
    QVBoxLayout,
    QWidget,
)

from gesture_control.app.ui.config_panel import ConfigPanel
from gesture_control.app.ui.controller import RuntimeController
from gesture_control.app.ui.launcher_panel import LauncherPanel
from gesture_control.app.ui.model_panel import ModelPanel
from gesture_control.app.ui.status_monitor import StatusMonitor
from gesture_control.domain.entities import RuntimeState

_DEFAULT_CONFIG_PATH = "configs"


def _make_tray_icon(colour: str) -> QIcon:
    """Create a 16×16 solid-colour icon."""
    px = QPixmap(16, 16)
    px.fill(QColor(colour))
    return QIcon(px)


class MainWindow(QMainWindow):
    """
    Root window. Owns RuntimeController and wires signals between panels.

    Dependency rule: MainWindow → RuntimeController → (bootstrap, runtime).
    No panel communicates with RuntimeController directly; all routing goes
    through MainWindow.
    """

    def __init__(self, config_path: str = _DEFAULT_CONFIG_PATH) -> None:
        super().__init__()
        self._config_path = config_path
        self._last_mode = ""
        self.setWindowTitle("Gesture Control")
        self.setMinimumSize(400, 300)

        # --- Subcomponents ---
        self._controller = RuntimeController(parent=self)
        self._launcher = LauncherPanel()
        self._monitor = StatusMonitor()
        self._config_panel = ConfigPanel(config_path=config_path)

        # ModelPanel: pre-populate from model_source.yaml if present
        try:
            from gesture_control.utils.config_loader import load_runtime_config

            _cfg = load_runtime_config(Path(config_path))
            _model_src = _cfg.model_source
        except Exception:
            _model_src = None
        self._model_panel = ModelPanel(default_source=_model_src)

        # --- Session log ---
        self._log = QTextEdit()
        self._log.setReadOnly(True)
        self._log.setMaximumHeight(120)

        # --- Layout ---
        central = QWidget()
        layout = QVBoxLayout(central)
        layout.addWidget(self._launcher)
        layout.addWidget(self._monitor)
        layout.addWidget(self._config_panel)
        layout.addWidget(self._model_panel)
        layout.addWidget(self._log)
        self.setCentralWidget(central)

        # --- System tray ---
        self._tray = QSystemTrayIcon(self)
        self._tray.setIcon(_make_tray_icon("#888888"))
        tray_menu = QMenu()
        tray_menu.addAction("Show", self.show)
        tray_menu.addAction("Quit", self.close)
        self._tray.setContextMenu(tray_menu)
        self._tray.activated.connect(self._on_tray_activated)
        self._tray.show()

        # --- Signal wiring ---
        self._launcher.start_requested.connect(self._on_start_requested)
        self._launcher.stop_requested.connect(self._controller.stop)

        self._controller.runtime_started.connect(self._launcher.on_runtime_started)
        self._controller.runtime_started.connect(
            lambda: self._config_panel.set_runtime_active(True)
        )
        self._controller.runtime_stopped.connect(self._launcher.on_runtime_stopped)
        self._controller.runtime_stopped.connect(
            lambda: self._config_panel.set_runtime_active(False)
        )
        self._controller.runtime_stopped.connect(
            lambda: self._tray.setIcon(_make_tray_icon("#888888"))
        )
        self._controller.error_occurred.connect(self._launcher.on_error)
        self._controller.state_updated.connect(self._monitor.on_state)
        self._controller.state_updated.connect(self._on_tray_update)
        self._controller.state_updated.connect(self._on_log_event)

        self._config_panel.apply_requested.connect(self._on_apply_requested)

    # ------------------------------------------------------------------
    # Internal slots
    # ------------------------------------------------------------------

    def _on_start_requested(self) -> None:
        self._launcher.on_start_clicked()  # transition to "building" state
        self._controller.start(
            self._config_path,
            artifact_path=self._model_panel.artifact_path(),
            hf_revision_override=self._model_panel.hf_revision_override(),
            show_window=self._launcher.show_window,
        )

    def _on_apply_requested(self) -> None:
        """YAML has been written; restart the runtime with the new config."""
        self._controller.stop()
        self._controller.wait(timeout_ms=3_000)
        self._on_start_requested()

    def _on_tray_update(self, state: RuntimeState) -> None:
        if not state.gaze_permitted:
            self._tray.setIcon(_make_tray_icon("#888888"))
            return
        if state.info.get("accepted"):
            self._tray.setIcon(_make_tray_icon("#FFA500"))
            self._tray.showMessage(
                "Gesture",
                state.last_label,
                QSystemTrayIcon.MessageIcon.Information,
                800,
            )
            QTimer.singleShot(
                300, lambda: self._tray.setIcon(_make_tray_icon("#44AA44"))
            )
        else:
            self._tray.setIcon(_make_tray_icon("#44AA44"))

    def _on_log_event(self, state: RuntimeState) -> None:
        if state.mode != self._last_mode:
            ts = datetime.datetime.now().strftime("%H:%M:%S")
            self._log.append(f"[{ts}] Mode \u2192 {state.mode.upper()}")
            self._last_mode = state.mode
        if state.info.get("accepted"):
            ts = datetime.datetime.now().strftime("%H:%M:%S")
            self._log.append(
                f"[{ts}] Dispatched: {state.last_label} ({state.last_confidence:.0%})"
            )

    def _on_tray_activated(self, reason: QSystemTrayIcon.ActivationReason) -> None:
        if reason == QSystemTrayIcon.ActivationReason.Trigger:
            self.show()
            self.raise_()
            self.activateWindow()

    def changeEvent(self, event: QEvent) -> None:  # type: ignore[override]
        if event.type() == QEvent.Type.WindowStateChange and self.isMinimized():
            event.ignore()
            self.hide()
            return
        super().changeEvent(event)

    def closeEvent(self, event) -> None:  # type: ignore[override]
        """Graceful shutdown: stop the runtime before closing."""
        self._controller.stop()
        self._controller.wait(timeout_ms=3_000)
        self._tray.hide()
        super().closeEvent(event)
