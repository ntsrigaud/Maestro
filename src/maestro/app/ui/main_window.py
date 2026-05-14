from __future__ import annotations

from pathlib import Path

import yaml
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
        # Gaze setting written to YAML when the runtime stops (if changed while running).
        self._pending_gaze: bool | None = None

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

        self._gaze_action = QAction("Gaze detection", self)
        self._gaze_action.setCheckable(True)
        self._gaze_action.setChecked(self._read_gaze_enabled())
        self._gaze_action.setToolTip(
            "When enabled, gesture commands are only dispatched while the user\n"
            "is looking at the camera. Takes effect on the next session start."
        )
        self._gaze_action.toggled.connect(self._on_gaze_toggled)

        tray_menu = QMenu()
        tray_menu.addAction(self._active_action)
        tray_menu.addAction(self._preview_action)
        tray_menu.addAction(self._gaze_action)
        tray_menu.addSeparator()
        tray_menu.addAction("Quit", self.close)
        self._tray.setContextMenu(tray_menu)
        self._tray.show()

        self._controller.runtime_started.connect(self._on_runtime_started)
        self._controller.runtime_stopped.connect(self._on_runtime_stopped)
        self._controller.runtime_stopped.connect(
            lambda: self._tray.setIcon(_make_tray_icon("#6B7280"))
        )
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

    def _on_gaze_toggled(self, enabled: bool) -> None:
        """Persist the gaze-detection setting or defer if a session is running."""
        if self._controller.is_running:
            # Cannot safely change the running container; defer to session stop.
            self._pending_gaze = enabled
            label = "enabled" if enabled else "disabled"
            self._tray.showMessage(
                "Maestro",
                f"Gaze detection {label} — takes effect on next session.",
                QSystemTrayIcon.MessageIcon.Information,
                2500,
            )
        else:
            self._write_gaze_enabled(enabled)

    def _on_runtime_started(self) -> None:
        self._tray.setToolTip("Maestro: active")
        self._preview_window.reset()
        self._sync_preview_visibility()

    def _on_runtime_stopped(self) -> None:
        self._tray.setToolTip("Maestro: inactive")
        self._preview_window.hide()
        with QSignalBlocker(self._active_action):
            self._active_action.setChecked(False)
        # Flush any gaze setting that was toggled while the session was running.
        if self._pending_gaze is not None:
            self._write_gaze_enabled(self._pending_gaze)
            self._pending_gaze = None

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

    def _inference_yaml_path(self) -> Path:
        return Path(self._config_path) / "inference.yaml"

    def _read_gaze_enabled(self) -> bool:
        """Read ``gaze.enabled`` from inference.yaml; defaults to False."""
        path = self._inference_yaml_path()
        if not path.exists():
            return False
        try:
            data = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
            return bool(data.get("gaze", {}).get("enabled", False))
        except Exception:
            return False

    def _write_gaze_enabled(self, value: bool) -> None:
        """Persist ``gaze.enabled`` to inference.yaml."""
        path = self._inference_yaml_path()
        if not path.exists():
            return
        try:
            data = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
            if not isinstance(data.get("gaze"), dict):
                data["gaze"] = {}
            data["gaze"]["enabled"] = value
            path.write_text(
                yaml.dump(
                    data,
                    default_flow_style=False,
                    sort_keys=False,
                    allow_unicode=True,
                ),
                encoding="utf-8",
            )
        except Exception:
            pass  # Non-fatal: worst case the setting reverts on next launch

    def closeEvent(self, event) -> None:  # type: ignore[override]
        self._controller.stop()
        self._controller.wait(timeout_ms=3_000)
        self._preview_window.close()
        self._tray.hide()
        super().closeEvent(event)
