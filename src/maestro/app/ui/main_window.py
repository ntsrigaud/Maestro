from __future__ import annotations

from pathlib import Path

import yaml
from PySide6.QtCore import QCoreApplication, QSignalBlocker, Qt
from PySide6.QtGui import QAction, QColor, QIcon, QPainter, QPen, QPixmap
from PySide6.QtWidgets import QMenu, QSystemTrayIcon, QWidget

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

        # Inactive tray icon — the real app icon when available, grey dot otherwise.
        icon_path = get_app_icon()
        self._base_icon = QIcon(str(icon_path)) if icon_path else None
        self._tray_icons_cache: dict[str, QIcon] = {}
        self._inactive_icon = self._tray_icon_for_colour("#6B7280")

        self._tray = QSystemTrayIcon(self)
        self._tray.setIcon(self._inactive_icon)
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
        self._tray.setIcon(self._inactive_icon)
        self._preview_window.hide()
        with QSignalBlocker(self._active_action):
            self._active_action.setChecked(False)
        # Flush any gaze setting that was toggled while the session was running.
        if self._pending_gaze is not None:
            self._write_gaze_enabled(self._pending_gaze)
            self._pending_gaze = None

    def _on_error(self, message: str) -> None:
        self._tray.setIcon(self._tray_icon_for_colour("#DC2626"))
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
        colour = self._tray_colour_for_state(state)
        self._tray.setIcon(self._tray_icon_for_colour(colour))

        if state.info.get("accepted"):
            self._tray.showMessage(
                "Gesture",
                str(state.last_label),
                QSystemTrayIcon.MessageIcon.Information,
                800,
            )

    def _tray_colour_for_state(self, state: RuntimeState) -> str:
        """Map runtime state to the same colour theme used by floating preview badge."""
        gaze_enabled = bool(state.info.get("gaze_enabled", True))
        laser_active = bool(state.info.get("laser_active", False))
        pen_active = bool(state.info.get("pen_active", False))

        if not gaze_enabled:
            return "#1E293B"  # dark slate — gaze detection disabled
        if not state.gaze_permitted:
            return "#4B5563"  # grey — gaze absent
        if laser_active:
            return "#2563EB"  # blue — laser pointer active
        if pen_active:
            return "#7C3AED"  # violet — annotation pen active
        if state.mode == "armed":
            return "#059669"  # green — armed
        return "#D97706"  # amber — idle/cooldown with gaze present

    def _tray_icon_for_colour(self, colour: str) -> QIcon:
        cached = self._tray_icons_cache.get(colour)
        if cached is not None:
            return cached
        icon = _make_state_tray_icon(self._base_icon, colour)
        self._tray_icons_cache[colour] = icon
        return icon

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
        QCoreApplication.quit()
