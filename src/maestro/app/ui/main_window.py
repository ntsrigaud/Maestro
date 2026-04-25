from __future__ import annotations

import datetime
from pathlib import Path

from PySide6.QtCore import QEvent, Qt, QTimer
from PySide6.QtGui import QColor, QIcon, QPixmap
from PySide6.QtWidgets import (
    QGroupBox,
    QLabel,
    QMainWindow,
    QMenu,
    QSplitter,
    QSystemTrayIcon,
    QTabWidget,
    QTextEdit,
    QVBoxLayout,
    QWidget,
)

from maestro.app.ui.config_panel import ConfigPanel
from maestro.app.ui.controller import RuntimeController
from maestro.app.ui.launcher_panel import LauncherPanel
from maestro.app.ui.model_panel import ModelPanel
from maestro.app.ui.preview_widget import PreviewWidget
from maestro.app.ui.status_monitor import StatusMonitor
from maestro.domain.entities import RuntimeState

_DEFAULT_CONFIG_PATH = "configs"

# ---------------------------------------------------------------------------
# Application stylesheet
# ---------------------------------------------------------------------------
# Colours follow an 8-shade neutral scale + accent (indigo) + semantic colours.
# Spacing is on a 4 px grid; font sizes on a 3-step scale (12 / 13 / 15 px).
# ---------------------------------------------------------------------------
_STYLESHEET = """
/* ── Globals ─────────────────────────────────────────────────────────── */
QWidget {
    font-family: -apple-system, "Segoe UI", "Helvetica Neue", Arial, sans-serif;
    font-size: 13px;
    color: #E2E8F0;
    background-color: #0F172A;
}

/* ── Main window / panels ────────────────────────────────────────────── */
QMainWindow {
    background-color: #0F172A;
}

/* Sidebar has a slightly lighter surface */
#sidebar {
    background-color: #1E293B;
    border-right: 1px solid #334155;
}

/* Content area */
#contentArea {
    background-color: #0F172A;
}

/* ── Group boxes ─────────────────────────────────────────────────────── */
QGroupBox {
    font-size: 12px;
    font-weight: 600;
    color: #94A3B8;
    letter-spacing: 0.05em;
    text-transform: uppercase;
    border: 1px solid #334155;
    border-radius: 8px;
    margin-top: 10px;
    padding: 12px 12px 8px 12px;
}
QGroupBox::title {
    subcontrol-origin: margin;
    left: 10px;
    padding: 0 4px;
    top: -1px;
}

/* ── Buttons ─────────────────────────────────────────────────────────── */
QPushButton {
    background-color: #334155;
    color: #E2E8F0;
    border: 1px solid #475569;
    border-radius: 6px;
    padding: 6px 14px;
    font-size: 13px;
}
QPushButton:hover {
    background-color: #3D5068;
    border-color: #64748B;
}
QPushButton:pressed {
    background-color: #1E3A5F;
}
QPushButton:disabled {
    background-color: #1E293B;
    color: #475569;
    border-color: #334155;
}

/* Primary action button (Start/Stop) */
QPushButton#primaryButton {
    background-color: #4F46E5;
    color: #FFFFFF;
    border: none;
    font-size: 14px;
    font-weight: 600;
    border-radius: 8px;
    padding: 8px 16px;
}
QPushButton#primaryButton:hover {
    background-color: #4338CA;
}
QPushButton#primaryButton:pressed {
    background-color: #3730A3;
}
QPushButton#primaryButton:disabled {
    background-color: #312E81;
    color: #818CF8;
}

/* ── Text inputs ─────────────────────────────────────────────────────── */
QLineEdit, QSpinBox, QDoubleSpinBox, QComboBox {
    background-color: #1E293B;
    color: #E2E8F0;
    border: 1px solid #334155;
    border-radius: 6px;
    padding: 4px 8px;
    selection-background-color: #4F46E5;
}
QLineEdit:focus, QSpinBox:focus, QDoubleSpinBox:focus, QComboBox:focus {
    border-color: #6366F1;
    outline: none;
}
QComboBox::drop-down {
    border: none;
    width: 20px;
}
QComboBox QAbstractItemView {
    background-color: #1E293B;
    border: 1px solid #334155;
    selection-background-color: #4F46E5;
}
QSpinBox::up-button, QSpinBox::down-button,
QDoubleSpinBox::up-button, QDoubleSpinBox::down-button {
    background-color: #334155;
    border: none;
    width: 16px;
}

/* ── Check boxes ─────────────────────────────────────────────────────── */
QCheckBox {
    spacing: 6px;
    color: #CBD5E1;
}
QCheckBox::indicator {
    width: 16px;
    height: 16px;
    border-radius: 4px;
    border: 1px solid #475569;
    background-color: #1E293B;
}
QCheckBox::indicator:checked {
    background-color: #4F46E5;
    border-color: #4F46E5;
}

/* ── Progress bar ────────────────────────────────────────────────────── */
QProgressBar {
    background-color: #1E293B;
    border: 1px solid #334155;
    border-radius: 4px;
    text-align: center;
    color: #94A3B8;
    font-size: 11px;
}
QProgressBar::chunk {
    background-color: #4F46E5;
    border-radius: 3px;
}
QProgressBar#confidenceBar::chunk {
    background-color: #059669;
}

/* ── Scroll areas ────────────────────────────────────────────────────── */
QScrollArea {
    background-color: transparent;
    border: none;
}
QScrollBar:vertical {
    background: #1E293B;
    width: 8px;
    border-radius: 4px;
}
QScrollBar::handle:vertical {
    background: #475569;
    border-radius: 4px;
    min-height: 30px;
}
QScrollBar::add-line:vertical, QScrollBar::sub-line:vertical {
    height: 0;
}

/* ── Tabs (Settings panel) ───────────────────────────────────────────── */
QTabWidget::pane {
    background-color: #1E293B;
    border: 1px solid #334155;
    border-radius: 8px;
    padding: 8px;
}
QTabBar::tab {
    background-color: #0F172A;
    color: #94A3B8;
    border: 1px solid #334155;
    border-bottom: none;
    border-radius: 6px 6px 0 0;
    padding: 6px 16px;
    margin-right: 2px;
    font-size: 12px;
    font-weight: 500;
}
QTabBar::tab:selected {
    background-color: #1E293B;
    color: #E2E8F0;
    border-color: #334155;
}
QTabBar::tab:hover:!selected {
    background-color: #1E293B;
    color: #CBD5E1;
}

/* ── Session log ─────────────────────────────────────────────────────── */
QTextEdit#sessionLog {
    background-color: #0D1117;
    color: #94A3B8;
    border: 1px solid #334155;
    border-radius: 6px;
    font-family: "SF Mono", "Fira Code", "Consolas", monospace;
    font-size: 11px;
    padding: 4px;
}

/* ── Stat labels ─────────────────────────────────────────────────────── */
QLabel#statKey {
    color: #64748B;
    font-size: 12px;
}
QLabel#statValue {
    color: #CBD5E1;
    font-weight: 500;
}
QLabel#gazeActive {
    color: #10B981;
    font-weight: 600;
}
QLabel#gazeAbsent {
    color: #64748B;
}
QLabel#statusChipLabel {
    font-size: 12px;
    font-weight: 600;
}

/* ── Splitter ────────────────────────────────────────────────────────── */
QSplitter::handle {
    background-color: #334155;
    width: 1px;
}

/* ── Horizontal divider ─────────────────────────────────────────────── */
QFrame#divider {
    color: #334155;
}

/* ── App title label ─────────────────────────────────────────────────── */
QLabel#appTitle {
    font-size: 15px;
    font-weight: 700;
    color: #E2E8F0;
    letter-spacing: 0.02em;
}
QLabel#appSubtitle {
    font-size: 11px;
    color: #64748B;
}
"""


def _make_tray_icon(colour: str) -> QIcon:
    """Create a 16×16 solid-colour tray icon."""
    px = QPixmap(16, 16)
    px.fill(QColor(colour))
    return QIcon(px)


def _section_label(text: str) -> QLabel:
    """Return a small all-caps section header label."""
    lbl = QLabel(text.upper())
    lbl.setObjectName("statKey")
    lbl.setContentsMargins(0, 8, 0, 2)
    return lbl


class MainWindow(QMainWindow):
    """
    Root window — two-pane layout.

    ┌─────────────────────────────────────────────────────────┐
    │  Sidebar (fixed ~240 px)  │  Content area               │
    │  ─────────────────────    │  ────────────────────────── │
    │  Maestro                  │  ┌── Camera Preview ──────┐ │
    │                           │  │  PreviewWidget         │ │
    │  ● Status                 │  └────────────────────────┘ │
    │  [  Start Session  ]      │                             │
    │  ☐ Show camera preview    │  ┌── Settings ────────────┐ │
    │                           │  │  [Config] [Model]  tab │ │
    │  Runtime ─────────────    │  └────────────────────────┘ │
    │  Mode  GESTURE            │                             │
    │  Gaze  ✓ Active           │  Session Log                │
    │  ...                      │                             │
    │                           │                             │
    │  Performance ─────────    │                             │
    │  FPS   29 Hz              │                             │
    └─────────────────────────────────────────────────────────┘

    Dependency rule: MainWindow → RuntimeController → (bootstrap, runtime).
    No panel talks to RuntimeController directly.
    """

    def __init__(self, config_path: str = _DEFAULT_CONFIG_PATH) -> None:
        super().__init__()
        self._config_path = config_path
        self._last_mode = ""
        self.setWindowTitle("Maestro — Gesture Control")
        self.setMinimumSize(760, 540)
        self.resize(980, 660)

        # Apply stylesheet
        self.setStyleSheet(_STYLESHEET)

        # ── Subcomponents ──────────────────────────────────────────────
        self._controller = RuntimeController(parent=self)
        self._launcher = LauncherPanel()
        self._preview = PreviewWidget()
        self._monitor = StatusMonitor()
        self._config_panel = ConfigPanel(config_path=config_path)

        try:
            from maestro.utils.config_loader import load_runtime_config

            _cfg = load_runtime_config(Path(config_path))
            _model_src = _cfg.model_source
        except Exception:
            _model_src = None
        self._model_panel = ModelPanel(default_source=_model_src)

        # ── Sidebar ────────────────────────────────────────────────────
        sidebar = QWidget()
        sidebar.setObjectName("sidebar")
        sidebar.setFixedWidth(240)
        sidebar_layout = QVBoxLayout(sidebar)
        sidebar_layout.setContentsMargins(16, 16, 16, 16)
        sidebar_layout.setSpacing(12)

        # App identity
        title = QLabel("Maestro")
        title.setObjectName("appTitle")
        subtitle = QLabel("Gesture Control System")
        subtitle.setObjectName("appSubtitle")
        sidebar_layout.addWidget(title)
        sidebar_layout.addWidget(subtitle)

        # Control
        ctrl_box = QGroupBox("Control")
        ctrl_box_layout = QVBoxLayout(ctrl_box)
        ctrl_box_layout.setSpacing(8)
        ctrl_box_layout.addWidget(self._launcher)
        sidebar_layout.addWidget(ctrl_box)

        # Status monitor
        sidebar_layout.addWidget(self._monitor)

        sidebar_layout.addStretch()

        # ── Content area ───────────────────────────────────────────────
        content = QWidget()
        content.setObjectName("contentArea")
        content_layout = QVBoxLayout(content)
        content_layout.setContentsMargins(16, 16, 16, 16)
        content_layout.setSpacing(12)

        # Preview (collapses when hidden)
        preview_box = QGroupBox("Camera Preview")
        preview_box_layout = QVBoxLayout(preview_box)
        preview_box_layout.setContentsMargins(8, 8, 8, 8)
        preview_box_layout.addWidget(self._preview)
        content_layout.addWidget(preview_box)

        # Settings tabs: Config + Model
        settings_tabs = QTabWidget()
        settings_tabs.addTab(self._config_panel, "Configuration")
        settings_tabs.addTab(self._model_panel, "Model Source")
        content_layout.addWidget(settings_tabs)

        # Session log
        self._log = QTextEdit()
        self._log.setObjectName("sessionLog")
        self._log.setReadOnly(True)
        self._log.setFixedHeight(110)
        self._log.setPlaceholderText("Session events will appear here…")
        content_layout.addWidget(self._log)

        # ── Splitter: sidebar | content ────────────────────────────────
        splitter = QSplitter(Qt.Orientation.Horizontal)
        splitter.addWidget(sidebar)
        splitter.addWidget(content)
        splitter.setChildrenCollapsible(False)
        splitter.setStretchFactor(0, 0)  # sidebar: fixed
        splitter.setStretchFactor(1, 1)  # content: expands
        self.setCentralWidget(splitter)

        # ── System tray ────────────────────────────────────────────────
        self._tray = QSystemTrayIcon(self)
        self._tray.setIcon(_make_tray_icon("#6B7280"))
        tray_menu = QMenu()
        tray_menu.addAction("Show", self.show)
        tray_menu.addAction("Quit", self.close)
        self._tray.setContextMenu(tray_menu)
        self._tray.activated.connect(self._on_tray_activated)
        self._tray.show()

        # ── Signal wiring ──────────────────────────────────────────────
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
            lambda: self._tray.setIcon(_make_tray_icon("#6B7280"))
        )
        self._controller.error_occurred.connect(self._launcher.on_error)
        self._controller.state_updated.connect(self._monitor.on_state)
        self._controller.state_updated.connect(self._preview.on_state)
        self._controller.state_updated.connect(self._on_tray_update)
        self._controller.state_updated.connect(self._on_log_event)
        self._controller.runtime_stopped.connect(self._preview.hide)

        self._config_panel.apply_requested.connect(self._on_apply_requested)

    # ------------------------------------------------------------------
    # Internal slots
    # ------------------------------------------------------------------

    def _on_start_requested(self) -> None:
        self._launcher.on_start_clicked()
        self._controller.start(
            self._config_path,
            artifact_path=self._model_panel.artifact_path(),
            hf_revision_override=self._model_panel.hf_revision_override(),
            show_window=self._launcher.show_window,
        )

    def _on_apply_requested(self) -> None:
        """YAML written; restart runtime with the new config."""
        self._controller.stop()
        self._controller.wait(timeout_ms=3_000)
        self._on_start_requested()

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
        else:
            self._tray.setIcon(_make_tray_icon("#059669"))

    def _on_log_event(self, state: RuntimeState) -> None:
        if state.mode != self._last_mode:
            ts = datetime.datetime.now().strftime("%H:%M:%S")
            self._log.append(f"[{ts}] Mode → {state.mode.upper()}")
            self._last_mode = state.mode
        if state.info.get("accepted"):
            ts = datetime.datetime.now().strftime("%H:%M:%S")
            self._log.append(f"[{ts}] Dispatched: {state.last_label} ({state.last_confidence:.0%})")

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
