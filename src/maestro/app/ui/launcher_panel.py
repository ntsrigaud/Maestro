from __future__ import annotations

from PySide6.QtCore import Signal
from PySide6.QtWidgets import (
    QCheckBox,
    QHBoxLayout,
    QLabel,
    QPushButton,
    QVBoxLayout,
    QWidget,
)

# ---------------------------------------------------------------------------
# Design tokens (colours only — spacing handled by layouts)
# ---------------------------------------------------------------------------
_COLOUR = {
    "idle": "#6B7280",      # neutral grey
    "building": "#D97706",  # amber
    "running": "#059669",   # emerald green
    "error": "#DC2626",     # red
}


class LauncherPanel(QWidget):
    """
    Primary runtime control surface.

    Emits start_requested / stop_requested. Never calls RuntimeController
    directly — that wiring is done in MainWindow.

    Visual structure
    ----------------
    ┌─────────────────────────────────┐
    │  ● STATUS CHIP                  │
    │  [   Start / Stop button   ]    │
    │  ☐ Show preview window          │
    └─────────────────────────────────┘
    """

    start_requested: Signal = Signal()
    stop_requested: Signal = Signal()

    _LABELS = {
        "idle": "Start Session",
        "building": "Starting\u2026",
        "running": "Stop Session",
        "error": "Retry",
    }
    _CHIP_TEXT = {
        "idle": "Idle",
        "building": "Starting",
        "running": "Running",
        "error": "Error",
    }

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self._state = "idle"

        # Status chip: coloured dot + text label
        self._dot = QLabel()
        self._dot.setFixedSize(10, 10)
        self._status_text = QLabel(self._CHIP_TEXT["idle"])
        self._status_text.setObjectName("statusChipLabel")

        chip_row = QHBoxLayout()
        chip_row.setSpacing(6)
        chip_row.addWidget(self._dot)
        chip_row.addWidget(self._status_text)
        chip_row.addStretch()

        # Primary action button
        self._btn = QPushButton(self._LABELS["idle"])
        self._btn.setObjectName("primaryButton")
        self._btn.setMinimumHeight(36)

        # Preview toggle
        self._show_window_cb = QCheckBox("Show camera preview")
        self._show_window_cb.setChecked(False)

        layout = QVBoxLayout(self)
        layout.setSpacing(8)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.addLayout(chip_row)
        layout.addWidget(self._btn)
        layout.addWidget(self._show_window_cb)

        self._btn.clicked.connect(self._on_click)
        self._apply_state()

    # ------------------------------------------------------------------
    # Public slots (connected via MainWindow)
    # ------------------------------------------------------------------

    def on_runtime_started(self) -> None:
        self._set_state("running")

    def on_runtime_stopped(self) -> None:
        self._set_state("idle")

    def on_error(self, _msg: str) -> None:
        self._set_state("error")

    def on_start_clicked(self) -> None:
        """Called by MainWindow immediately after start() is invoked."""
        self._set_state("building")

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------

    def _on_click(self) -> None:
        if self._state in ("idle", "error"):
            self.start_requested.emit()
        elif self._state == "running":
            self.stop_requested.emit()

    def _set_state(self, state: str) -> None:
        self._state = state
        self._apply_state()

    def _apply_state(self) -> None:
        colour = _COLOUR[self._state]
        self._dot.setStyleSheet(
            f"background-color: {colour}; border-radius: 5px;"
        )
        self._status_text.setText(self._CHIP_TEXT[self._state])
        self._status_text.setStyleSheet(f"color: {colour}; font-weight: 600;")
        self._btn.setText(self._LABELS[self._state])
        self._btn.setEnabled(self._state != "building")

    @property
    def state(self) -> str:
        return self._state

    @property
    def show_window(self) -> bool:
        return self._show_window_cb.isChecked()
