from __future__ import annotations

from PySide6.QtCore import Signal
from PySide6.QtWidgets import QCheckBox, QHBoxLayout, QLabel, QPushButton, QWidget


class LauncherPanel(QWidget):
    """
    Emits start_requested / stop_requested. Never calls RuntimeController
    directly — that wiring is done in MainWindow.
    """

    start_requested: Signal = Signal()
    stop_requested: Signal = Signal()

    _LABELS = {
        "idle": "Start",
        "building": "Starting\u2026",
        "running": "Stop",
        "error": "Retry",
    }
    _COLOURS = {
        "idle": "#888888",
        "building": "#FFA500",
        "running": "#44AA44",
        "error": "#CC3333",
    }

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self._state = "idle"

        self._btn = QPushButton(self._LABELS["idle"])
        self._pill = QLabel()
        self._pill.setFixedSize(12, 12)
        self._show_window_cb = QCheckBox("Show preview window")
        self._show_window_cb.setChecked(False)

        layout = QHBoxLayout(self)
        layout.addWidget(self._btn)
        layout.addWidget(self._pill)
        layout.addWidget(self._show_window_cb)
        layout.addStretch()

        self._btn.clicked.connect(self._on_click)
        self._apply_state()

    # ------------------------------------------------------------------
    # Public slots (connect RuntimeController signals here via MainWindow)
    # ------------------------------------------------------------------

    def on_runtime_started(self) -> None:
        self._set_state("running")

    def on_runtime_stopped(self) -> None:
        self._set_state("idle")

    def on_error(self, _msg: str) -> None:
        self._set_state("error")

    def on_start_clicked(self) -> None:
        """Called by MainWindow after RuntimeController.start() is invoked."""
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
        self._btn.setText(self._LABELS[self._state])
        self._btn.setEnabled(self._state != "building")
        colour = self._COLOURS[self._state]
        self._pill.setStyleSheet(f"background-color: {colour}; border-radius: 6px;")

    @property
    def state(self) -> str:
        return self._state

    @property
    def show_window(self) -> bool:
        """Whether the user wants the OpenCV preview window shown."""
        return self._show_window_cb.isChecked()
