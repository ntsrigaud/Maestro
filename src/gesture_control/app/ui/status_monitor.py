from __future__ import annotations

from PySide6.QtCore import Slot
from PySide6.QtWidgets import (
    QFormLayout,
    QLabel,
    QProgressBar,
    QWidget,
)

from gesture_control.domain.entities import RuntimeState


class StatusMonitor(QWidget):
    """
    Displays per-frame RuntimeState telemetry.

    Connect RuntimeController.state_updated to this widget's on_state slot.
    This widget never queries the runtime; it is entirely signal-driven.
    """

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)

        self._mode_label = QLabel("—")
        self._gaze_label = QLabel("—")
        self._gesture_label = QLabel("—")
        self._confidence_bar = QProgressBar()
        self._confidence_bar.setRange(0, 100)
        self._fps_label = QLabel("—")
        self._latency_e2e_label = QLabel("—")
        self._latency_infer_label = QLabel("—")
        self._valid_ratio_label = QLabel("—")

        layout = QFormLayout(self)
        layout.addRow("Mode:", self._mode_label)
        layout.addRow("Gaze:", self._gaze_label)
        layout.addRow("Gesture:", self._gesture_label)
        layout.addRow("Confidence:", self._confidence_bar)
        layout.addRow("FPS:", self._fps_label)
        layout.addRow("Latency (e2e):", self._latency_e2e_label)
        layout.addRow("Latency (infer):", self._latency_infer_label)
        layout.addRow("Valid ratio:", self._valid_ratio_label)

    @Slot(object)
    def on_state(self, state: RuntimeState) -> None:
        """Update all fields from a RuntimeState. Called on the main thread."""
        self._mode_label.setText(state.mode.upper())
        self._gaze_label.setText("✓ Active" if state.gaze_permitted else "✗ Absent")

        gesture = state.last_label or "—"
        self._gesture_label.setText(gesture)
        self._confidence_bar.setValue(int(state.last_confidence * 100))

        info = state.info
        fps = info.get("fps", 0.0)
        self._fps_label.setText(f"{fps:.1f} Hz")

        e2e = info.get("latency_e2e_ms", 0.0)
        self._latency_e2e_label.setText(f"{e2e:.1f} ms")

        infer = info.get("latency_infer_ms", 0.0)
        self._latency_infer_label.setText(f"{infer:.1f} ms")

        ratio = info.get("valid_ratio", 0.0)
        self._valid_ratio_label.setText(f"{ratio:.0%}")
