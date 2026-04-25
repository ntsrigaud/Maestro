from __future__ import annotations

from PySide6.QtCore import Slot
from PySide6.QtWidgets import (
    QFrame,
    QGridLayout,
    QGroupBox,
    QLabel,
    QProgressBar,
    QVBoxLayout,
    QWidget,
)

from maestro.domain.entities import RuntimeState

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _stat_pair(label_text: str) -> tuple[QLabel, QLabel]:
    """Return (key_label, value_label) for a stat row."""
    key = QLabel(label_text)
    key.setObjectName("statKey")
    val = QLabel("—")
    val.setObjectName("statValue")
    return key, val


def _divider() -> QFrame:
    line = QFrame()
    line.setFrameShape(QFrame.Shape.HLine)
    line.setObjectName("divider")
    return line


class StatusMonitor(QWidget):
    """
    Live telemetry display, signal-driven.

    Visual structure
    ----------------
    ┌─ Runtime ─────────────────────┐
    │ Mode        GESTURE           │
    │ Gaze        ✓ Active          │
    │ Gesture     swipe_right       │
    │ Confidence  ████░░░░  78 %    │
    ├─ Performance ─────────────────┤
    │ FPS         29.3 Hz           │
    │ Latency     12.4 ms / 8.1 ms  │
    │ Valid ratio 94 %              │
    └───────────────────────────────┘
    """

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)

        # Runtime group
        runtime_box = QGroupBox("Runtime")
        rg = QGridLayout(runtime_box)
        rg.setColumnStretch(1, 1)

        _, self._mode_val = _stat_pair("Mode")
        _, self._gaze_val = _stat_pair("Gaze")
        _, self._gesture_val = _stat_pair("Gesture")
        conf_key, _ = _stat_pair("Confidence")

        self._confidence_bar = QProgressBar()
        self._confidence_bar.setRange(0, 100)
        self._confidence_bar.setObjectName("confidenceBar")
        self._confidence_bar.setTextVisible(True)
        self._confidence_bar.setFormat("%v %")
        self._confidence_bar.setFixedHeight(14)

        k_mode = QLabel("Mode"); k_mode.setObjectName("statKey")
        k_gaze = QLabel("Gaze"); k_gaze.setObjectName("statKey")
        k_gesture = QLabel("Gesture"); k_gesture.setObjectName("statKey")

        rg.addWidget(k_mode, 0, 0)
        rg.addWidget(self._mode_val, 0, 1)
        rg.addWidget(k_gaze, 1, 0)
        rg.addWidget(self._gaze_val, 1, 1)
        rg.addWidget(k_gesture, 2, 0)
        rg.addWidget(self._gesture_val, 2, 1)
        rg.addWidget(conf_key, 3, 0)
        rg.addWidget(self._confidence_bar, 3, 1)

        # Performance group
        perf_box = QGroupBox("Performance")
        pg = QGridLayout(perf_box)
        pg.setColumnStretch(1, 1)

        k_fps = QLabel("FPS"); k_fps.setObjectName("statKey")
        k_lat = QLabel("Latency"); k_lat.setObjectName("statKey")
        k_vr = QLabel("Valid ratio"); k_vr.setObjectName("statKey")
        self._fps_val = QLabel("—"); self._fps_val.setObjectName("statValue")
        self._latency_val = QLabel("—"); self._latency_val.setObjectName("statValue")
        self._valid_ratio_val = QLabel("—"); self._valid_ratio_val.setObjectName("statValue")

        pg.addWidget(k_fps, 0, 0)
        pg.addWidget(self._fps_val, 0, 1)
        pg.addWidget(k_lat, 1, 0)
        pg.addWidget(self._latency_val, 1, 1)
        pg.addWidget(k_vr, 2, 0)
        pg.addWidget(self._valid_ratio_val, 2, 1)

        layout = QVBoxLayout(self)
        layout.setSpacing(8)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.addWidget(runtime_box)
        layout.addWidget(perf_box)

    @Slot(object)
    def on_state(self, state: RuntimeState) -> None:
        """Update all fields from a RuntimeState. Called on the main thread."""
        self._mode_val.setText(state.mode.upper())

        if state.gaze_permitted:
            self._gaze_val.setText("✓ Active")
            self._gaze_val.setObjectName("gazeActive")
        else:
            self._gaze_val.setText("✗ Absent")
            self._gaze_val.setObjectName("gazeAbsent")
        # Force stylesheet re-evaluation after objectName change
        self._gaze_val.style().unpolish(self._gaze_val)
        self._gaze_val.style().polish(self._gaze_val)

        self._gesture_val.setText(state.last_label or "—")
        self._confidence_bar.setValue(int(state.last_confidence * 100))

        info = state.info
        fps = info.get("fps", 0.0)
        self._fps_val.setText(f"{fps:.1f} Hz")

        e2e = info.get("latency_e2e_ms", 0.0)
        infer = info.get("latency_infer_ms", 0.0)
        self._latency_val.setText(f"{e2e:.1f} ms  /  {infer:.1f} ms")

        ratio = info.get("valid_ratio", 0.0)
        self._valid_ratio_val.setText(f"{ratio:.0%}")
