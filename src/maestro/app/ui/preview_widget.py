from __future__ import annotations

import numpy as np
from PySide6.QtCore import Qt, Slot
from PySide6.QtGui import QImage, QPixmap
from PySide6.QtWidgets import QLabel, QSizePolicy

from maestro.domain.entities import RuntimeState

# Maximum display width in pixels.  Frames are scaled down to fit; the
# aspect ratio is preserved.  Keeps the window manageable for typical
# 1280×720 camera resolutions.
_MAX_DISPLAY_WIDTH = 480


class PreviewWidget(QLabel):
    """
    Qt-native camera preview panel.

    Receives ``RuntimeState`` objects via the ``on_state`` slot and displays
    ``state.preview_frame`` (a BGR numpy array) as a scaled ``QPixmap``.
    All rendering happens on the main thread through the Qt signal/slot
    mechanism, eliminating the threading violation caused by calling
    ``cv2.imshow()`` from the QThread worker.

    Thread safety
    -------------
    ``on_state`` is connected via a *queued* Qt connection (cross-thread
    signal).  Qt delivers the call on the main thread regardless of which
    thread emits the signal.  ``state.preview_frame`` is a fresh canvas
    copy produced by ``OverlayRenderer`` in the worker thread; it is never
    mutated after being attached to the state object, so no additional copy
    is needed here.

    Lifecycle
    ---------
    The widget starts hidden.  It becomes visible on the first frame that
    carries a non-``None`` ``preview_frame``.  Call ``hide()`` to collapse
    it when the runtime stops.
    """

    def __init__(self, parent: QLabel | None = None) -> None:
        super().__init__(parent)
        self.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.setSizePolicy(
            QSizePolicy.Policy.Expanding,
            QSizePolicy.Policy.Fixed,
        )
        self.setMinimumHeight(0)
        self.hide()

    @Slot(object)
    def on_state(self, state: RuntimeState) -> None:
        """Update the preview from a RuntimeState.  Called on the main thread."""
        frame: np.ndarray | None = state.preview_frame
        if frame is None:
            return

        h, w = frame.shape[:2]

        # The frame is already in RGB order: OverlayRenderer is instantiated
        # with output_rgb=True whenever show_window=True (see bootstrap.py).
        # Pass bytes directly to QImage — no channel copy needed.
        rgb_bytes = frame.tobytes()
        stride = w * 3  # bytes per row (3 channels, uint8)
        image = QImage(rgb_bytes, w, h, stride, QImage.Format.Format_RGB888)
        pixmap = QPixmap.fromImage(image)

        display_width = min(w, _MAX_DISPLAY_WIDTH)
        if display_width < w:
            pixmap = pixmap.scaledToWidth(
                display_width,
                Qt.TransformationMode.SmoothTransformation,
            )

        self.setFixedHeight(pixmap.height())
        self.setPixmap(pixmap)

        if not self.isVisible():
            self.show()
