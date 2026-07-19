from __future__ import annotations

import logging

from PySide6.QtCore import Qt
from PySide6.QtWidgets import QMainWindow, QVBoxLayout, QWidget

from maestro.app.ui.controller import RuntimeController
from maestro.app.ui.preview_widget import PreviewWidget

LOGGER = logging.getLogger(__name__)


class CLIPreviewWindow(QMainWindow):
    """
    Minimal Qt window for CLI preview mode.

    Provides the same Qt-native frame rendering as the full GUI (via
    ``PreviewWidget``) without the dashboard panels.  This ensures a
    single, consistent display path across both CLI and GUI modes.

    Threading model
    ---------------
    ``RuntimeController`` owns a ``QThread`` worker that runs
    ``build_container()`` and ``Runtime.run()``.  Rendered frames are
    delivered as ``RuntimeState.preview_frame`` over Qt's queued signal
    mechanism and displayed by ``PreviewWidget`` on the main thread.
    No ``cv2`` window or ``cv2.imshow()`` call is made anywhere.

    Lifecycle
    ---------
    1. Caller creates ``CLIPreviewWindow``, calls ``window.show()``.
    2. Caller calls ``window.start(config_path, ...)`` which forwards
       to ``RuntimeController.start()``.
    3. When ``Runtime.run()`` exits (normal or error), ``runtime_stopped``
       closes the window and the ``QApplication`` event loop returns.
    4. Pressing **Q** or **Escape** stops the runtime gracefully.
    5. Closing the window (title-bar ✕) also stops the runtime.
    """

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.setWindowTitle("Maestro — Preview")

        self._controller = RuntimeController(parent=self)
        self._preview = PreviewWidget()

        central = QWidget()
        layout = QVBoxLayout(central)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(0)
        layout.addWidget(self._preview)
        self.setCentralWidget(central)

        # Frames flow: worker thread → state_updated signal (queued) →
        # PreviewWidget.on_state() on main thread.
        self._controller.state_updated.connect(self._preview.on_state)
        self._controller.runtime_stopped.connect(self.close)
        self._controller.error_occurred.connect(self._on_error)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def start(
        self,
        config_path: str,
        *,
        artifact_path: str | None = None,
        hf_revision_override: str | None = None,
    ) -> None:
        """Start the inference worker.  Must be called after ``show()``."""
        self._controller.start(
            config_path,
            artifact_path=artifact_path,
            hf_revision_override=hf_revision_override,
            show_window=True,
        )

    # ------------------------------------------------------------------
    # Qt event overrides
    # ------------------------------------------------------------------

    def keyPressEvent(self, event: object) -> None:  # type: ignore[override]
        key = getattr(event, "key", lambda: None)()
        if key in (Qt.Key.Key_Q, Qt.Key.Key_Escape):
            self._controller.stop()
        super().keyPressEvent(event)  # type: ignore[arg-type]

    def closeEvent(self, event: object) -> None:  # type: ignore[override]
        """Ensure the worker is stopped before the window is destroyed."""
        self._controller.stop()
        self._controller.wait(timeout_ms=5_000)
        super().closeEvent(event)  # type: ignore[arg-type]

    # ------------------------------------------------------------------
    # Private slots
    # ------------------------------------------------------------------

    def _on_error(self, message: str) -> None:
        LOGGER.error("Runtime error: %s", message)
        self.close()
