from __future__ import annotations

import logging

from PySide6.QtCore import QObject, QThread, Signal

from maestro.app.bootstrap import build_container
from maestro.app.runtime import Runtime
from maestro.domain.entities import RuntimeState

LOGGER = logging.getLogger(__name__)

# Emit the state signal at most once every N frames.
# At 30 FPS with N=3, UI updates are capped at ~10 Hz.
_EMIT_EVERY_DEFAULT = 3


class _RuntimeThread(QThread):
    """
    Private worker thread.

    Lifecycle
    ---------
    QThread.started  → Qt emits automatically as thread begins
    build_succeeded  → emitted after build_container() returns successfully
    state_ready      → emitted per throttled frame during Runtime.run()
    error_emitted    → emitted on any unhandled exception
    QThread.finished → Qt emits automatically when run() returns
    """

    state_ready: Signal = Signal(object)  # RuntimeState payload
    build_succeeded: Signal = Signal()
    error_emitted: Signal = Signal(str)

    def __init__(
        self,
        config_path: str,
        artifact_path: str | None,
        hf_revision_override: str | None,
        hand_preference_override: str | None,
        show_window: bool,
        minimal_preview: bool,
        emit_every: int,
        parent: QObject | None = None,
    ) -> None:
        super().__init__(parent)
        self._config_path = config_path
        self._artifact_path = artifact_path
        self._hf_revision_override = hf_revision_override
        self._hand_preference_override = hand_preference_override
        self._show_window = show_window
        self._minimal_preview = minimal_preview
        self._emit_every = emit_every
        self._frame_counter = 0
        self._runtime: Runtime | None = None

    # ------------------------------------------------------------------
    # QThread interface
    # ------------------------------------------------------------------

    def run(self) -> None:
        """Execute in the worker thread. Must not be called from the main thread."""
        try:
            LOGGER.info("Worker: calling build_container(%s)", self._config_path)
            container = build_container(
                self._config_path,
                model_artifact_path=self._artifact_path,
                hf_revision_override=self._hf_revision_override,
                hand_preference_override=self._hand_preference_override,
                show_window=self._show_window,
                minimal_preview=self._minimal_preview,
            )
            self._runtime = Runtime(container, on_state=self._handle_state)
            self.build_succeeded.emit()
            LOGGER.info("Worker: starting Runtime.run()")
            self._runtime.run()
        except Exception as exc:  # noqa: BLE001
            LOGGER.exception("Worker error: %s", exc)
            self.error_emitted.emit(str(exc))
        finally:
            LOGGER.info("Worker: exiting")

    # ------------------------------------------------------------------
    # on_state callback — called from inside Runtime.run() (worker thread).
    # Throttle: emit the Qt signal every _emit_every frames only.
    # The frame counter is private to this thread; no locking needed.
    # ------------------------------------------------------------------

    def _handle_state(self, state: RuntimeState) -> None:
        self._frame_counter += 1
        if self._frame_counter % self._emit_every == 0:
            self.state_ready.emit(state)

    # ------------------------------------------------------------------
    # Thread-safe stop — safe to call from the main thread.
    # Runtime.stop() is a single boolean write, safe under the GIL.
    # ------------------------------------------------------------------

    def stop(self) -> None:
        if self._runtime is not None:
            self._runtime.stop()


class RuntimeController(QObject):
    """
    Owns the worker QThread and bridges Runtime state to Qt signals.

    All methods must be called from the main (Qt event loop) thread.
    Signals are delivered to the main thread via Qt's queued connection
    mechanism regardless of which thread emits them.

    Signal contracts
    ----------------
    state_updated(RuntimeState) — emitted per throttled frame while running
    runtime_started()           — emitted after build_container() succeeds
    runtime_stopped()           — emitted after Runtime.run() exits (any cause)
    error_occurred(str)         — emitted on unhandled exception in the worker
    """

    state_updated: Signal = Signal(object)  # RuntimeState
    runtime_started: Signal = Signal()
    runtime_stopped: Signal = Signal()
    error_occurred: Signal = Signal(str)

    def __init__(self, parent: QObject | None = None) -> None:
        super().__init__(parent)
        self._thread: _RuntimeThread | None = None

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def start(
        self,
        config_path: str,
        *,
        artifact_path: str | None = None,
        hf_revision_override: str | None = None,
        hand_preference_override: str | None = None,
        show_window: bool = False,
        minimal_preview: bool = False,
        emit_every: int = _EMIT_EVERY_DEFAULT,
    ) -> None:
        """Spawn the worker thread. No-op if already running."""
        if self._thread is not None and self._thread.isRunning():
            LOGGER.warning("RuntimeController.start() called while already running; ignoring")
            return

        self._thread = _RuntimeThread(
            config_path=config_path,
            artifact_path=artifact_path,
            hf_revision_override=hf_revision_override,
            hand_preference_override=hand_preference_override,
            show_window=show_window,
            minimal_preview=minimal_preview,
            emit_every=emit_every,
        )
        # Qt delivers cross-thread signals via the event queue automatically.
        self._thread.state_ready.connect(self.state_updated)
        self._thread.build_succeeded.connect(self.runtime_started)
        self._thread.finished.connect(self.runtime_stopped)
        self._thread.error_emitted.connect(self.error_occurred)
        self._thread.start()

    def stop(self) -> None:
        """Request a graceful shutdown. Returns immediately; does not block."""
        if self._thread is not None:
            self._thread.stop()

    def wait(self, timeout_ms: int = 5_000) -> bool:
        """
        Block the calling thread until the worker exits.

        Returns True if the thread exited within timeout_ms, False otherwise.
        Use this in teardown / test fixtures only — never from a UI slot.

        After the thread exits, calls deleteLater() to signal Qt to clean up the
        C++ QThread object, preventing abort() during application shutdown.
        """
        if self._thread is not None and self._thread.isRunning():
            exited = self._thread.wait(timeout_ms)
        else:
            exited = True

        # Always call deleteLater() after the thread is no longer running, even if
        # wait() timed out or the thread was never started. This ensures Qt properly
        # cleans up the QThread object instead of trying to destroy it in a destructor,
        # which triggers abort() during application shutdown.
        if self._thread is not None:
            self._thread.deleteLater()
            self._thread = None

        return exited

    def reset(self) -> None:
        """Reset the controller state (mainly for testing). Must not be called while running."""
        if self._thread is not None:
            if self._thread.isRunning():
                raise RuntimeError(
                    "RuntimeController.reset() called while thread is running. "
                    "Call stop() and wait() first."
                )
            self._thread.deleteLater()
            self._thread = None

    @property
    def is_running(self) -> bool:
        """True if the worker thread is alive."""
        return self._thread is not None and self._thread.isRunning()
