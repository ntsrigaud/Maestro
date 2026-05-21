from __future__ import annotations

import logging
import queue
import threading
import time
from collections.abc import Callable
from typing import Protocol

import numpy as np
import torch

from maestro.domain.entities import Prediction, RuntimeState, SequenceWindow, StabilizedPrediction
from maestro.evaluation.benchmark_latency import (
    LatencyBenchmark,
    write_latency_report,
)
from maestro.infrastructure.dispatch.app_adapters.keyboard_adapter import (
    KeyboardAdapter,
)
from maestro.infrastructure.model.checkpoint_loader import InferenceArtifact
from maestro.interfaces.command_dispatcher import ICommandDispatcher
from maestro.interfaces.feature_transformer import IFeatureTransformer
from maestro.interfaces.feedback_renderer import IFeedbackRenderer
from maestro.interfaces.frame_source import IFrameSource
from maestro.interfaces.landmark_extractor import IHandLandmarkExtractor
from maestro.interfaces.sequence_buffer import ISequenceBuffer
from maestro.pipeline.activation import GestureActivationController
from maestro.pipeline.annotation_pen import AnnotationPenTracker
from maestro.pipeline.hand_selection import HandSelector
from maestro.pipeline.laser_pointer import LaserPointerTracker
from maestro.pipeline.stabilization import PredictionStabilizer
from maestro.utils.schema_validation import RuntimeConfig

LOGGER = logging.getLogger(__name__)


class _InferenceWorker:
    """Background worker that runs model forward passes on queued windows."""

    def __init__(self, artifact: InferenceArtifact) -> None:
        self._artifact = artifact
        self._model = artifact.model
        self._device = next(self._model.parameters()).device
        self._model.eval()

        self._in_queue: queue.Queue[SequenceWindow] = queue.Queue(maxsize=1)
        self._out_queue: queue.Queue[tuple[Prediction, float]] = queue.Queue(maxsize=1)
        self._stop = threading.Event()
        self._thread = threading.Thread(target=self._run, name="inference-worker", daemon=True)

    def start(self) -> None:
        self._thread.start()

    def stop(self) -> None:
        self._stop.set()
        self.clear()
        self._thread.join(timeout=1.0)

    def submit(self, sequence: SequenceWindow) -> None:
        if self._stop.is_set():
            return

        try:
            self._in_queue.put_nowait(sequence)
        except queue.Full:
            try:
                self._in_queue.get_nowait()
            except queue.Empty:
                pass
            try:
                self._in_queue.put_nowait(sequence)
            except queue.Full:
                # If another producer beat us, keep the freshest entry already queued.
                pass

    def poll(self) -> tuple[Prediction, float] | None:
        try:
            return self._out_queue.get_nowait()
        except queue.Empty:
            return None

    def clear(self) -> None:
        while True:
            try:
                self._in_queue.get_nowait()
            except queue.Empty:
                break
        while True:
            try:
                self._out_queue.get_nowait()
            except queue.Empty:
                break

    def _run(self) -> None:
        while not self._stop.is_set():
            try:
                sequence = self._in_queue.get(timeout=0.05)
            except queue.Empty:
                continue

            infer_start = time.perf_counter()
            try:
                features = sequence.features.astype(np.float32, copy=False)
                tensor = torch.from_numpy(features).unsqueeze(0).to(self._device)
                with torch.no_grad():
                    logits = self._model(tensor)
                    probs = torch.softmax(logits, dim=1).squeeze(0).cpu().numpy()

                labels = self._artifact.class_labels
                class_probs = {label: float(probs[idx]) for idx, label in enumerate(labels)}
                top_idx = int(np.argmax(probs))
                prediction = Prediction(
                    class_probs=class_probs,
                    top_class=labels[top_idx],
                    confidence=float(probs[top_idx]),
                )
                infer_ms = (time.perf_counter() - infer_start) * 1000.0

                try:
                    self._out_queue.put_nowait((prediction, infer_ms))
                except queue.Full:
                    try:
                        self._out_queue.get_nowait()
                    except queue.Empty:
                        pass
                    try:
                        self._out_queue.put_nowait((prediction, infer_ms))
                    except queue.Full:
                        pass
            except Exception:  # noqa: BLE001
                LOGGER.exception("InferenceWorker: error during inference — skipping window")


class RuntimeDependencies(Protocol):
    config: RuntimeConfig
    frame_source: IFrameSource
    landmark_extractor: IHandLandmarkExtractor
    hand_selector: HandSelector
    feature_transformer: IFeatureTransformer
    sequence_buffer: ISequenceBuffer
    model_artifact: InferenceArtifact
    stabilizer: PredictionStabilizer
    activation_controller: GestureActivationController
    dispatcher: ICommandDispatcher
    overlay_renderer: IFeedbackRenderer
    benchmark: LatencyBenchmark
    show_window: bool
    label_remap: dict[str, str]


class Runtime:
    def __init__(
        self,
        container: RuntimeDependencies,
        on_state: Callable[[RuntimeState], None] | None = None,
    ) -> None:
        self._container = container
        self._running = False
        self._on_state = on_state
        self._inference_worker: _InferenceWorker | None = None
        cfg = getattr(container, "config", None)
        infer_cfg = getattr(cfg, "inference", None)
        # Pre-compute the active mode display string from config.  This never
        # changes at runtime; storing it once avoids repeated attribute lookups
        # and getattr fallbacks in the hot loop.
        app_cfg = container.config.app
        nav_mode = getattr(app_cfg, "navigation_mode", "workspace")
        self._active_mode = f"desktop\u00b7{nav_mode}"
        # Continuous gesture trackers — stateful per-frame handlers for
        # pointing gestures.  Created here (not in AppContainer) because they
        # are internal runtime concerns rather than injected dependencies.
        #   point_one  → laser pointer (⌘+L / Ctrl+L)
        #   point_two  → annotation pen (⌘+P / Ctrl+P)
        mirror_x = getattr(app_cfg, "mirror_display", True)
        pointer_cfg = getattr(infer_cfg, "pointer", None)
        pointer_gain_x = getattr(pointer_cfg, "gain_x", 2.4)
        pointer_gain_y = getattr(pointer_cfg, "gain_y", 2.2)
        self._laser_tracker = LaserPointerTracker(
            keyboard=KeyboardAdapter(),
            mirror_x=mirror_x,
            pointer_gain_x=pointer_gain_x,
            pointer_gain_y=pointer_gain_y,
        )
        self._pen_tracker = AnnotationPenTracker(
            keyboard=KeyboardAdapter(),
            mirror_x=mirror_x,
            pointer_gain_x=pointer_gain_x,
            pointer_gain_y=pointer_gain_y,
        )

    def stop(self) -> None:
        self._running = False

    def set_pointer_gain(self, gain_x: float, gain_y: float) -> None:
        """Update pointer/pen movement gain on both continuous trackers.

        Safe to call from the main thread while the runtime loop runs in the
        worker thread — Python's GIL makes the underlying float attribute
        writes atomic (same guarantee as ``Runtime.stop()``).
        """
        self._laser_tracker.set_pointer_gain(gain_x, gain_y)
        self._pen_tracker.set_pointer_gain(gain_x, gain_y)

    def run(
        self,
        benchmark_output: str | None = None,
        max_frames: int = 0,
    ) -> None:
        app_name = self._container.config.app.app_name
        artifact = getattr(self._container, "model_artifact", None)
        artifact_path = getattr(artifact, "artifact_path", "<unavailable>")
        benchmark = getattr(self._container, "benchmark", None)

        LOGGER.info("Starting runtime for %s", app_name)
        LOGGER.info("Loaded model artifact: %s", artifact_path)
        self._running = True
        if artifact is not None:
            self._inference_worker = _InferenceWorker(artifact)
            self._inference_worker.start()

        processed_frames = 0
        valid_landmarks = 0
        emitted_windows = 0
        start_time = time.perf_counter()
        inference_ms = 0.0
        latest_label = "unknown"
        latest_conf = 0.0
        _consecutive_frame_errors = 0
        _MAX_CONSECUTIVE_FRAME_ERRORS = 10

        try:
            while self._running:
                loop_start = time.perf_counter()

                t0 = time.perf_counter()
                try:
                    frame = self._container.frame_source.read()
                    _consecutive_frame_errors = 0
                except RuntimeError as _frame_exc:
                    _consecutive_frame_errors += 1
                    if _consecutive_frame_errors >= _MAX_CONSECUTIVE_FRAME_ERRORS:
                        LOGGER.error(
                            "Camera failed %d consecutive times — stopping runtime",
                            _MAX_CONSECUTIVE_FRAME_ERRORS,
                        )
                        break
                    LOGGER.warning(
                        "Frame capture failed (transient, %d/%d): %s",
                        _consecutive_frame_errors,
                        _MAX_CONSECUTIVE_FRAME_ERRORS,
                        _frame_exc,
                    )
                    continue
                if benchmark is not None:
                    benchmark.record_capture((time.perf_counter() - t0) * 1000.0)

                t1 = time.perf_counter()
                multi_packet = self._container.landmark_extractor.extract(frame)
                landmark_packet = self._container.hand_selector.select(multi_packet)
                if benchmark is not None:
                    benchmark.record_extraction((time.perf_counter() - t1) * 1000.0)

                t2 = time.perf_counter()
                feature_frame = self._container.feature_transformer.transform(landmark_packet)
                if benchmark is not None:
                    benchmark.record_feature((time.perf_counter() - t2) * 1000.0)

                self._container.sequence_buffer.push(feature_frame)

                processed_frames += 1
                if landmark_packet.valid:
                    valid_landmarks += 1

                stabilized_prediction = None
                if self._container.sequence_buffer.ready():
                    window = self._container.sequence_buffer.window()
                    emitted_windows += 1
                    if self._inference_worker is not None:
                        self._inference_worker.submit(window)

                # Advance the cooldown clock unconditionally, before polling the
                # inference worker.  This ensures the COOLDOWN → ARMED transition
                # fires even when no inference windows are available (e.g. the hand
                # has left the camera frame).  The label is reset here so the UI
                # immediately reflects the ARMED state; if a fresh inference result
                # arrives in the same iteration it will overwrite the reset below.
                if self._container.activation_controller.tick():
                    latest_label = "unknown"
                    latest_conf = 0.0

                worker_result = self._inference_worker.poll() if self._inference_worker else None
                if worker_result is not None:
                    prediction, inference_ms = worker_result
                    if benchmark is not None:
                        benchmark.record_inference(inference_ms)

                    stabilized_prediction = self._container.stabilizer.update(prediction)
                    # Apply label remapping (e.g. swipe direction correction for
                    # mirror-view training data).  Remapping runs after stabilization
                    # so that threshold/voting decisions use the raw model labels;
                    # the remapped label flows to command dispatch and UI display.
                    _remap = self._container.label_remap
                    if _remap and stabilized_prediction.label in _remap:
                        stabilized_prediction = StabilizedPrediction(
                            label=_remap[stabilized_prediction.label],
                            confidence=stabilized_prediction.confidence,
                            accepted=stabilized_prediction.accepted,
                            reason=stabilized_prediction.reason,
                        )
                    latest_label = stabilized_prediction.label
                    latest_conf = stabilized_prediction.confidence

                    LOGGER.debug(
                        "stabilizer: label=%r conf=%.2f accepted=%s reason=%s",
                        stabilized_prediction.label,
                        stabilized_prediction.confidence,
                        stabilized_prediction.accepted,
                        stabilized_prediction.reason,
                    )

                    command = self._container.activation_controller.route(stabilized_prediction)
                    if command is not None:
                        LOGGER.info(
                            "command: id=%r source_gesture=%r",
                            command.command_id,
                            command.source_gesture,
                        )
                        result = self._container.dispatcher.dispatch(command)
                        LOGGER.info(
                            "dispatch: success=%s detail=%s",
                            result.success,
                            result.detail,
                        )

                # Continuous gesture trackers — update every frame using the
                # current landmarks and latest stable label.  These drive
                # real-time cursor tracking independently of the edge-trigger
                # cooldown: point_one → laser pointer, point_two → annotation pen.
                self._laser_tracker.update(
                    label=latest_label,
                    landmarks=landmark_packet.landmarks if landmark_packet.valid else None,
                    valid=landmark_packet.valid,
                )
                self._pen_tracker.update(
                    label=latest_label,
                    landmarks=landmark_packet.landmarks if landmark_packet.valid else None,
                    valid=landmark_packet.valid,
                )

                elapsed = time.perf_counter() - start_time
                fps = (processed_frames / elapsed) if elapsed > 0 else 0.0
                mode = str(self._container.activation_controller.mode)
                valid_ratio = self._container.sequence_buffer.current_valid_ratio()

                loop_e2e_ms = (time.perf_counter() - loop_start) * 1000.0
                if benchmark is not None:
                    benchmark.record_end_to_end(loop_e2e_ms)

                window_ready = self._container.sequence_buffer.ready()
                landmarks_info = landmark_packet.landmarks  # type: ignore[union-attr]

                state = RuntimeState(
                    mode=mode,
                    last_label=latest_label,
                    last_confidence=latest_conf,
                    active_mode=self._active_mode,
                    info={
                        "landmarks": landmarks_info,
                        "window_ready": window_ready,
                        "valid_ratio": valid_ratio,
                        "latency_infer_ms": inference_ms,
                        "latency_e2e_ms": loop_e2e_ms,
                        "fps": fps,
                        "accepted": (
                            bool(stabilized_prediction.accepted) if stabilized_prediction else False
                        ),
                        "laser_active": self._laser_tracker.active,
                        "pen_active": self._pen_tracker.active,
                    },
                )

                overlay_frame = self._container.overlay_renderer.render(frame, state)
                # Attach the rendered frame before emitting state so consumers
                # (GUI PreviewWidget via Qt signal, CLI on_state callback) can
                # display it in their own thread-safe context.
                if self._container.show_window:
                    state.preview_frame = overlay_frame.frame

                if self._on_state is not None:
                    self._on_state(state)

                if processed_frames % 30 == 0:
                    LOGGER.info(
                        "Processed %d frames, valid landmarks %d, windows %d, fps %.2f",
                        processed_frames,
                        valid_landmarks,
                        emitted_windows,
                        fps,
                    )

                if max_frames > 0 and processed_frames >= max_frames:
                    self.stop()
        except KeyboardInterrupt:
            LOGGER.info("Keyboard interrupt received. Shutting down runtime.")
        finally:
            total_seconds = max(time.perf_counter() - start_time, 1e-9)
            if benchmark is not None:
                report = benchmark.report(
                    total_frames=processed_frames,
                    total_seconds=total_seconds,
                )
                if benchmark_output:
                    output_path = write_latency_report(report, benchmark_output)
                    LOGGER.info("Wrote latency benchmark report to %s", output_path)

            self._container.frame_source.close()
            self._container.landmark_extractor.close()
            if self._inference_worker is not None:
                self._inference_worker.stop()
                self._inference_worker = None
            self._laser_tracker.deactivate()
            self._pen_tracker.deactivate()
            self._running = False
            LOGGER.info("Runtime shutdown complete.")
