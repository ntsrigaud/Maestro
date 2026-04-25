from __future__ import annotations

import logging
import time
from collections.abc import Callable
from typing import Protocol

import numpy as np
import torch

from maestro.domain.entities import Prediction, RuntimeState, SequenceWindow
from maestro.evaluation.benchmark_latency import (
    LatencyBenchmark,
    write_latency_report,
)
from maestro.infrastructure.model.checkpoint_loader import InferenceArtifact
from maestro.interfaces.command_dispatcher import ICommandDispatcher
from maestro.interfaces.feature_transformer import IFeatureTransformer
from maestro.interfaces.feedback_renderer import IFeedbackRenderer
from maestro.interfaces.frame_source import IFrameSource
from maestro.interfaces.gaze_detector import IGazeIntentDetector
from maestro.interfaces.landmark_extractor import IHandLandmarkExtractor
from maestro.interfaces.sequence_buffer import ISequenceBuffer
from maestro.pipeline.activation import GestureActivationController
from maestro.pipeline.hand_selection import HandSelector
from maestro.pipeline.stabilization import PredictionStabilizer
from maestro.utils.schema_validation import RuntimeConfig

LOGGER = logging.getLogger(__name__)


class RuntimeDependencies(Protocol):
    config: RuntimeConfig
    frame_source: IFrameSource
    gaze_detector: IGazeIntentDetector
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


class Runtime:
    def __init__(
        self,
        container: RuntimeDependencies,
        on_state: Callable[[RuntimeState], None] | None = None,
    ) -> None:
        self._container = container
        self._running = False
        self._on_state = on_state
        # Pre-compute the active mode display string from config.  This never
        # changes at runtime; storing it once avoids repeated attribute lookups
        # and getattr fallbacks in the hot loop.
        app_cfg = container.config.app
        nav_mode = getattr(app_cfg, "navigation_mode", "workspace")
        self._active_mode = f"desktop\u00b7{nav_mode}"

    def stop(self) -> None:
        self._running = False

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

        processed_frames = 0
        valid_landmarks = 0
        emitted_windows = 0
        start_time = time.perf_counter()
        inference_ms = 0.0
        latest_label = "unknown"
        latest_conf = 0.0
        _prev_gaze_permitted: bool = False

        try:
            while self._running:
                loop_start = time.perf_counter()

                t0 = time.perf_counter()
                frame = self._container.frame_source.read()
                if benchmark is not None:
                    benchmark.record_capture((time.perf_counter() - t0) * 1000.0)

                # Stage 2 — gaze gate: bypass the downstream pipeline when
                # the user is not directing attention toward the camera.
                gaze_permitted = self._container.gaze_detector.update(frame)

                # Rising edge: gaze just became permitted → transition IDLE → ARMED.
                if gaze_permitted and not _prev_gaze_permitted:
                    self._container.activation_controller.rearm()
                _prev_gaze_permitted = gaze_permitted

                if not gaze_permitted:
                    absent_state = RuntimeState(
                        mode=str(self._container.activation_controller.mode),
                        gaze_permitted=False,
                        active_mode=self._active_mode,
                    )
                    overlay_frame = self._container.overlay_renderer.render(frame, absent_state)
                    # Attach the rendered frame so the caller (GUI widget or
                    # CLI on_state callback) can display it in its own
                    # thread-safe context.  overlay_frame.frame is a fresh
                    # canvas copy produced by OverlayRenderer, so no
                    # additional copy is needed.
                    if self._container.show_window:
                        absent_state.preview_frame = overlay_frame.frame
                    processed_frames += 1
                    if self._on_state is not None:
                        self._on_state(absent_state)
                    if max_frames > 0 and processed_frames >= max_frames:
                        self.stop()
                    continue

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

                    infer_start = time.perf_counter()
                    prediction = self._predict(window)
                    inference_ms = (time.perf_counter() - infer_start) * 1000.0
                    if benchmark is not None:
                        benchmark.record_inference(inference_ms)

                    stabilized_prediction = self._container.stabilizer.update(prediction)
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

                elapsed = time.perf_counter() - start_time
                fps = (processed_frames / elapsed) if elapsed > 0 else 0.0
                mode = str(self._container.activation_controller.mode)
                valid_ratio = self._container.sequence_buffer.current_valid_ratio()

                loop_e2e_ms = (time.perf_counter() - loop_start) * 1000.0
                if benchmark is not None:
                    benchmark.record_end_to_end(loop_e2e_ms)

                state = RuntimeState(
                    mode=mode,
                    last_label=latest_label,
                    last_confidence=latest_conf,
                    gaze_permitted=gaze_permitted,
                    active_mode=self._active_mode,
                    info={
                        "landmarks": landmark_packet.landmarks,
                        "window_ready": self._container.sequence_buffer.ready(),
                        "valid_ratio": valid_ratio,
                        "latency_infer_ms": inference_ms,
                        "latency_e2e_ms": loop_e2e_ms,
                        "fps": fps,
                        "accepted": (
                            bool(stabilized_prediction.accepted) if stabilized_prediction else False
                        ),
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
            self._container.gaze_detector.close()
            self._container.landmark_extractor.close()
            self._running = False
            LOGGER.info("Runtime shutdown complete.")

    def _predict(self, sequence: SequenceWindow) -> Prediction:
        features = np.asarray(sequence.features, dtype=np.float32)
        tensor = torch.from_numpy(features).unsqueeze(0)

        model = self._container.model_artifact.model
        device = next(model.parameters()).device
        tensor = tensor.to(device)

        with torch.no_grad():
            logits = model(tensor)
            probs = torch.softmax(logits, dim=1).squeeze(0).cpu().numpy()

        labels = self._container.model_artifact.class_labels
        class_probs = {label: float(probs[idx]) for idx, label in enumerate(labels)}
        top_idx = int(np.argmax(probs))
        return Prediction(
            class_probs=class_probs,
            top_class=labels[top_idx],
            confidence=float(probs[top_idx]),
        )
