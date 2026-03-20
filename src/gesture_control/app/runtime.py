from __future__ import annotations

import logging
from typing import Protocol

from gesture_control.interfaces.feature_transformer import IFeatureTransformer
from gesture_control.interfaces.frame_source import IFrameSource
from gesture_control.interfaces.landmark_extractor import IHandLandmarkExtractor
from gesture_control.interfaces.sequence_buffer import ISequenceBuffer
from gesture_control.utils.schema_validation import RuntimeConfig

LOGGER = logging.getLogger(__name__)


class RuntimeDependencies(Protocol):
    config: RuntimeConfig
    frame_source: IFrameSource
    landmark_extractor: IHandLandmarkExtractor
    feature_transformer: IFeatureTransformer
    sequence_buffer: ISequenceBuffer


class Runtime:
    def __init__(self, container: RuntimeDependencies) -> None:
        self._container = container
        self._running = False

    def stop(self) -> None:
        self._running = False

    def run(self) -> None:
        app_name = self._container.config.app.app_name
        LOGGER.info("Starting runtime for %s", app_name)
        self._running = True

        processed_frames = 0
        valid_landmarks = 0
        emitted_windows = 0

        try:
            while self._running:
                frame = self._container.frame_source.read()
                landmark_packet = self._container.landmark_extractor.extract(frame)
                feature_frame = self._container.feature_transformer.transform(landmark_packet)
                self._container.sequence_buffer.push(feature_frame)

                processed_frames += 1
                if landmark_packet.valid:
                    valid_landmarks += 1
                if self._container.sequence_buffer.ready():
                    self._container.sequence_buffer.window()
                    emitted_windows += 1

                if processed_frames % 30 == 0:
                    LOGGER.info(
                        "Processed %d frames, valid landmarks %d, windows %d",
                        processed_frames,
                        valid_landmarks,
                        emitted_windows,
                    )
        except KeyboardInterrupt:
            LOGGER.info("Keyboard interrupt received. Shutting down runtime.")
        finally:
            self._container.frame_source.close()
            self._container.landmark_extractor.close()
            self._running = False
            LOGGER.info("Runtime shutdown complete.")
