from __future__ import annotations

import logging

from gesture_control.app.bootstrap import AppContainer

LOGGER = logging.getLogger(__name__)


class Runtime:
    def __init__(self, container: AppContainer) -> None:
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

        try:
            while self._running:
                frame = self._container.frame_source.read()
                landmark_packet = self._container.landmark_extractor.extract(frame)

                processed_frames += 1
                if landmark_packet.valid:
                    valid_landmarks += 1

                if processed_frames % 30 == 0:
                    LOGGER.info(
                        "Processed %d frames, valid landmarks %d",
                        processed_frames,
                        valid_landmarks,
                    )
        except KeyboardInterrupt:
            LOGGER.info("Keyboard interrupt received. Shutting down runtime.")
        finally:
            self._container.frame_source.close()
            self._container.landmark_extractor.close()
            self._running = False
            LOGGER.info("Runtime shutdown complete.")
