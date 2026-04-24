from __future__ import annotations

import time

import cv2

from maestro.domain.entities import FramePacket


class OpenCVFrameSource:
    def __init__(
        self,
        camera_index: int = 0,
        frame_width: int = 1280,
        frame_height: int = 720,
        target_fps: int = 30,
    ) -> None:
        self._frame_id = 0
        self._target_fps = target_fps
        self._frame_period_sec = 1.0 / target_fps if target_fps > 0 else 0.0
        self._last_read_time = 0.0

        self._camera = cv2.VideoCapture(camera_index)
        if not self._camera.isOpened():
            raise RuntimeError(f"Failed to open camera at index {camera_index}")

        self._camera.set(cv2.CAP_PROP_FRAME_WIDTH, frame_width)
        self._camera.set(cv2.CAP_PROP_FRAME_HEIGHT, frame_height)
        self._camera.set(cv2.CAP_PROP_FPS, target_fps)
        self._camera.set(cv2.CAP_PROP_BUFFERSIZE, 1)

    def read(self) -> FramePacket:
        # Keep read cadence close to target FPS to avoid busy-looping.
        now = time.perf_counter()
        elapsed = now - self._last_read_time
        if self._last_read_time > 0 and elapsed < self._frame_period_sec:
            time.sleep(self._frame_period_sec - elapsed)

        ok, frame = self._camera.read()
        if not ok or frame is None:
            raise RuntimeError("Failed to capture frame from camera")

        packet = FramePacket(
            frame=frame,
            timestamp_ms=int(time.time() * 1000),
            frame_id=self._frame_id,
        )
        self._frame_id += 1
        self._last_read_time = time.perf_counter()
        return packet

    def close(self) -> None:
        if self._camera is not None:
            self._camera.release()
