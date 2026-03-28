from __future__ import annotations

from typing import Any

import cv2

from gesture_control.domain.entities import FramePacket, RuntimeState


class OverlayRenderer:
    """Renders non-blocking runtime diagnostics onto frames."""

    def __init__(self) -> None:
        self._font = cv2.FONT_HERSHEY_SIMPLEX

    def render(self, frame: FramePacket, state: RuntimeState) -> FramePacket:
        canvas = frame.frame.copy()
        info = state.info

        self._draw_landmarks(canvas, info.get("landmarks"))

        line1 = f"mode={state.mode}"
        line2 = f"pred={state.last_label} conf={state.last_confidence:.2f}"
        line3 = (
            "window="
            f"{info.get('window_ready', False)} "
            f"ratio={float(info.get('valid_ratio', 0.0)):.2f}"
        )
        line4 = (
            f"latency_ms e2e={float(info.get('latency_e2e_ms', 0.0)):.1f} "
            f"infer={float(info.get('latency_infer_ms', 0.0)):.1f}"
        )
        line5 = f"fps={float(info.get('fps', 0.0)):.1f}"

        self._put(canvas, line1, 12, 24)
        self._put(canvas, line2, 12, 48)
        self._put(canvas, line3, 12, 72)
        self._put(canvas, line4, 12, 96)
        self._put(canvas, line5, 12, 120)

        return FramePacket(frame=canvas, timestamp_ms=frame.timestamp_ms, frame_id=frame.frame_id)

    def _draw_landmarks(self, image: Any, landmarks: Any) -> None:
        if not isinstance(landmarks, list):
            return

        height, width = image.shape[:2]
        for point in landmarks:
            if not isinstance(point, list) or len(point) < 2:
                continue
            x = int(float(point[0]) * width)
            y = int(float(point[1]) * height)
            if x < 0 or y < 0 or x >= width or y >= height:
                continue
            cv2.circle(image, (x, y), 2, (0, 255, 0), -1)

    def _put(self, image: Any, text: str, x: int, y: int) -> None:
        cv2.putText(image, text, (x, y), self._font, 0.6, (0, 255, 255), 2, cv2.LINE_AA)
