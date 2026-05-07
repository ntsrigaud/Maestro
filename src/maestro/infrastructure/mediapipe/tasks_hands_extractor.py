from __future__ import annotations

import logging
import math
from pathlib import Path

import cv2
import mediapipe as mp
from mediapipe.tasks.python import BaseOptions
from mediapipe.tasks.python import vision

from maestro.domain.entities import FramePacket, HandLandmarks, MultiHandPacket

LOGGER = logging.getLogger(__name__)


class MediaPipeTasksHandsExtractor:
    """MediaPipe Tasks hand extractor with optional GPU delegate.

    Notes:
    - Tasks GPU path on macOS requires SRGBA input for stable operation.
    - If model file is missing or delegate initialization fails, callers should
      fall back to the classic solutions-based extractor.
    """

    def __init__(
        self,
        model_path: str | Path,
        *,
        use_gpu_delegate: bool = True,
        min_detection_confidence: float = 0.7,
        min_tracking_confidence: float = 0.5,
    ) -> None:
        model_path = Path(model_path)
        if not model_path.exists():
            raise FileNotFoundError(f"Hand landmarker task file not found: {model_path}")

        delegate = BaseOptions.Delegate.GPU if use_gpu_delegate else BaseOptions.Delegate.CPU
        options = vision.HandLandmarkerOptions(
            base_options=BaseOptions(
                model_asset_path=str(model_path),
                delegate=delegate,
            ),
            running_mode=vision.RunningMode.VIDEO,
            num_hands=2,
            min_hand_detection_confidence=min_detection_confidence,
            min_tracking_confidence=min_tracking_confidence,
        )
        self._detector = vision.HandLandmarker.create_from_options(options)
        self._last_ts_ms = 0

    def extract(self, frame: FramePacket) -> MultiHandPacket:
        rgb = frame.rgb_frame if frame.rgb_frame is not None else cv2.cvtColor(frame.frame, cv2.COLOR_BGR2RGB)
        rgba = cv2.cvtColor(rgb, cv2.COLOR_RGB2RGBA)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGBA, data=rgba)

        ts_ms = max(int(frame.timestamp_ms), self._last_ts_ms + 1)
        self._last_ts_ms = ts_ms
        result = self._detector.detect_for_video(mp_image, timestamp_ms=ts_ms)

        if not result.hand_landmarks:
            return MultiHandPacket(hands=[], timestamp_ms=frame.timestamp_ms)

        hands: list[HandLandmarks] = []
        handedness_groups = result.handedness or []
        for idx, landmarks in enumerate(result.hand_landmarks):
            label = "Unknown"
            detect_conf = 0.0
            if idx < len(handedness_groups) and handedness_groups[idx]:
                category = handedness_groups[idx][0]
                label = str(category.category_name)
                detect_conf = float(category.score)

            coords_list: list[list[float]] = []
            valid_points = 0
            for point in landmarks:
                coords = [float(point.x), float(point.y), float(point.z)]
                coords_list.append(coords)
                if all(not math.isnan(v) for v in coords):
                    valid_points += 1

            track_conf = valid_points / 21.0 if coords_list else 0.0
            hands.append(
                HandLandmarks(
                    label=label,
                    landmarks=coords_list,
                    detect_conf=detect_conf,
                    track_conf=track_conf,
                )
            )

        return MultiHandPacket(hands=hands, timestamp_ms=frame.timestamp_ms)

    def close(self) -> None:
        self._detector.close()
