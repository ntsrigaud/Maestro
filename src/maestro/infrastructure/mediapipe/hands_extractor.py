from __future__ import annotations

import math

import cv2
import mediapipe as mp

from maestro.domain.entities import FramePacket, HandLandmarks, MultiHandPacket


class MediaPipeHandsExtractor:
    """MediaPipe Hands wrapper that extracts landmarks for all detected hands.

    The extractor is deliberately preference-agnostic: it always requests both
    hands from MediaPipe and returns a :class:`MultiHandPacket` containing every
    detected hand.  Hand-preference selection (left / right / both) is the
    responsibility of the downstream :class:`~maestro.pipeline.HandSelector`
    step so that the extractor remains stateless and testable in isolation.
    """

    def __init__(
        self,
        static_image_mode: bool = False,
        min_detection_confidence: float = 0.7,
        min_tracking_confidence: float = 0.5,
    ) -> None:
        self._hands = mp.solutions.hands.Hands(
            static_image_mode=static_image_mode,
            max_num_hands=2,
            min_detection_confidence=min_detection_confidence,
            min_tracking_confidence=min_tracking_confidence,
        )

    def extract(self, frame: FramePacket) -> MultiHandPacket:
        # cv2.cvtColor produces a C-contiguous RGB copy; a reversed-stride
        # view (frame[:, :, ::-1]) is not C-contiguous and can cause silent
        # failures in MediaPipe's C extensions.
        rgb_frame = cv2.cvtColor(frame.frame, cv2.COLOR_BGR2RGB)
        result = self._hands.process(rgb_frame)

        if not result.multi_hand_landmarks or not result.multi_handedness:
            return MultiHandPacket(hands=[], timestamp_ms=frame.timestamp_ms)

        hands: list[HandLandmarks] = []
        for handedness_proto, landmark_proto in zip(
            result.multi_handedness, result.multi_hand_landmarks
        ):
            cls = handedness_proto.classification[0]
            raw_landmarks = landmark_proto.landmark

            coords_list: list[list[float]] = []
            valid_points = 0
            for point in raw_landmarks:
                coords = [float(point.x), float(point.y), float(point.z)]
                coords_list.append(coords)
                if all(not math.isnan(v) for v in coords):
                    valid_points += 1

            track_conf = valid_points / 21.0 if coords_list else 0.0
            hands.append(
                HandLandmarks(
                    label=cls.label,
                    landmarks=coords_list,
                    detect_conf=float(cls.score),
                    track_conf=track_conf,
                )
            )

        return MultiHandPacket(hands=hands, timestamp_ms=frame.timestamp_ms)

    def close(self) -> None:
        self._hands.close()


