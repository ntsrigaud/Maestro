from __future__ import annotations

import math

import mediapipe as mp

from gesture_control.domain.entities import FramePacket, LandmarkPacket

# Maps the user-facing hand preference (from the user's physical viewpoint) to
# the MediaPipe handedness label expected in the raw, non-mirrored camera frame.
# MediaPipe assumes a selfie/mirrored view when assigning labels, so in a standard
# unmirrored webcam frame the labels are inverted relative to the user's own hands:
#   user's right hand → MediaPipe reports "Left"
#   user's left hand  → MediaPipe reports "Right"
# "both" maps to None, meaning no label filtering is applied.
_USER_TO_MEDIAPIPE_LABEL: dict[str, str | None] = {
    "right": "left",
    "left": "right",
    "both": None,
}


class MediaPipeHandsExtractor:
    def __init__(
        self,
        static_image_mode: bool = False,
        max_num_hands: int = 1,
        min_detection_confidence: float = 0.7,
        min_tracking_confidence: float = 0.5,
        hand_preference: str = "both",
    ) -> None:
        self._min_detection_confidence = min_detection_confidence
        self._min_tracking_confidence = min_tracking_confidence
        # Translate the user-facing preference to the MediaPipe label that will
        # appear in the raw (non-mirrored) frame.  None means no filtering.
        self._required_label = _USER_TO_MEDIAPIPE_LABEL[hand_preference.lower()]
        self._hands = mp.solutions.hands.Hands(
            static_image_mode=static_image_mode,
            max_num_hands=max_num_hands,
            min_detection_confidence=min_detection_confidence,
            min_tracking_confidence=min_tracking_confidence,
        )

    def extract(self, frame: FramePacket) -> LandmarkPacket:
        rgb_frame = frame.frame[:, :, ::-1]
        result = self._hands.process(rgb_frame)

        if not result.multi_hand_landmarks or not result.multi_handedness:
            return LandmarkPacket(
                landmarks=[],
                handedness=None,
                detect_conf=0.0,
                track_conf=0.0,
                timestamp_ms=frame.timestamp_ms,
                valid=False,
            )

        handedness = result.multi_handedness[0].classification[0]
        landmark_list = result.multi_hand_landmarks[0].landmark

        landmarks: list[list[float]] = []
        valid_points = 0
        for point in landmark_list:
            coords = [float(point.x), float(point.y), float(point.z)]
            landmarks.append(coords)
            if all(not math.isnan(v) for v in coords):
                valid_points += 1

        detect_conf = float(handedness.score)
        track_conf = valid_points / 21.0 if landmarks else 0.0
        preference_ok = (
            self._required_label is None
            or handedness.label.lower() == self._required_label
        )
        is_valid = (
            detect_conf >= self._min_detection_confidence
            and track_conf >= self._min_tracking_confidence
            and preference_ok
        )

        return LandmarkPacket(
            landmarks=landmarks,
            handedness=handedness.label,
            detect_conf=detect_conf,
            track_conf=track_conf,
            timestamp_ms=frame.timestamp_ms,
            valid=is_valid,
        )

    def close(self) -> None:
        self._hands.close()
