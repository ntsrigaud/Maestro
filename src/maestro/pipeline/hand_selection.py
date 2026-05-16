"""Hand-preference selection step.

``HandSelector`` sits between the landmark extractor and the feature
transformer.  The extractor is preference-agnostic and always returns all
detected hands (up to two) in a :class:`~maestro.domain.entities.MultiHandPacket`.
``HandSelector`` applies the user's configured ``hand_preference`` and produces
a single :class:`~maestro.domain.entities.LandmarkPacket` suitable for the
downstream feature transformer.

Handedness mapping rationale
-----------------------------
MediaPipe Hands assumes a selfie / mirrored view when assigning handedness
labels.  In a standard, unmirrored webcam frame the labels are therefore
inverted relative to the user's physical hands:

    user's right hand  →  MediaPipe label ``"Left"``
    user's left hand   →  MediaPipe label ``"Right"``

``HandSelector`` encodes this inversion so callers always work in the
user-facing frame of reference.
"""

from __future__ import annotations

from maestro.domain.entities import HandLandmarks, LandmarkPacket, MultiHandPacket

# Maps the user-facing hand preference to the MediaPipe label that will
# appear in a raw (non-mirrored) webcam frame.
# ``None`` means no filtering: any detected hand is accepted.
_USER_TO_MEDIAPIPE_LABEL: dict[str, str | None] = {
    "right": "left",  # user's right hand → MediaPipe "Left"
    "left": "right",  # user's left hand  → MediaPipe "Right"
    "both": None,  # accept either hand
}


class HandSelector:
    """Select the preferred hand from a :class:`MultiHandPacket`.

    Parameters
    ----------
    hand_preference:
        ``"left"``, ``"right"``, or ``"both"`` (from the user's perspective).
        When ``"both"``, the hand with the highest detection confidence is
        chosen.  When a specific side is requested, only a hand whose MediaPipe
        label matches the inverted preference is accepted; if no such hand is
        present the packet is marked invalid.
    min_handedness_conf:
        Minimum MediaPipe handedness classification score required for the
        returned :class:`LandmarkPacket` to be considered valid.
    min_track_conf:
        Minimum fraction of finite landmarks (``valid_points / 21``) required
        for the returned :class:`LandmarkPacket` to be considered valid.
    """

    def __init__(
        self,
        hand_preference: str = "both",
        min_handedness_conf: float = 0.7,
        min_track_conf: float = 0.5,
    ) -> None:
        key = hand_preference.lower()
        if key not in _USER_TO_MEDIAPIPE_LABEL:
            raise ValueError(
                f"hand_preference must be 'left', 'right', or 'both'; got {hand_preference!r}"
            )
        self._required_label: str | None = _USER_TO_MEDIAPIPE_LABEL[key]
        self._min_handedness_conf = min_handedness_conf
        self._min_track_conf = min_track_conf

    def select(self, packet: MultiHandPacket) -> LandmarkPacket:
        """Select the appropriate hand and return a :class:`LandmarkPacket`.

        Returns a packet with ``valid=False`` when no hands were detected or
        when the preferred hand is absent from the detections.  In the latter
        case the first detected hand's metadata is preserved for diagnostic use.
        """
        if not packet.hands:
            return LandmarkPacket(
                landmarks=[],
                handedness=None,
                detect_conf=0.0,
                track_conf=0.0,
                timestamp_ms=packet.timestamp_ms,
                valid=False,
            )

        hand: HandLandmarks | None
        if self._required_label is None:
            # "both": pick the hand with highest detection confidence.
            hand = max(packet.hands, key=lambda h: h.detect_conf)
        else:
            hand = next(
                (h for h in packet.hands if h.label.lower() == self._required_label),
                None,
            )

        if hand is None:
            # Preferred hand was not among the detections.  Return the first
            # detected hand's metadata so callers can log / debug what was seen.
            fallback = packet.hands[0]
            return LandmarkPacket(
                landmarks=fallback.landmarks,
                handedness=fallback.label,
                detect_conf=fallback.detect_conf,
                track_conf=fallback.track_conf,
                timestamp_ms=packet.timestamp_ms,
                valid=False,
            )

        is_valid = (
            hand.detect_conf >= self._min_handedness_conf
            and hand.track_conf >= self._min_track_conf
        )
        return LandmarkPacket(
            landmarks=hand.landmarks,
            handedness=hand.label,
            detect_conf=hand.detect_conf,
            track_conf=hand.track_conf,
            timestamp_ms=packet.timestamp_ms,
            valid=is_valid,
        )
