from __future__ import annotations

import math
from collections import deque
from typing import cast

import numpy as np

from maestro.domain.entities import FeatureFrame, LandmarkPacket

# MediaPipe Hand landmark indices for the 5 fingertips.
# Used to compute spread features (distance of each fingertip from wrist origin
# in normalised, wrist-relative space).
_FINGERTIP_INDICES: list[int] = [4, 8, 12, 16, 20]  # thumb, index, middle, ring, pinky

# (parent, joint, child) triplets for 10 key joint-angle features.
# Angles are computed in normalised (wrist-relative, scale-normalised) space.
# Covers the primary flexion/extension angles at MCP and PIP joints of each finger.
_JOINT_TRIPLETS: list[tuple[int, int, int]] = [
    (0, 1, 2),  # thumb  CMC → MCP → IP
    (1, 2, 3),  # thumb  MCP → IP  → TIP
    (0, 5, 6),  # index  wrist→ MCP → PIP
    (5, 6, 7),  # index  MCP → PIP → DIP
    (0, 9, 10),  # middle wrist→ MCP → PIP
    (9, 10, 11),  # middle MCP → PIP → DIP
    (0, 13, 14),  # ring   wrist→ MCP → PIP
    (13, 14, 15),  # ring   MCP → PIP → DIP
    (0, 17, 18),  # pinky  wrist→ MCP → PIP
    (17, 18, 19),  # pinky  MCP → PIP → DIP
]


class LandmarkFeatureTransformer:
    """Transforms hand landmarks into normalized position + velocity features."""

    def __init__(
        self,
        *,
        enable_orientation_normalization: bool = False,
        min_scale: float = 1e-6,
        wrist_trajectory_window: int = 24,
    ) -> None:
        self._enable_orientation_normalization = enable_orientation_normalization
        self._min_scale = min_scale
        self._prev_normalized: list[list[float]] | None = None
        self._prev_timestamp_ms: int | None = None
        # Raw wrist position history (bounded to the last *wrist_trajectory_window*
        # frames).  The trajectory feature is always the displacement from the
        # oldest entry in the deque, so it is bounded by the window duration and
        # never grows unboundedly during a long inference session.
        self._wrist_history: deque[list[float]] = deque(maxlen=wrist_trajectory_window)
        # Previous raw wrist position used to compute per-frame global wrist
        # velocity in camera-normalized coordinates (feature-schema-v5).
        self._prev_raw_wrist: list[float] | None = None

    def transform(self, landmarks: LandmarkPacket) -> FeatureFrame:
        if not landmarks.valid or len(landmarks.landmarks) != 21:
            vector = self._zero_vector()
            self._clear_velocity_state()
            return FeatureFrame(vector=vector, timestamp_ms=landmarks.timestamp_ms, valid=False)

        if not self._all_finite(landmarks.landmarks):
            vector = self._zero_vector()
            self._clear_velocity_state()
            return FeatureFrame(vector=vector, timestamp_ms=landmarks.timestamp_ms, valid=False)

        normalized = self._normalize_positions(landmarks.landmarks)
        if normalized is None:
            vector = self._zero_vector()
            self._clear_velocity_state()
            return FeatureFrame(vector=vector, timestamp_ms=landmarks.timestamp_ms, valid=False)

        velocity = self._compute_velocity(normalized, landmarks.timestamp_ms)

        # Wrist trajectory: 3-D displacement of the current raw wrist position
        # from the oldest position in the bounded history deque.  Using a deque
        # of maxlen=window_length ensures the reference point is always at most
        # one window-duration in the past, preventing unbounded drift during
        # long inference sessions.
        raw_wrist = [
            float(landmarks.landmarks[0][0]),
            float(landmarks.landmarks[0][1]),
            float(landmarks.landmarks[0][2]),
        ]

        # Global wrist velocity in camera-normalized coordinates.
        # Computed BEFORE updating history so self._prev_raw_wrist and
        # self._prev_timestamp_ms refer to the immediately preceding frame.
        # Unlike velocity (dims 71-133), which is computed in wrist-relative
        # space and therefore near-zero for pure translations, this feature
        # directly encodes the direction and speed of overall hand movement.
        if self._prev_raw_wrist is not None and self._prev_timestamp_ms is not None:
            dt_vel = (landmarks.timestamp_ms - self._prev_timestamp_ms) / 1000.0
            if dt_vel > 0:
                wrist_vel_raw: list[float] = [
                    (raw_wrist[i] - self._prev_raw_wrist[i]) / dt_vel for i in range(3)
                ]
            else:
                wrist_vel_raw = [0.0, 0.0, 0.0]
        else:
            wrist_vel_raw = [0.0, 0.0, 0.0]
        self._prev_raw_wrist = raw_wrist

        self._wrist_history.append(raw_wrist)
        origin = self._wrist_history[0]
        wrist_traj: list[float] = [
            raw_wrist[0] - origin[0],
            raw_wrist[1] - origin[1],
            raw_wrist[2] - origin[2],
        ]

        # Fingertip spread: Euclidean distance of each fingertip from the
        # wrist origin in normalised space.  Distinguishes open vs. closed hand.
        spread: list[float] = [
            self._distance(normalized[tip], [0.0, 0.0, 0.0]) for tip in _FINGERTIP_INDICES
        ]

        # Feature vector layout (147 dims):
        #   [0:63]    normalised positions (21 × 3)
        #   [63:68]   fingertip spread (5)
        #   [68:71]   wrist trajectory — net displacement from oldest frame (3)
        #   [71:134]  normalised velocity in wrist-relative space (21 × 3)
        #   [134:144] joint angles (10)
        #   [144:147] wrist velocity in camera-normalized coordinates (3)  ← v5
        #
        # Dims 144-146 are the instantaneous velocity of the wrist landmark
        # in MediaPipe's camera-normalized coordinate system (NOT wrist-relative).
        # For a pure lateral swipe the wrist-relative velocity (dims 71-133) is
        # near-zero because all landmarks translate together; this block provides
        # the missing directional signal.
        joint_angles = self._joint_angles(normalized)
        vector = np.asarray(
            (
                self._flatten(normalized)
                + spread
                + wrist_traj
                + self._flatten(velocity)
                + joint_angles
                + wrist_vel_raw
            ),
            dtype=np.float32,
        )

        self._prev_normalized = normalized
        self._prev_timestamp_ms = landmarks.timestamp_ms
        return FeatureFrame(vector=vector, timestamp_ms=landmarks.timestamp_ms, valid=True)

    def _clear_velocity_state(self) -> None:
        self._prev_normalized = None
        self._prev_timestamp_ms = None
        self._prev_raw_wrist = None
        self._wrist_history.clear()

    def _normalize_positions(self, raw: list[list[float]]) -> list[list[float]] | None:
        wrist = raw[0]
        relative: list[list[float]] = []
        for point in raw:
            relative.append(
                [
                    float(point[0] - wrist[0]),
                    float(point[1] - wrist[1]),
                    float(point[2] - wrist[2]),
                ]
            )

        scale = self._distance(relative[9], [0.0, 0.0, 0.0])
        if scale < self._min_scale:
            return None

        normalized = [[coord / scale for coord in point] for point in relative]
        if self._enable_orientation_normalization:
            oriented = self._apply_orientation_normalization(normalized)
            if oriented is not None:
                normalized = oriented

        return normalized

    def _apply_orientation_normalization(
        self, normalized: list[list[float]]
    ) -> list[list[float]] | None:
        palm_x = self._subtract(normalized[5], normalized[17])
        palm_y = normalized[9]

        x_hat = self._unit_vector(palm_x)
        y_hat = self._unit_vector(palm_y)
        if x_hat is None or y_hat is None:
            return None

        z_hat = self._unit_vector(self._cross(x_hat, y_hat))
        if z_hat is None:
            return None

        y_orth = self._unit_vector(self._cross(z_hat, x_hat))
        if y_orth is None:
            return None

        oriented: list[list[float]] = []
        for point in normalized:
            oriented.append(
                [
                    self._dot(point, x_hat),
                    self._dot(point, y_orth),
                    self._dot(point, z_hat),
                ]
            )
        return oriented

    def _compute_velocity(
        self,
        normalized: list[list[float]],
        timestamp_ms: int,
    ) -> list[list[float]]:
        if self._prev_normalized is None or self._prev_timestamp_ms is None:
            return [[0.0, 0.0, 0.0] for _ in range(21)]

        delta_ms = timestamp_ms - self._prev_timestamp_ms
        if delta_ms <= 0:
            return [[0.0, 0.0, 0.0] for _ in range(21)]

        dt = delta_ms / 1000.0
        velocity: list[list[float]] = []
        for curr, prev in zip(normalized, self._prev_normalized, strict=True):
            velocity.append(
                [
                    (curr[0] - prev[0]) / dt,
                    (curr[1] - prev[1]) / dt,
                    (curr[2] - prev[2]) / dt,
                ]
            )
        return velocity

    @staticmethod
    def _all_finite(points: list[list[float]]) -> bool:
        for point in points:
            for value in point:
                if not math.isfinite(value):
                    return False
        return True

    @staticmethod
    def _distance(a: list[float], b: list[float]) -> float:
        return math.sqrt((a[0] - b[0]) ** 2 + (a[1] - b[1]) ** 2 + (a[2] - b[2]) ** 2)

    @staticmethod
    def _flatten(points: list[list[float]]) -> list[float]:
        flat: list[float] = []
        for point in points:
            flat.extend(point)
        return flat

    @staticmethod
    def _zero_vector() -> np.ndarray:
        return np.zeros(147, dtype=np.float32)

    @staticmethod
    def _joint_angles(normalized: list[list[float]]) -> list[float]:
        """Compute 10 key joint-bend angles (in radians) in normalised space.

        For each (parent, joint, child) triplet the angle at *joint* is the
        arccos of the cosine between vectors (parent→joint) and (child→joint).
        Returns zero for degenerate (coincident) landmarks.
        """
        angles: list[float] = []
        for parent, joint, child in _JOINT_TRIPLETS:
            a = normalized[parent]
            b = normalized[joint]
            c = normalized[child]
            # Vectors from joint outward
            ba = [a[i] - b[i] for i in range(3)]
            bc = [c[i] - b[i] for i in range(3)]
            norm_ba = math.sqrt(ba[0] ** 2 + ba[1] ** 2 + ba[2] ** 2)
            norm_bc = math.sqrt(bc[0] ** 2 + bc[1] ** 2 + bc[2] ** 2)
            if norm_ba < 1e-8 or norm_bc < 1e-8:
                angles.append(0.0)
            else:
                cos_angle = (ba[0] * bc[0] + ba[1] * bc[1] + ba[2] * bc[2]) / (norm_ba * norm_bc)
                # Clamp to [-1, 1] for numerical safety before arccos.
                angles.append(math.acos(max(-1.0, min(1.0, cos_angle))))
        return angles

    @staticmethod
    def _subtract(a: list[float], b: list[float]) -> list[float]:
        return [a[0] - b[0], a[1] - b[1], a[2] - b[2]]

    @staticmethod
    def _dot(a: list[float], b: list[float]) -> float:
        return float(a[0] * b[0] + a[1] * b[1] + a[2] * b[2])

    @staticmethod
    def _cross(a: list[float], b: list[float]) -> list[float]:
        return [
            a[1] * b[2] - a[2] * b[1],
            a[2] * b[0] - a[0] * b[2],
            a[0] * b[1] - a[1] * b[0],
        ]

    @staticmethod
    def _unit_vector(v: list[float]) -> list[float] | None:
        norm = math.sqrt(v[0] ** 2 + v[1] ** 2 + v[2] ** 2)
        if norm <= 0.0:
            return None
        return cast(list[float], [v[0] / norm, v[1] / norm, v[2] / norm])
