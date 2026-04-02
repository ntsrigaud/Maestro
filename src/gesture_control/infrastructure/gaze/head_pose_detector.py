"""Head-pose-based gaze intent detector using MediaPipe Face Mesh and solvePnP."""

from __future__ import annotations

import logging
import math
import time
from collections.abc import Callable
from enum import Enum, auto

import cv2
import mediapipe as mp
import numpy as np

from gesture_control.domain.entities import FramePacket
from gesture_control.utils.schema_validation import GazeConfig

LOGGER = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Six-point PnP face model
# ---------------------------------------------------------------------------
# Canonical 3-D coordinates (millimetres) for six anatomical landmarks that
# are reliably detected by MediaPipe Face Mesh.  These serve as the object
# points in the Perspective-n-Point solve.
#
# Landmark index → anatomy:
#   1   → nose tip     (between brows, closest to camera in neutral pose)
#   152 → chin         (bottom of mandible)
#   226 → left  eye outer corner  (temporal canthus, left side of face)
#   446 → right eye outer corner  (temporal canthus, right side of face)
#   57  → left  mouth corner
#   287 → right mouth corner
#
# The coordinate system is face-centred: origin at the nose tip, X rightward,
# Y upward, Z toward the camera.  Values are adapted from the Fisher (2016)
# generic face model.
_LANDMARK_INDICES: tuple[int, ...] = (1, 152, 226, 446, 57, 287)

_MODEL_POINTS_3D = np.array(
    [
        [0.0, 0.0, 0.0],  # 1   nose tip
        [0.0, -63.6, -12.5],  # 152 chin
        [-43.3, 32.7, -26.0],  # 226 left eye outer corner
        [43.3, 32.7, -26.0],  # 446 right eye outer corner
        [-28.9, -28.9, -24.1],  # 57  left mouth corner
        [28.9, -28.9, -24.1],  # 287 right mouth corner
    ],
    dtype=np.float64,
)

# Zero distortion coefficients.  The barrel distortion of typical webcam
# lenses is modest and does not materially affect the head-pose angle
# estimates at the ±30°/±25° acceptance-cone granularity used here.
_DIST_COEFFS = np.zeros((4, 1), dtype=np.float64)


# ---------------------------------------------------------------------------
# Internal state machine enum
# ---------------------------------------------------------------------------


class _GazeState(Enum):
    ABSENT = auto()
    ACQUIRING = auto()
    ACTIVE = auto()
    LOSING = auto()


# ---------------------------------------------------------------------------
# Euler-angle extraction
# ---------------------------------------------------------------------------


def _euler_yaw_pitch(rotation_matrix: np.ndarray) -> tuple[float, float]:
    """Return (roll_deg, head_turn_deg) from a 3×3 object-to-camera rotation.

    Uses the ZYX Euler decomposition of the rotation matrix R produced by
    ``cv2.Rodrigues(rvec)`` after a ``cv2.solvePnP`` call.

    In OpenCV camera coordinates (X right, Y down, Z forward into scene) the
    decomposition yields:

    * First angle  α (``yaw`` below)   = rotation around camera-Z (forward).
      In practice this captures **head tilt** (ear-toward-shoulder roll).
    * Second angle β (``pitch`` below) = rotation around camera-Y (down).
      In practice this captures **head turn** (looking left/right, i.e. yaw
      in world terms).  This is the critical angle for the acceptance cone.
    * Third  angle γ (roll)            = rotation around camera-X (right).
      Captures nodding up/down; not returned — less relevant for a camera
      positioned at screen level.

    For a face looking directly at the camera both returned angles are ≈ 0°.

    Returns
    -------
    (yaw_deg, pitch_deg) :
        yaw_deg   — head tilt / roll  (camera-Z rotation; checked vs yaw_threshold).
        pitch_deg — head turn L/R     (camera-Y rotation; checked vs pitch_threshold).
    """
    sy = math.sqrt(rotation_matrix[0, 0] ** 2 + rotation_matrix[1, 0] ** 2)
    singular = sy < 1e-6

    if not singular:
        pitch = math.atan2(-rotation_matrix[2, 0], sy)
        yaw = math.atan2(rotation_matrix[1, 0], rotation_matrix[0, 0])
    else:
        # Gimbal-lock fallback: yaw is indeterminate; set to zero.
        pitch = math.atan2(-rotation_matrix[2, 0], sy)
        yaw = 0.0

    return math.degrees(yaw), math.degrees(pitch)


# ---------------------------------------------------------------------------
# HeadPoseGazeDetector
# ---------------------------------------------------------------------------


class HeadPoseGazeDetector:
    """Gaze-intent detector based on MediaPipe Face Mesh and PnP head-pose.

    Implements ``IGazeIntentDetector`` structurally (duck-typed Protocol).

    ``update()`` returns ``True`` when the user's head is oriented toward the
    camera (states GAZE_ACTIVE or GAZE_LOSING within the grace period) and
    ``False`` otherwise.

    Face Mesh is sampled once every ``config.sample_interval_frames`` frames;
    the cached boolean is returned on intermediate frames.  At 30 FPS with the
    default ``sample_interval_frames=3`` the amortised per-loop cost is ~2 ms.

    State machine
    -------------
    ABSENT    : no qualifying head pose detected.
    ACQUIRING : qualifying pose detected; stability timer running.
    ACTIVE    : pose stable for ``arm_ms``; gaze is permitted.
    LOSING    : pose left acceptance cone; grace timer running.

    Transitions from GAZE_LOSING to GAZE_ABSENT invoke ``on_gaze_lost()``,
    which the bootstrap wires to ``GestureActivationController.force_idle()``
    to prevent a stale ARMED state from persisting after the user looks away.

    Parameters
    ----------
    config :
        ``GazeConfig`` instance, typically loaded from ``inference.yaml``.
    on_gaze_lost :
        Optional zero-argument callable invoked when the grace period expires
        and the state transitions LOSING → ABSENT.  May be left ``None``
        during construction and set later via ``set_on_gaze_lost()``.
    """

    def __init__(
        self,
        config: GazeConfig,
        on_gaze_lost: Callable[[], None] | None = None,
    ) -> None:
        self._cfg = config
        self._on_gaze_lost = on_gaze_lost

        self._face_mesh = mp.solutions.face_mesh.FaceMesh(
            max_num_faces=1,
            refine_landmarks=False,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5,
        )

        self._state: _GazeState = _GazeState.ABSENT
        self._stability_start_ms: float | None = None
        self._grace_start_ms: float | None = None

        # Frame counter drives the sub-sampling; starts at 0 so the very
        # first call always triggers a Face Mesh evaluation.
        self._frame_counter: int = 0
        self._last_permitted: bool = False

        # Consecutive within_cone=False count while ACQUIRING.  A minimum of
        # two consecutive misses is required before resetting the stability
        # timer.  This provides hysteresis against single-frame detection
        # noise that would otherwise prevent ACQUIRING → ACTIVE transitions.
        self._acquiring_false_count: int = 0

    # ------------------------------------------------------------------
    # Public interface  (satisfies IGazeIntentDetector)
    # ------------------------------------------------------------------

    def update(self, frame: FramePacket) -> bool:
        """Advance the gaze state machine and return ``gaze_permitted``."""
        if self._frame_counter % self._cfg.sample_interval_frames == 0:
            h, w = frame.frame.shape[:2]
            # Use cv2.cvtColor to produce a C-contiguous RGB copy.  A
            # reversed-stride view (frame[:, :, ::-1]) can cause silent
            # failures in some solvePnP and MediaPipe C extensions that
            # require contiguous memory.
            rgb = cv2.cvtColor(frame.frame, cv2.COLOR_BGR2RGB)
            results = self._face_mesh.process(rgb)
            within_cone = self._is_within_cone(results, h, w)
            now_ms = time.monotonic() * 1000.0
            self._transition(within_cone, now_ms)
            self._last_permitted = self._state in (_GazeState.ACTIVE, _GazeState.LOSING)
            LOGGER.debug(
                "Gaze update: state=%s within_cone=%s permitted=%s",
                self._state.name,
                within_cone,
                self._last_permitted,
            )

        self._frame_counter += 1
        if self._frame_counter % 90 == 0:
            LOGGER.info(
                "Gaze status: state=%s permitted=%s (frame %d)",
                self._state.name,
                self._last_permitted,
                self._frame_counter,
            )
        return self._last_permitted

    def close(self) -> None:
        """Release the MediaPipe Face Mesh context."""
        self._face_mesh.close()

    def set_on_gaze_lost(self, callback: Callable[[], None]) -> None:
        """Register (or replace) the gaze-lost callback after construction.

        Provided so that the bootstrap can wire ``force_idle`` after both the
        gaze detector and the activation controller have been instantiated.
        """
        self._on_gaze_lost = callback

    # ------------------------------------------------------------------
    # Head-pose estimation
    # ------------------------------------------------------------------

    def _is_within_cone(self, results, h: int, w: int) -> bool:  # type: ignore[no-untyped-def]
        """Return True if the detected head pose is within the acceptance cone."""
        if not results.multi_face_landmarks:
            return False

        landmarks = results.multi_face_landmarks[0].landmark

        image_points = np.array(
            [[landmarks[idx].x * w, landmarks[idx].y * h] for idx in _LANDMARK_INDICES],
            dtype=np.float64,
        )

        # Approximate camera intrinsics from the image width.  A focal length
        # equal to the image width gives a ~53° diagonal FoV, which is typical
        # for consumer webcams (Logitech C920: 78° diagonal = ~53° vertical).
        focal_length = float(w)
        camera_matrix = np.array(
            [
                [focal_length, 0.0, w / 2.0],
                [0.0, focal_length, h / 2.0],
                [0.0, 0.0, 1.0],
            ],
            dtype=np.float64,
        )

        success, rvec, _ = cv2.solvePnP(
            _MODEL_POINTS_3D,
            image_points,
            camera_matrix,
            _DIST_COEFFS,
            flags=cv2.SOLVEPNP_ITERATIVE,
        )
        if not success:
            return False

        rmat, _ = cv2.Rodrigues(rvec)
        yaw_deg, pitch_deg = _euler_yaw_pitch(rmat)

        within = (
            abs(yaw_deg) <= self._cfg.yaw_threshold_deg
            and abs(pitch_deg) <= self._cfg.pitch_threshold_deg
        )
        LOGGER.debug(
            "Gaze pose: tilt=%.1f°(±%.0f°) turn=%.1f°(±%.0f°) → within=%s",
            yaw_deg,
            self._cfg.yaw_threshold_deg,
            pitch_deg,
            self._cfg.pitch_threshold_deg,
            within,
        )
        return within

    # ------------------------------------------------------------------
    # State machine
    # ------------------------------------------------------------------

    def _transition(self, within_cone: bool, now_ms: float) -> None:
        """Advance the four-state machine given current cone membership."""
        if self._state is _GazeState.ABSENT:
            if within_cone:
                self._state = _GazeState.ACQUIRING
                self._stability_start_ms = now_ms
                LOGGER.debug("Gaze: ABSENT → ACQUIRING")

        elif self._state is _GazeState.ACQUIRING:
            if not within_cone:
                # Hysteresis: require two consecutive out-of-cone samples
                # before resetting the stability timer.  A single noisy
                # Face Mesh result must not abort a valid acquisition.
                self._acquiring_false_count += 1
                if self._acquiring_false_count >= 2:
                    self._state = _GazeState.ABSENT
                    self._stability_start_ms = None
                    self._acquiring_false_count = 0
                    LOGGER.debug("Gaze: ACQUIRING → ABSENT (pose lost)")
                else:
                    LOGGER.debug(
                        "Gaze: ACQUIRING — transient miss (%d/2); stability timer preserved",
                        self._acquiring_false_count,
                    )
            else:
                self._acquiring_false_count = 0
                if (
                    self._stability_start_ms is not None
                    and now_ms - self._stability_start_ms >= self._cfg.arm_ms
                ):
                    self._state = _GazeState.ACTIVE
                    self._stability_start_ms = None
                    LOGGER.debug(
                        "Gaze: ACQUIRING → ACTIVE (stable for %d ms)", self._cfg.arm_ms
                    )

        elif self._state is _GazeState.ACTIVE:
            if not within_cone:
                self._state = _GazeState.LOSING
                self._grace_start_ms = now_ms
                LOGGER.debug("Gaze: ACTIVE → LOSING (grace period started)")

        elif self._state is _GazeState.LOSING:
            if within_cone:
                self._state = _GazeState.ACTIVE
                self._grace_start_ms = None
                LOGGER.debug("Gaze: LOSING → ACTIVE (recovered)")
            elif (
                self._grace_start_ms is not None
                and now_ms - self._grace_start_ms >= self._cfg.grace_ms
            ):
                self._state = _GazeState.ABSENT
                self._grace_start_ms = None
                LOGGER.debug("Gaze: LOSING → ABSENT (grace period expired)")
                if self._on_gaze_lost is not None:
                    self._on_gaze_lost()
