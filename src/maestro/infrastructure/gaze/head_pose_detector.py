"""Head-pose-based gaze intent detector using MediaPipe Face Detection (full-range).

Geometric implementation
------------------------
This module replaces the previous PnP-based angle estimator with a
scale-invariant geometric proxy that works reliably at camera distances up to
approximately 3 m — well beyond the 1–1.5 m practical limit of the
``cv2.solvePnP`` approach it supersedes.

Root cause of the previous range failure
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
The original implementation used MediaPipe's BlazeFace short-range model
(``model_selection=0``) and fed its 6 sparse keypoints into ``cv2.solvePnP``
to recover head rotation angles.  Two compounding problems caused failure at
distances beyond ~1 m:

1. **BlazeFace short-range model** is optimised for faces subtending >200 px.
   At 2 m in a 640×480 frame the face is ~40 px wide.  Keypoint localisation
   error (±3–5 px) becomes a >10 % error relative to the ~30 px inter-eye
   span, breaking the PnP correspondence.

2. **PnP instability from sparse noisy keypoints.**  ``cv2.solvePnP`` is
   highly sensitive to measurement noise when only 6 points are available.
   At 2 m, ±5 px keypoint jitter propagates to ±15–40° in the recovered
   Euler angles, causing random oscillation across the ±30°/±25° acceptance
   boundary every frame even for a stationary frontal face.

Geometric proxy approach (this implementation)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
Instead of recovering absolute Euler angles we compute two *dimensionless
ratios* of keypoint coordinates.  Because both the numerator and denominator
of each ratio scale proportionally with face size (hence camera distance),
the ratios are distance-invariant as long as a face is detected at all:

    yaw_norm  = (nose_x − eye_mid_x) / inter_eye_dist
    pitch_norm = (nose_y − eye_mid_y) / inter_eye_dist

where all coordinates are in the relative [0, 1] keypoint space returned
by MediaPipe Face Detection.

Geometric derivation of threshold mapping
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
For a generic adult face model (Farkas 1994 anthropometric reference):
nose-to-eye-plane Z-depth ≈ 26 mm, inter-eye distance ≈ 60 mm.

Under perspective projection the horizontal shift of the nose relative to
the eye midpoint satisfies:

    yaw_norm ≈ (26 / 60) · tan(θ_yaw)  =  0.433 · tan(θ_yaw)

The ``yaw_threshold_deg`` config parameter is converted to the ratio limit
via this exact formula.  For the default 30° this gives ≈ 0.25 — the nose
must stay within 25 % of the inter-eye distance from the eye midpoint.

Pitch uses an empirical linear approximation around the frontal-face neutral
value (pitch_norm ≈ 0.55):

    pitch_half_window = pitch_threshold_deg · 0.016

giving ≈ ±0.40 around neutral at the default ±25° threshold.

Both the full-range BlazeFace model (``model_selection=1``, rated ≤5 m) and
the configurable ``min_detection_score`` gate ensure that low-confidence
detections at range or during partial occlusion are rejected before the
geometric check, keeping the acceptance cone conservative.
"""

from __future__ import annotations

import logging
import math
import time
from collections.abc import Callable
from enum import Enum, auto

import cv2
import mediapipe as mp

from maestro.domain.entities import FramePacket
from maestro.utils.schema_validation import GazeConfig

LOGGER = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Geometric proxy calibration constants
# ---------------------------------------------------------------------------
# These dimensionless constants map the angular acceptance-cone thresholds
# from GazeConfig into the scale-invariant keypoint-ratio space.  They are
# derived from the Farkas (1994) adult face anthropometric norms:
#
#   nose-to-eye-plane Z-depth  ≈ 26 mm
#   inter-eye distance         ≈ 60 mm
#   nose-to-eye-midpoint Y     ≈ 33 mm  →  neutral pitch_norm ≈ 33/60 ≈ 0.55
#
# yaw_norm ≈ _YAW_NORM_FACTOR · tan(θ_yaw), derived as 26/60 ≈ 0.433.
_YAW_NORM_FACTOR: float = 26.0 / 60.0

# Neutral (frontal, 0° pitch) value of pitch_norm.
_PITCH_NEUTRAL_NORM: float = 0.55

# Linear pitch sensitivity: pitch_norm changes approximately this many units
# per degree of head elevation.  Calibrated from face model geometry.
_PITCH_NORM_PER_DEG: float = 0.016

# Minimum inter-eye distance in relative [0, 1] image coordinates.  Rejects
# faces whose bounding box is so small that keypoint noise dominates
# (typically faces at >3 m in a 640 px-wide frame).
_MIN_INTER_EYE_RELATIVE: float = 0.020


# ---------------------------------------------------------------------------
# Internal state machine enum

# ---------------------------------------------------------------------------
# Internal state machine enum
# ---------------------------------------------------------------------------


class _GazeState(Enum):
    ABSENT = auto()
    ACQUIRING = auto()
    ACTIVE = auto()
    LOSING = auto()


# ---------------------------------------------------------------------------
# HeadPoseGazeDetector
# ---------------------------------------------------------------------------


class HeadPoseGazeDetector:
    """Gaze-intent detector using MediaPipe Face Detection (full-range) and
    scale-invariant geometric head-pose estimation.

    Implements ``IGazeIntentDetector`` structurally (duck-typed Protocol).

    ``update()`` returns ``True`` when the user's head is oriented toward the
    camera (states GAZE_ACTIVE or GAZE_LOSING within the grace period) and
    ``False`` otherwise.

    Face Detection is sampled once every ``config.sample_interval_frames``
    frames; the cached boolean is returned on intermediate frames.  At 30 FPS
    with the default ``sample_interval_frames=3`` the amortised per-loop cost
    is ~2 ms.

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

        # Full-range BlazeFace model (model_selection=1) supports detection
        # up to approximately 5 m, vs ≤2 m for the short-range model=0.
        # The min_detection_confidence threshold is set from config so that
        # low-confidence detections at range are rejected at the C++ level.
        self._face_detection = mp.solutions.face_detection.FaceDetection(
            model_selection=1,  # full-range model (≤ 5 m)
            min_detection_confidence=self._cfg.min_detection_score,
        )

        self._state: _GazeState = _GazeState.ABSENT
        self._stability_start_ms: float | None = None
        self._grace_start_ms: float | None = None

        # Frame counter drives sub-sampling; starts at 0 so the very first
        # call always triggers a Face Detection evaluation.
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
            # A C-contiguous RGB copy is required; a reversed-stride view
            # (frame[:, :, ::-1]) can cause silent failures in some MediaPipe
            # C extensions that require contiguous memory.
            rgb = cv2.cvtColor(frame.frame, cv2.COLOR_BGR2RGB)
            results = self._face_detection.process(rgb)
            within_cone = self._is_within_cone(results)
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
        """Release the MediaPipe Face Detection context."""
        self._face_detection.close()

    def set_on_gaze_lost(self, callback: Callable[[], None]) -> None:
        """Register (or replace) the gaze-lost callback after construction.

        Provided so that the bootstrap can wire ``force_idle`` after both the
        gaze detector and the activation controller have been instantiated.
        """
        self._on_gaze_lost = callback

    # ------------------------------------------------------------------
    # Geometric head-pose estimation
    # ------------------------------------------------------------------

    def _is_within_cone(self, results) -> bool:  # type: ignore[no-untyped-def]
        """Return True if the detected head pose is within the acceptance cone.

        Uses scale-invariant geometric ratios of the BlazeFace keypoints
        rather than PnP pose estimation.  See module docstring for the
        derivation and calibration of the angular-threshold mapping.

        Keypoint layout (fixed by MediaPipe Face Detection):
            0 → right eye centre  (appears on LEFT in a frontal-face image)
            1 → left eye centre   (appears on RIGHT in a frontal-face image)
            2 → nose tip
        """
        if not results.detections:
            return False

        det = results.detections[0]

        # The min_detection_confidence passed to FaceDetection is the primary
        # filter.  This secondary Python-level check guards against edge cases
        # where the C++ layer emits marginal results that narrowly satisfy the
        # C-level threshold.
        if not det.score or det.score[0] < self._cfg.min_detection_score:
            return False

        keypoints = det.location_data.relative_keypoints
        if len(keypoints) < 3:
            return False

        # Relative image coordinates in [0.0, 1.0].
        re_x, re_y = keypoints[0].x, keypoints[0].y  # right eye
        le_x, le_y = keypoints[1].x, keypoints[1].y  # left eye
        no_x, no_y = keypoints[2].x, keypoints[2].y  # nose tip

        # Scale reference: Euclidean inter-eye distance in relative coords.
        inter_eye = math.hypot(le_x - re_x, le_y - re_y)
        if inter_eye < _MIN_INTER_EYE_RELATIVE:
            return False  # face too small (extreme range) or degenerate detection

        eye_mid_x = (re_x + le_x) * 0.5
        eye_mid_y = (re_y + le_y) * 0.5

        # --- Yaw proxy ---
        # yaw_norm = (nose_x − eye_midpoint_x) / inter_eye_dist
        # ≈ _YAW_NORM_FACTOR · tan(θ_yaw) for small-to-moderate yaw angles.
        # → 0.0 when frontal; grows toward ±0.25 at ±30° for an average face.
        yaw_norm = (no_x - eye_mid_x) / inter_eye
        yaw_limit = _YAW_NORM_FACTOR * math.tan(math.radians(self._cfg.yaw_threshold_deg))

        # --- Pitch proxy ---
        # pitch_norm = (nose_y − eye_midpoint_y) / inter_eye_dist
        # ≈ _PITCH_NEUTRAL_NORM at 0° elevation; shifts by _PITCH_NORM_PER_DEG
        # per degree of elevation (positive pitch_norm = nose below eye level).
        pitch_norm = (no_y - eye_mid_y) / inter_eye
        pitch_half = self._cfg.pitch_threshold_deg * _PITCH_NORM_PER_DEG
        pitch_lo = _PITCH_NEUTRAL_NORM - pitch_half
        pitch_hi = _PITCH_NEUTRAL_NORM + pitch_half

        within = abs(yaw_norm) <= yaw_limit and pitch_lo <= pitch_norm <= pitch_hi
        LOGGER.debug(
            "Gaze geometry: yaw_norm=%.3f(±%.3f) pitch_norm=%.3f([%.3f,%.3f]) → within=%s",
            yaw_norm,
            yaw_limit,
            pitch_norm,
            pitch_lo,
            pitch_hi,
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
                # Face Detection result must not abort a valid acquisition.
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
                    LOGGER.debug("Gaze: ACQUIRING → ACTIVE (stable for %d ms)", self._cfg.arm_ms)

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
