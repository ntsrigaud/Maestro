"""Unit tests: HeadPoseGazeDetector state machine and NullGazeDetector.

Coverage targets (per gaze-aware-activation.md §6):

* All four state transitions in _transition():
    ABSENT     → ACQUIRING  (within_cone first seen)
    ACQUIRING  → ABSENT     (cone interrupted before stability threshold)
    ACQUIRING  → ACTIVE     (stable for ≥ arm_ms)
    ACTIVE     → LOSING     (cone left)
    LOSING     → ACTIVE     (recovered within grace period)
    LOSING     → ABSENT     (grace period elapsed — fires on_gaze_lost)

* Amortization cache: non-sampled frames return cached bool without calling
  _is_within_cone.

* on_gaze_lost callback: fired only on LOSING → ABSENT transition.
  set_on_gaze_lost() replaces a previously registered callback.

* NullGazeDetector: update() always returns True; close() is a no-op.
"""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import numpy as np

from maestro.domain.entities import FramePacket
from maestro.infrastructure.gaze.head_pose_detector import HeadPoseGazeDetector
from maestro.infrastructure.gaze.null_detector import NullGazeDetector
from maestro.utils.schema_validation import GazeConfig

# ---------------------------------------------------------------------------
# Patch targets
# ---------------------------------------------------------------------------

_MP_FACE_DETECTION = (
    "maestro.infrastructure.gaze.head_pose_detector.mp.solutions.face_detection.FaceDetection"
)
_MONOTONIC = "maestro.infrastructure.gaze.head_pose_detector.time.monotonic"

# ---------------------------------------------------------------------------
# Shared test data
# ---------------------------------------------------------------------------

# GazeConfig with short timers (arm_ms=200, grace_ms=1000 — minimum allowed)
# and sample_interval_frames=1 so every call triggers a Face Mesh evaluation.
# All constructor values respect the Pydantic Field constraints (ge/le).
_CFG = GazeConfig(
    enabled=True,
    arm_ms=200,
    grace_ms=1000,
    sample_interval_frames=1,
)

_FRAME = FramePacket(
    frame=np.zeros((480, 640, 3), dtype=np.uint8),
    timestamp_ms=0.0,
    frame_id=0,
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_detector(
    cfg: GazeConfig = _CFG,
    on_gaze_lost: object = None,
) -> HeadPoseGazeDetector:
    """Return a HeadPoseGazeDetector with MediaPipe Face Detection mocked out."""
    with patch(_MP_FACE_DETECTION):
        return HeadPoseGazeDetector(config=cfg, on_gaze_lost=on_gaze_lost)  # type: ignore[arg-type]


def _update(
    detector: HeadPoseGazeDetector,
    within_cone: bool,
    now_s: float,
) -> bool:
    """Call detector.update() with _is_within_cone and time.monotonic controlled."""
    with (
        patch.object(detector, "_is_within_cone", return_value=within_cone),
        patch(_MONOTONIC, return_value=now_s),
    ):
        return detector.update(_FRAME)


# ---------------------------------------------------------------------------
# NullGazeDetector
# ---------------------------------------------------------------------------


class TestNullGazeDetector:
    def test_update_always_returns_true(self) -> None:
        det = NullGazeDetector()
        assert det.update(_FRAME) is True

    def test_repeated_calls_all_return_true(self) -> None:
        det = NullGazeDetector()
        assert all(det.update(_FRAME) for _ in range(10))

    def test_close_does_not_raise(self) -> None:
        det = NullGazeDetector()
        det.close()  # must not raise

    def test_satisfies_igaze_intent_detector_structurally(self) -> None:
        """Duck-type check: both required methods exist with correct arity."""
        det = NullGazeDetector()
        assert callable(det.update)
        assert callable(det.close)

    def test_head_pose_detector_satisfies_igaze_intent_detector_structurally(
        self,
    ) -> None:
        det = _make_detector()
        assert callable(det.update)
        assert callable(det.close)


# ---------------------------------------------------------------------------
# HeadPoseGazeDetector — initial state
# ---------------------------------------------------------------------------


class TestInitialState:
    def test_returns_false_before_any_face_detected(self) -> None:
        det = _make_detector()
        result = _update(det, within_cone=False, now_s=0.0)
        assert result is False

    def test_outside_cone_on_first_call_returns_false(self) -> None:
        det = _make_detector()
        assert _update(det, within_cone=False, now_s=0.0) is False


# ---------------------------------------------------------------------------
# HeadPoseGazeDetector — state machine transitions
# ---------------------------------------------------------------------------


class TestStateMachineTransitions:
    """ABSENT → ACQUIRING → ACTIVE → LOSING → ABSENT."""

    # -- ABSENT → ACQUIRING ---------------------------------------------------

    def test_absent_to_acquiring_returns_false(self) -> None:
        """First cone entry starts the stability timer; gaze is not yet permitted."""
        det = _make_detector()
        result = _update(det, within_cone=True, now_s=0.0)
        assert result is False  # ACQUIRING: gaze_permitted = False

    # -- ACQUIRING → ABSENT ---------------------------------------------------

    def test_acquiring_to_absent_when_cone_interrupted(self) -> None:
        """Two consecutive out-of-cone samples while ACQUIRING resets to ABSENT."""
        det = _make_detector()
        _update(det, within_cone=True, now_s=0.0)  # ABSENT → ACQUIRING
        _update(det, within_cone=False, now_s=0.050)  # 1st miss — tolerated
        result = _update(det, within_cone=False, now_s=0.100)  # 2nd miss → ABSENT
        assert result is False

    # -- ACQUIRING hysteresis -------------------------------------------------

    def test_acquiring_single_miss_is_tolerated(self) -> None:
        """A single out-of-cone result while ACQUIRING does not reset the state."""
        det = _make_detector()
        _update(det, within_cone=True, now_s=0.0)  # ACQUIRING
        _update(det, within_cone=False, now_s=0.050)  # 1st miss — tolerated
        # Detector is still ACQUIRING; next True at ≥ arm_ms must activate.
        result = _update(det, within_cone=True, now_s=0.201)  # elapsed 201 ms > 200 ms
        assert result is True  # timer was preserved → ACTIVE

    def test_acquiring_false_count_resets_on_true(self) -> None:
        """A True sample after a tolerated miss resets the false counter."""
        det = _make_detector()
        _update(det, within_cone=True, now_s=0.0)  # ACQUIRING
        _update(det, within_cone=False, now_s=0.050)  # 1st miss — tolerated (count=1)
        _update(det, within_cone=True, now_s=0.100)  # True resets count to 0
        # Now a single miss should again be tolerated.
        _update(det, within_cone=False, now_s=0.150)  # tolerated again
        result = _update(det, within_cone=True, now_s=0.201)
        assert result is True  # still ACTIVE-capable; timer never lost

    def test_acquiring_interrupted_then_new_hold_can_succeed(self) -> None:
        """After a full interruption (2 misses) a fresh hold reaches ACTIVE."""
        det = _make_detector()
        _update(det, within_cone=True, now_s=0.0)  # ACQUIRING
        _update(det, within_cone=False, now_s=0.050)  # 1st miss — tolerated
        _update(det, within_cone=False, now_s=0.100)  # 2nd miss → ABSENT
        _update(det, within_cone=True, now_s=1.0)  # new ACQUIRING (fresh timer)
        result = _update(det, within_cone=True, now_s=1.250)  # 250 ms > 200 ms arm_ms
        assert result is True  # ACTIVE

    # -- ACQUIRING → ACTIVE ---------------------------------------------------

    def test_acquiring_to_active_after_arm_ms_elapsed(self) -> None:
        """Pose stable for ≥ arm_ms transitions to ACTIVE (gaze_permitted = True)."""
        det = _make_detector()  # arm_ms = 200
        _update(det, within_cone=True, now_s=0.0)  # ACQUIRING (t=0 ms)
        result = _update(det, within_cone=True, now_s=0.201)  # 201 ms ≥ 200 ms
        assert result is True

    def test_acquiring_before_arm_ms_stays_acquiring(self) -> None:
        """Pose within cone but arm_ms not yet elapsed remains ACQUIRING."""
        det = _make_detector()  # arm_ms = 200
        _update(det, within_cone=True, now_s=0.0)  # ACQUIRING
        result = _update(det, within_cone=True, now_s=0.150)  # only 150 ms elapsed
        assert result is False

    # -- ACTIVE → LOSING ------------------------------------------------------

    def test_active_to_losing_when_cone_left(self) -> None:
        """Leaving the cone from ACTIVE enters LOSING; gaze still permitted (grace)."""
        det = _make_detector()
        _update(det, within_cone=True, now_s=0.0)
        _update(det, within_cone=True, now_s=0.201)  # ACTIVE
        result = _update(det, within_cone=False, now_s=0.202)  # ACTIVE → LOSING
        assert result is True  # grace period: still permitted

    # -- LOSING → ACTIVE ------------------------------------------------------

    def test_losing_to_active_when_cone_recovered(self) -> None:
        """Re-entering cone within grace period returns to ACTIVE."""
        det = _make_detector()
        _update(det, within_cone=True, now_s=0.0)
        _update(det, within_cone=True, now_s=0.201)  # ACTIVE
        _update(det, within_cone=False, now_s=0.202)  # LOSING
        result = _update(det, within_cone=True, now_s=0.500)  # recovered
        assert result is True

    def test_recovering_from_losing_resets_grace_timer(self) -> None:
        """After LOSING → ACTIVE, a subsequent cone loss starts a fresh grace timer."""
        det = _make_detector()  # grace_ms = 1000
        _update(det, within_cone=True, now_s=0.0)
        _update(det, within_cone=True, now_s=0.201)  # ACTIVE
        _update(det, within_cone=False, now_s=0.202)  # LOSING
        _update(det, within_cone=True, now_s=0.400)  # LOSING → ACTIVE
        _update(det, within_cone=False, now_s=0.401)  # ACTIVE → LOSING again
        result = _update(det, within_cone=False, now_s=0.900)  # 499 ms < 1000 ms grace
        assert result is True  # still in grace period

    # -- LOSING → ABSENT ------------------------------------------------------

    def test_losing_to_absent_after_grace_ms_elapsed(self) -> None:
        """Grace period expiry transitions LOSING → ABSENT (gaze_permitted = False)."""
        det = _make_detector()  # grace_ms = 1000
        _update(det, within_cone=True, now_s=0.0)
        _update(det, within_cone=True, now_s=0.201)  # ACTIVE
        _update(det, within_cone=False, now_s=1.0)  # LOSING (grace_start = 1000 ms)
        result = _update(det, within_cone=False, now_s=2.010)  # 1010 ms > 1000 ms
        assert result is False  # ABSENT

    def test_losing_within_grace_still_permitted(self) -> None:
        """Within the grace window the detector remains permitted."""
        det = _make_detector()  # grace_ms = 1000
        _update(det, within_cone=True, now_s=0.0)
        _update(det, within_cone=True, now_s=0.201)  # ACTIVE
        _update(det, within_cone=False, now_s=1.0)  # LOSING
        result = _update(det, within_cone=False, now_s=1.800)  # 800 ms < grace
        assert result is True


# ---------------------------------------------------------------------------
# HeadPoseGazeDetector — on_gaze_lost callback
# ---------------------------------------------------------------------------


class TestOnGazeLostCallback:
    def test_callback_fired_on_losing_to_absent(self) -> None:
        """on_gaze_lost must be called exactly once when grace expires."""
        cb = MagicMock()
        det = _make_detector(on_gaze_lost=cb)
        _update(det, within_cone=True, now_s=0.0)
        _update(det, within_cone=True, now_s=0.201)  # ACTIVE
        _update(det, within_cone=False, now_s=1.0)  # LOSING
        _update(det, within_cone=False, now_s=2.010)  # → ABSENT
        cb.assert_called_once()

    def test_callback_not_fired_when_acquiring_is_interrupted(self) -> None:
        """Interrupting ACQUIRING (2 consecutive misses) must not fire the callback."""
        cb = MagicMock()
        det = _make_detector(on_gaze_lost=cb)
        _update(det, within_cone=True, now_s=0.0)  # ACQUIRING
        _update(det, within_cone=False, now_s=0.050)  # 1st miss — tolerated
        _update(det, within_cone=False, now_s=0.100)  # 2nd miss → ABSENT
        cb.assert_not_called()

    def test_callback_not_fired_when_losing_recovers(self) -> None:
        """Recovering within the grace period must not fire the callback."""
        cb = MagicMock()
        det = _make_detector(on_gaze_lost=cb)
        _update(det, within_cone=True, now_s=0.0)
        _update(det, within_cone=True, now_s=0.201)  # ACTIVE
        _update(det, within_cone=False, now_s=1.0)  # LOSING
        _update(det, within_cone=True, now_s=1.400)  # LOSING → ACTIVE (recovered)
        cb.assert_not_called()

    def test_callback_fired_once_not_on_subsequent_absent_calls(self) -> None:
        """After ABSENT is reached, further calls in ABSENT must not re-fire."""
        cb = MagicMock()
        det = _make_detector(on_gaze_lost=cb)
        _update(det, within_cone=True, now_s=0.0)
        _update(det, within_cone=True, now_s=0.201)  # ACTIVE
        _update(det, within_cone=False, now_s=1.0)  # LOSING
        _update(det, within_cone=False, now_s=2.010)  # → ABSENT (callback fires)
        _update(det, within_cone=False, now_s=3.0)  # stays ABSENT
        _update(det, within_cone=False, now_s=4.0)  # stays ABSENT
        cb.assert_called_once()  # still exactly one call

    def test_set_on_gaze_lost_replaces_previous_callback(self) -> None:
        """set_on_gaze_lost() must replace any previously registered callback."""
        cb_old = MagicMock()
        cb_new = MagicMock()
        det = _make_detector()
        det.set_on_gaze_lost(cb_old)
        det.set_on_gaze_lost(cb_new)  # replaces
        _update(det, within_cone=True, now_s=0.0)
        _update(det, within_cone=True, now_s=0.201)  # ACTIVE
        _update(det, within_cone=False, now_s=1.0)  # LOSING
        _update(det, within_cone=False, now_s=2.010)  # → ABSENT
        cb_old.assert_not_called()
        cb_new.assert_called_once()

    def test_no_callback_registered_does_not_raise(self) -> None:
        """When on_gaze_lost is None, LOSING → ABSENT must not raise."""
        det = _make_detector(on_gaze_lost=None)
        _update(det, within_cone=True, now_s=0.0)
        _update(det, within_cone=True, now_s=0.201)  # ACTIVE
        _update(det, within_cone=False, now_s=1.0)  # LOSING
        _update(det, within_cone=False, now_s=2.010)  # → ABSENT  (no callback → OK)


# ---------------------------------------------------------------------------
# HeadPoseGazeDetector — amortization cache
# ---------------------------------------------------------------------------


class TestAmortizationCache:
    """Verify that _is_within_cone is only called on sampled frames."""

    def test_non_sampled_frames_return_cached_result(self) -> None:
        """With sample_interval_frames=3, frames 1 and 2 must return the cached bool."""
        cfg = GazeConfig(
            enabled=True,
            arm_ms=200,
            grace_ms=1000,
            sample_interval_frames=3,
        )
        det = _make_detector(cfg=cfg)

        # Frame 0 (counter % 3 == 0): sampled — within cone → ACQUIRING → False.
        with (
            patch.object(det, "_is_within_cone", return_value=True),
            patch(_MONOTONIC, return_value=0.0),
        ):
            r0 = det.update(_FRAME)

        # Frames 1 and 2: NOT sampled — _is_within_cone must not be called.
        mock_cone = MagicMock(return_value=True)
        with patch.object(det, "_is_within_cone", mock_cone):
            r1 = det.update(_FRAME)  # counter=1
            r2 = det.update(_FRAME)  # counter=2

        assert r0 is False
        assert r1 is False  # cached
        assert r2 is False  # cached
        mock_cone.assert_not_called()

    def test_sampled_frame_updates_cached_result(self) -> None:
        """On the sampling frame the cached result must reflect the new state."""
        cfg = GazeConfig(
            enabled=True,
            arm_ms=200,
            grace_ms=1000,
            sample_interval_frames=2,
        )
        det = _make_detector(cfg=cfg)

        # Frame 0: sampled → ACQUIRING (False).
        with (
            patch.object(det, "_is_within_cone", return_value=True),
            patch(_MONOTONIC, return_value=0.0),
        ):
            r0 = det.update(_FRAME)

        # Frame 1: NOT sampled → returns cached False.
        r1 = det.update(_FRAME)

        # Frame 2: sampled → arm_ms elapsed → ACTIVE (True).
        with (
            patch.object(det, "_is_within_cone", return_value=True),
            patch(_MONOTONIC, return_value=0.250),
        ):
            r2 = det.update(_FRAME)

        assert r0 is False
        assert r1 is False  # stale cache
        assert r2 is True  # cache refreshed after sampling

    def test_sample_interval_one_calls_cone_every_frame(self) -> None:
        """With sample_interval_frames=1, _is_within_cone is called on every update."""
        det = _make_detector(_CFG)  # _CFG has sample_interval_frames=1
        mock_cone = MagicMock(return_value=False)
        with (
            patch.object(det, "_is_within_cone", mock_cone),
            patch(_MONOTONIC, return_value=0.0),
        ):
            for _ in range(5):
                det.update(_FRAME)
        assert mock_cone.call_count == 5


# ---------------------------------------------------------------------------
# HeadPoseGazeDetector — close()
# ---------------------------------------------------------------------------


class TestClose:
    def test_close_delegates_to_face_detection_close(self) -> None:
        """close() must release the MediaPipe Face Detection context."""
        with patch(_MP_FACE_DETECTION) as mock_fd_cls:
            mock_fd = MagicMock()
            mock_fd_cls.return_value = mock_fd
            det = HeadPoseGazeDetector(config=_CFG)
            det.close()
        mock_fd.close.assert_called_once()


# ---------------------------------------------------------------------------
# HeadPoseGazeDetector — _is_within_cone() geometric proxy
# ---------------------------------------------------------------------------


def _make_results(keypoints: list, score: float = 0.9) -> object:
    """Build a minimal mock MediaPipe Face Detection result with one detection."""
    from types import SimpleNamespace

    kp_ns = [SimpleNamespace(x=x, y=y) for x, y in keypoints]
    location_data = SimpleNamespace(relative_keypoints=kp_ns)
    detection = SimpleNamespace(location_data=location_data, score=[score])
    return SimpleNamespace(detections=[detection])


# Reference face: inter_eye=0.10, eye_mid=(0.45,0.40), nose_y for pitch_norm=0.55.
# With default GazeConfig (yaw_threshold_deg=30, pitch_threshold_deg=25):
#   yaw_limit = (26/60)*tan(30°) ≈ 0.250
#   pitch_lo  = 0.55 - 25*0.016 = 0.15
#   pitch_hi  = 0.55 + 25*0.016 = 0.95
_RE = (0.40, 0.40)  # right eye
_LE = (0.50, 0.40)  # left eye  →  inter_eye=0.10, eye_mid=(0.45,0.40)
_NOSE_FRONTAL = (0.45, 0.455)  # yaw_norm=0.0, pitch_norm=0.55
_MOUTH = (0.45, 0.51)
_RIGHT_EAR = (0.34, 0.42)
_LEFT_EAR = (0.56, 0.42)

_FRONTAL_KPS = [_RE, _LE, _NOSE_FRONTAL, _MOUTH, _RIGHT_EAR, _LEFT_EAR]


class TestIsWithinCone:
    """Geometric proxy tests for _is_within_cone()."""

    # -- guard conditions ---------------------------------------------------

    def test_returns_false_when_no_detections(self) -> None:
        from types import SimpleNamespace

        det = _make_detector()
        results = SimpleNamespace(detections=[])
        assert det._is_within_cone(results) is False

    def test_returns_false_when_fewer_than_three_keypoints(self) -> None:
        """Fewer than 3 keypoints means nose is unavailable; must return False."""
        det = _make_detector()
        results = _make_results([(0.4, 0.4), (0.5, 0.4)])  # only 2 keypoints
        assert det._is_within_cone(results) is False

    def test_returns_false_when_low_confidence(self) -> None:
        """A detection below min_detection_score must be rejected."""
        det = _make_detector()  # min_detection_score defaults to 0.5
        results = _make_results(_FRONTAL_KPS, score=0.3)
        assert det._is_within_cone(results) is False

    def test_returns_false_when_inter_eye_too_small(self) -> None:
        """A degenerate detection with very small inter-eye distance is rejected."""
        det = _make_detector()
        # right_eye and left_eye only 0.005 apart → inter_eye < _MIN_INTER_EYE_RELATIVE
        tiny_kps = [
            (0.500, 0.40),
            (0.505, 0.40),
            (0.502, 0.455),
            _MOUTH,
            _RIGHT_EAR,
            _LEFT_EAR,
        ]
        results = _make_results(tiny_kps)
        assert det._is_within_cone(results) is False

    # -- frontal (should be within cone) ------------------------------------

    def test_returns_true_for_frontal_face(self) -> None:
        """A perfectly frontal face (yaw_norm≈0, pitch_norm≈0.55) must return True."""
        det = _make_detector()
        results = _make_results(_FRONTAL_KPS)
        assert det._is_within_cone(results) is True

    def test_returns_true_for_small_yaw(self) -> None:
        """A small yaw (well within threshold) must return True."""
        det = _make_detector()
        # nose shifted right by 0.010 → yaw_norm = 0.010/0.10 = 0.10 (< 0.25 limit)
        kps = list(_FRONTAL_KPS)
        kps[2] = (0.46, 0.455)
        results = _make_results(kps)
        assert det._is_within_cone(results) is True

    # -- yaw out of cone ---------------------------------------------------

    def test_returns_false_when_yaw_exceeds_threshold(self) -> None:
        """A nose shift producing yaw_norm > yaw_limit must return False."""
        det = _make_detector()
        # nose at x=0.478 → yaw_norm = (0.478-0.45)/0.10 = 0.28 > ~0.25 limit
        kps = list(_FRONTAL_KPS)
        kps[2] = (0.478, 0.455)
        results = _make_results(kps)
        assert det._is_within_cone(results) is False

    def test_returns_false_when_yaw_negative_exceeds_threshold(self) -> None:
        """Negative yaw exceeding the threshold (head turned opposite direction)."""
        det = _make_detector()
        kps = list(_FRONTAL_KPS)
        kps[2] = (0.422, 0.455)  # yaw_norm = (0.422-0.45)/0.10 = -0.28
        results = _make_results(kps)
        assert det._is_within_cone(results) is False

    # -- pitch out of cone -------------------------------------------------

    def test_returns_false_when_pitch_too_high(self) -> None:
        """pitch_norm above pitch_hi (looking too far down) returns False."""
        det = _make_detector()
        # pitch_norm = (nose_y - eye_mid_y) / inter_eye > 0.95
        # nose_y = 0.40 + 0.96 * 0.10 = 0.496
        kps = list(_FRONTAL_KPS)
        kps[2] = (0.45, 0.496)
        results = _make_results(kps)
        assert det._is_within_cone(results) is False

    def test_returns_false_when_pitch_too_low(self) -> None:
        """pitch_norm below pitch_lo (looking too far up) returns False."""
        det = _make_detector()
        # pitch_norm = (nose_y - eye_mid_y) / inter_eye < 0.15
        # nose_y = 0.40 + 0.14 * 0.10 = 0.414
        kps = list(_FRONTAL_KPS)
        kps[2] = (0.45, 0.414)
        results = _make_results(kps)
        assert det._is_within_cone(results) is False

    # -- wider thresholds --------------------------------------------------

    def test_wider_yaw_threshold_accepts_rotated_face(self) -> None:
        """A yaw_threshold_deg=60° must accept a face that 30° rejects."""
        cfg = GazeConfig(
            enabled=True,
            arm_ms=200,
            grace_ms=1000,
            sample_interval_frames=1,
            yaw_threshold_deg=60.0,
        )
        det = _make_detector(cfg=cfg)
        # nose at x=0.478 → yaw_norm=0.28; yaw_limit at 60°=(26/60)*tan(60°)≈0.75
        kps = list(_FRONTAL_KPS)
        kps[2] = (0.478, 0.455)
        results = _make_results(kps)
        assert det._is_within_cone(results) is True
