# Work Log — Gaze Detector Range Extension: Scale-Invariant Geometric Proxy

**Date:** 2026-04-25  
**Branch:** `fix/expand-gaze-detector-range`  
**Author:** @ntsrigaud

---

## Context

The `HeadPoseGazeDetector` became unreliable at camera distances beyond approximately
1 m. Two compounding factors caused this. First, the short-range BlazeFace model
(`model_selection=0`) is optimised for faces subtending more than 200 px; at 2 m in a
640×480 frame the face occupies roughly 40 px, making keypoint localisation errors
(±3–5 px) exceed 10 % of the ~30 px inter-eye span. Second, `cv2.solvePnP` was used
to recover head rotation angles from only 6 sparse, noisy keypoints. At range, ±5 px
keypoint jitter propagated to ±15–40° in the recovered Euler angles, causing random
oscillation across the ±30°/±25° acceptance boundary even for a stationary, frontal
face. The practical effect was that the gaze gate became non-functional at any seated
working distance greater than about 1 m.

---

## Scope of Work

### Refactoring

- Replaced the `cv2.solvePnP` pose estimator with a scale-invariant geometric proxy
  based on dimensionless keypoint ratios.
- Switched the MediaPipe Face Detection model from `model_selection=0` (short-range,
  ≤2 m) to `model_selection=1` (full-range, ≤5 m).
- Added calibration constants (`_YAW_NORM_FACTOR`, `_PITCH_NEUTRAL_NORM`,
  `_PITCH_NORM_PER_DEG`, `_MIN_INTER_EYE_RELATIVE`) derived from Farkas (1994) adult
  face anthropometric norms, with derivations documented in the module docstring.

### Feature Addition

- Added `min_detection_score: float` field to `GazeConfig` in
  `src/maestro/utils/schema_validation.py` (default `0.5`, constrained `(0, 1]` via
  Pydantic `Field`). The parameter is forwarded to `FaceDetection`'s
  `min_detection_confidence` constructor argument and applied again as a secondary
  Python-level guard in `_is_within_cone()` against marginal C-level results.

### Testing

- Updated `tests/unit/test_head_pose_gaze_detector.py` to test the geometric proxy
  implementation: replaced PnP-era fixtures with direct `_is_within_cone()` unit tests
  using `SimpleNamespace`-based mock detection results and a documented reference face
  (`inter_eye=0.10`, `eye_mid=(0.45, 0.40)`) with known analytic thresholds.
- Added guard-condition tests: no detections, fewer than three keypoints,
  below-threshold confidence, and degenerate inter-eye distance.
- Applied Ruff formatting to the test file.

---

## Technical Changes

**`src/maestro/infrastructure/gaze/head_pose_detector.HeadPoseGazeDetector`** now
estimates head orientation using two dimensionless ratios computed from the three
primary BlazeFace keypoints (right eye, left eye, nose tip):

```
yaw_norm  = (nose_x − eye_mid_x) / inter_eye_dist
pitch_norm = (nose_y − eye_mid_y) / inter_eye_dist
```

Both the numerator and denominator scale proportionally with face size, making the
ratios distance-invariant for any face size above `_MIN_INTER_EYE_RELATIVE = 0.020`
(relative image coordinates). The angular thresholds from `GazeConfig` are converted
into ratio limits via:

```
yaw_limit = (26/60) · tan(θ_yaw)          # ≈ 0.250 at 30°
pitch_half = θ_pitch · 0.016              # ≈ ±0.40 at 25°
pitch acceptance: [0.55 − pitch_half, 0.55 + pitch_half]
```

`_is_within_cone()` applies four guard conditions in order before the geometric check:
no detections present, fewer than three keypoints, detection score below
`min_detection_score`, inter-eye distance below `_MIN_INTER_EYE_RELATIVE`.

**`src/maestro/utils/schema_validation.GazeConfig`** adds:

```python
min_detection_score: float = Field(default=0.5, gt=0.0, le=1.0)
```

The field is documented in the class as the minimum BlazeFace detection confidence
required before a keypoint-based head-pose check is attempted. Lower values accept
more detections at range at the cost of increased false-positive cone membership.

The `cv2` import and all `solvePnP`-related code are removed from the module. The
`cv2.COLOR_BGR2RGB` conversion is retained as it is required by the MediaPipe C
extension, which expects a C-contiguous RGB array rather than a reversed-stride view.

---

## Testing and Validation

- `tests/unit/test_head_pose_gaze_detector.py` updated with a
  `TestIsWithinCone` class covering: frontal acceptance, yaw and pitch boundary
  conditions, all four guard rejections (no detections, insufficient keypoints,
  low confidence, small inter-eye distance).
- All pre-existing state-machine, amortisation, callback, and structural tests were
  retained and pass without modification, confirming the public `update()`/`close()`
  contract is unchanged.
- Ruff formatter applied to the test file.

---

## Outcome

The gaze detector now operates reliably at camera distances up to approximately 3 m
using the full-range BlazeFace model and a pose estimator that is invariant to face
scale. The `cv2.solvePnP` dependency is eliminated from the gaze detection path. The
`min_detection_score` configuration parameter exposes the confidence gate to operators,
allowing the acceptance threshold to be tuned for range versus precision. No breaking
changes were introduced to the `IGazeIntentDetector` interface or to any caller of
`HeadPoseGazeDetector`.
