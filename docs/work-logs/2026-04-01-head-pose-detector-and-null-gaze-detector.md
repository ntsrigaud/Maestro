# Work Log - Head-Pose Gaze Detector and Null Gaze Detector

## Context

After the gaze-aware activation design and the supporting interface/configuration schema were added, the repository still lacked concrete infrastructure implementations that could satisfy the new detector contract. This work introduces both the production-oriented head-pose gaze detector and a null-object fallback detector so the runtime can support gaze gating when enabled while preserving the existing gesture-only behavior when disabled.

## Scope of Work

**Concrete Detector Implementations**

- Added `HeadPoseGazeDetector` under `src/gesture_control/infrastructure/gaze/head_pose_detector.py`.
- Added `NullGazeDetector` under `src/gesture_control/infrastructure/gaze/null_detector.py`.
- Exported both implementations through `src/gesture_control/infrastructure/gaze/__init__.py`.

**Head-Pose Estimation**

- Implemented head-pose-based gaze intent estimation using MediaPipe Face Mesh and OpenCV `solvePnP`.
- Defined a six-point canonical 3-D face model for PnP pose estimation using stable facial landmarks.
- Added yaw/pitch extraction from the recovered rotation matrix and acceptance-cone validation against configured thresholds.

**Gaze State Machine**

- Implemented a four-state internal gaze detector state machine:
  - `ABSENT`
  - `ACQUIRING`
  - `ACTIVE`
  - `LOSING`
- Added stability arming and grace-period semantics based on the previously defined `GazeConfig` fields.
- Added optional `on_gaze_lost` callback support so the detector can force the gesture activation controller back to IDLE when gaze is lost.

**Null-Object Fallback**

- Implemented `NullGazeDetector` as an always-permitted detector for the `gaze.enabled = false` path.
- Preserved the pre-gaze runtime behavior without requiring conditional branches in future bootstrap or runtime integration code.

## Technical Changes

- `src/gesture_control/infrastructure/gaze/head_pose_detector.py`: Added `HeadPoseGazeDetector`, which structurally satisfies `IGazeIntentDetector`. The implementation constructs a MediaPipe Face Mesh instance, samples every `config.sample_interval_frames`, and caches the last permitted result between sampled frames to control CPU cost.
- `src/gesture_control/infrastructure/gaze/head_pose_detector.py`: Added a six-point 3-D face model and corresponding MediaPipe landmark indices for `cv2.solvePnP` pose recovery. Camera intrinsics are approximated from image width, and distortion coefficients are treated as zero to keep the estimator simple and suitable for commodity webcam usage.
- `src/gesture_control/infrastructure/gaze/head_pose_detector.py`: Added `_euler_yaw_pitch()` to derive yaw and pitch from the recovered rotation matrix and `_is_within_cone()` to classify head pose against configured acceptance thresholds.
- `src/gesture_control/infrastructure/gaze/head_pose_detector.py`: Implemented the gaze state machine transitions and timing logic. `ACTIVE` and `LOSING` both return `True` for `gaze_permitted`, while transition from `LOSING` to `ABSENT` invokes the optional gaze-lost callback.
- `src/gesture_control/infrastructure/gaze/null_detector.py`: Added `NullGazeDetector` with `update(frame) -> True` and a no-op `close()`, implementing the null-object strategy for disabled gaze gating.
- `src/gesture_control/infrastructure/gaze/__init__.py`: Re-exported `HeadPoseGazeDetector` and `NullGazeDetector` via `__all__`, establishing the package-level infrastructure surface for gaze detection implementations.

## Testing and Validation

- Executed `ruff check src/gesture_control/infrastructure/gaze` successfully.
- Executed `pytest` successfully: 150 tests passed, 0 failures, with one pre-existing warning in the synthetic training smoke test.
- No dedicated detector tests were added in this branch; validation focused on integration safety, lint cleanliness, and preservation of the existing repository test baseline.
- No test suppressions were added or removed.

## Outcome

The repository now contains concrete gaze detector infrastructure compatible with the previously introduced protocol and configuration schema. A production-oriented head-pose detector is available for gaze-aware activation, while a null-object detector preserves the existing gesture-only behavior when gaze gating is disabled. This branch extends the infrastructure layer without changing the public detector contract or requiring immediate runtime wiring changes.
