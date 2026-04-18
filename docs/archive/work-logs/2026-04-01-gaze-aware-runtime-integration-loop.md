# Work Log - Gaze-Aware Runtime Integration Loop

## Context

Previous branches introduced the gaze-aware activation design, configuration schema, concrete detector implementations, and the activation-controller reset hook needed for gaze loss handling. The runtime still executed as a gesture-only pipeline: no gaze detector was constructed at bootstrap, no gaze signal influenced frame processing, and detector resources were not part of the application shutdown path. This work completes the first application-layer integration by wiring gaze detection into the container and enforcing a gaze-aware safety gate in the main event loop.

## Scope of Work

**Application Bootstrap Integration**

- Extended `AppContainer` with a `gaze_detector` dependency typed as `IGazeIntentDetector`.
- Updated `build_container()` to construct either `HeadPoseGazeDetector` or `NullGazeDetector` based on `config.inference.gaze.enabled`.
- Registered `activation_controller.force_idle` as the gaze-lost callback when gaze-aware activation is enabled.

**Runtime Event Loop Integration**

- Inserted gaze evaluation into the runtime loop immediately after frame capture and before hand-landmark extraction.
- Added an early-exit gaze gate that bypasses landmark extraction, feature transformation, buffering, model inference, stabilization, and dispatch when gaze permission is revoked.
- Rendered an overlay state with `gaze_permitted=False` while preserving the current activation mode for operator feedback.
- Propagated `gaze_permitted` into the normal runtime state path when gaze access is granted.

**Resource Lifecycle Management**

- Added `gaze_detector.close()` to runtime shutdown so detector-owned resources are always released in both normal and exceptional paths.
- Added unit coverage verifying gaze detector teardown when the runtime fails during frame capture.

## Technical Changes

- `src/gesture_control/app/bootstrap.py`: Added `gaze_detector: IGazeIntentDetector` to `AppContainer`. `build_container()` now reads `config.inference.gaze`; when `enabled=True`, it constructs `HeadPoseGazeDetector(config=gaze_cfg)` and registers `activation_controller.force_idle` via `set_on_gaze_lost()`. When `enabled=False`, it constructs `NullGazeDetector()` so the runtime path remains branch-free after container creation.
- `src/gesture_control/app/runtime.py`: Added `gaze_detector` to `RuntimeDependencies`. In the main event loop, `gaze_permitted = self._container.gaze_detector.update(frame)` is evaluated immediately after frame capture. When `gaze_permitted` is `False`, the runtime constructs `RuntimeState(mode=str(self._container.activation_controller.mode), gaze_permitted=False)`, renders the overlay, handles window key input, increments `processed_frames`, and `continue`s without invoking downstream gesture-processing stages.
- `src/gesture_control/app/runtime.py`: In the normal processing path, `gaze_permitted` is carried into the standard `RuntimeState` object so overlay rendering and future status surfaces can reflect detector output during active processing.
- `src/gesture_control/app/runtime.py`: Added `self._container.gaze_detector.close()` to the `finally` block, alongside existing frame source and landmark extractor teardown.
- `tests/unit/test_runtime.py`: Extended the runtime failure fixture with a mocked `gaze_detector` and added an assertion that `gaze_detector.close()` is invoked when `frame_source.read()` raises.

## Testing and Validation

- Executed `pytest tests/unit/test_runtime.py -q` successfully.
- Verified the runtime failure path closes `frame_source`, `gaze_detector`, and `landmark_extractor` exactly once.
- Verified the integration preserves the legacy non-gaze path through `NullGazeDetector` when `config.inference.gaze.enabled` is `False`.
- No test suppressions were added or removed.

## Outcome

The gaze-aware activation design is now connected to the application layer. Bootstrap selects an appropriate detector implementation, gaze loss can force the activation controller back to `IDLE`, and the runtime loop treats gaze permission as an outer safety gate that can bypass the downstream gesture pipeline before any gesture-processing work is performed. This branch extends runtime orchestration and resource management without introducing breaking public API changes beyond the container-level addition of the gaze detector dependency.
