# Work Log - Gaze-Driven Activation Refactor

## Context

This work removes the remaining gesture-based activation semantics from the runtime pipeline and completes the transition to gaze-driven activation. Prior iterations had already introduced gaze detection, gaze-aware runtime gating, and `force_idle()` / `rearm()` hooks on the activation controller, but the codebase still retained legacy concepts such as `activation_hold_ms`, `activation_gesture`, and `open_palm_hold` as a documented activation signal. This branch consolidates the final behavior: gaze is now the only activation mechanism, `open_palm_hold` is retained solely as a model output class, and the runtime re-arms command dispatch exclusively on gaze restoration.

## Scope of Work

**Runtime and Controller Refactor**

- Removed gesture-based activation hold logic from `GestureActivationController`.
- Removed `activation_gesture` and `activation_hold_ms` from controller construction and runtime configuration.
- Changed controller semantics so `IDLE` is exited only through explicit `rearm()` calls and not through any gesture label.
- Updated the runtime loop to call `rearm()` on the gaze-permitted rising edge, making gaze restoration the only path back to `ARMED`.

**Configuration and Schema Cleanup**

- Removed `activation_hold_ms` from `configs/inference.yaml`.
- Removed activation-hold validation from `InferenceConfig` in `schema_validation.py`.
- Removed `activation_gesture` from `configs/gestures/vocabulary_ipn.yaml`.
- Clarified `configs/gestures/command_map_ipn.yaml` so `open_palm_hold` is explicitly documented as an ignored model output with no command mapping.

**Gaze Detector Stabilization**

- Updated the head-pose detector implementation to improve gaze acquisition stability.
- Ensured detector input arrays are C-contiguous before downstream OpenCV operations, avoiding memory-layout-related instability.
- Updated gaze detector tests to reflect hysteresis-driven acquisition/loss behavior.

**Testing and Documentation Alignment**

- Migrated activation-controller unit tests from the legacy gesture-hold model to the new gaze-aware rearm/force-idle model.
- Updated dispatch integration tests to align with gaze-driven activation semantics.
- Removed obsolete activation-hold assumptions from checkpoint loader tests.
- Updated `README.md`, `docs/architecture/system-overview.md`, `docs/architecture/process-view.md`, and `ARCHITECTURE_SPECIFICATION.md` so they describe gaze-driven activation rather than gesture-triggered arming.
- Clarified in documentation that `open_palm_hold` remains in the model vocabulary but no longer performs any activation or command function.

## Technical Changes

- `src/gesture_control/pipeline/activation.py`: Removed all activation-hold timing state and label-specific activation handling. The controller now starts in `IDLE`, immediately returns `None` while idle, and relies on external `rearm()` to enter `ARMED`. `force_idle()` remains the explicit reset path for gaze-loss events. Edge-triggered dispatch and cooldown semantics remain unchanged.
- `src/gesture_control/app/runtime.py`: Added gaze-rising-edge detection via a `_prev_gaze_permitted` flag. When `gaze_permitted` transitions from `False` to `True`, the runtime calls `activation_controller.rearm()`. This makes gaze restoration the sole activation event.
- `src/gesture_control/app/bootstrap.py`: Removed vocabulary loading for activation purposes and stopped passing `activation_gesture` / `activation_hold_ms` into `GestureActivationController`.
- `src/gesture_control/utils/schema_validation.py`: Removed `activation_hold_ms` from `InferenceConfig`, simplifying the inference schema to gaze-driven activation parameters only.
- `configs/inference.yaml`: Deleted `activation_hold_ms` from runtime defaults.
- `configs/gestures/vocabulary_ipn.yaml`: Removed the `activation_gesture` field while retaining `open_palm_hold` in the gesture class list for model compatibility.
- `configs/gestures/command_map_ipn.yaml`: Updated comments to document that `open_palm_hold` is ignored by dispatch.
- `src/gesture_control/infrastructure/gaze/head_pose_detector.py`: Refined gaze acquisition behavior and memory handling to improve runtime stability under hysteresis.

## Testing and Validation

- Updated `tests/unit/test_activation_controller.py` to validate the gaze-driven model:
  - controller starts in `IDLE`,
  - `rearm()` transitions `IDLE -> ARMED`,
  - `force_idle()` returns the controller to `IDLE`,
  - dispatch still works correctly after re-arming.
- Updated `tests/integration/test_gesture_dispatch_pipeline.py` to remove activation-hold setup and arm the controller via `rearm()`.
- Updated `tests/unit/test_head_pose_gaze_detector.py` to match the stabilized hysteresis behavior.
- Updated `tests/unit/test_checkpoint_loader.py` to remove activation-hold assumptions.
- Executed `pytest`: 204 tests pass, 0 failures. One warning remains expected in the synthetic training smoke test due to the existing group-split fallback path.
- Executed `ruff check .`: all checks pass.
- No test suppressions were added or removed.

## Outcome

The system now uses a single activation model: gaze drives entry into and exit from the command-accepting state, while gestures are used only for command classification and dispatch. This removes the last conceptual overlap between gaze gating and gesture-based arming, simplifies configuration and controller logic, and makes the runtime behavior align cleanly with the implemented gaze-aware architecture. No new public API surface was introduced beyond the removal of obsolete activation-hold configuration and constructor parameters.
