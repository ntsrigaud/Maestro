# Work Log - Edge-Triggered Command Dispatch

## Context

During end-to-end validation of the IPN gesture vocabulary, a repeated-dispatch defect was identified: a single slow swipe gesture produces multiple consecutive accepted predictions across overlapping inference windows, causing the command to fire several times per physical gesture. An initial `_prev_label` frame-by-frame comparison approach failed because the temporal-voting window emits two to three unaccepted frames of the new label before the first `accepted=True` frame, causing `_prev_label` to be pre-advanced and the edge check to see no change on the eligible frame. This branch implements a corrected edge-trigger mechanism based on `_last_dispatched_label` and extends the cooldown window to provide a secondary guard.

## Scope of Work

- Implemented edge-triggered command dispatch in `GestureActivationController` using a `_last_dispatched_label` counter rather than a frame-by-frame label comparison.
- Added `edge_trigger_enabled` configuration field to `InferenceConfig` and `configs/inference.yaml`.
- Propagated `edge_trigger_enabled` from configuration through bootstrap to `GestureActivationController`.
- Extended `command_cooldown_ms` from 500 ms to 1000 ms in `configs/inference.yaml` as a secondary guard.
- Added integration tests for the edge-trigger invariant and the full predition-stream pipeline.
- Updated `ARCHITECTURE_SPECIFICATION.md` to document the edge-trigger design.

## Technical Changes

- `src/gesture_control/pipeline/activation.py`: Added `_last_dispatched_label: str` instance variable, initialised to `unknown_label`. In the ARMED routing path: (1) when the label is `unknown`, `_last_dispatched_label` is reset to `unknown_label` so that the same gesture performed again after a hand-neutral interval fires normally; (2) before command dispatch, if `edge_trigger_enabled` is `True` and `prediction.label == _last_dispatched_label`, the prediction is suppressed; (3) on successful dispatch, `_last_dispatched_label` is set to `prediction.label`. On IDLE → ARMED transition, `_last_dispatched_label` is also reset to ensure re-arming starts with a clean slate. Added an `edge_trigger_enabled: bool = True` constructor parameter. Extended class docstring with the design rationale for `_last_dispatched_label` over `_prev_label`.
- `src/gesture_control/utils/schema_validation.py`: Added `edge_trigger_enabled: bool = True` field to `InferenceConfig`.
- `src/gesture_control/app/bootstrap.py`: Passed `config.inference.edge_trigger_enabled` to `GestureActivationController`.
- `configs/inference.yaml`: Added `edge_trigger_enabled: true` and increased `command_cooldown_ms` to 1000 ms.

## Testing and Validation

- Added `TestEdgeTriggerSingleDispatch` in `tests/integration/test_gesture_dispatch_pipeline.py` with six tests:
  - Single swipe produces exactly one command across eight consecutive accepted predictions.
  - Same gesture re-dispatches after label transitions through `unknown`.
  - Repeated same gesture after cooldown expiry is suppressed by edge-trigger.
  - Different gesture after cooldown dispatches normally.
  - `edge_trigger_enabled=False` restores legacy cooldown-only behaviour.
  - Full pipeline test: prediction stream (`unknown × 2`, `swipe_right × 5`, `unknown × 2`) calls the keyboard adapter exactly once.
- Executed `pytest`: 150 tests pass, 0 failures (net addition of 8 tests).
- `ruff check .` — clean.
- No test suppressions were added or removed.

## Outcome

The system now enforces a strict one-gesture-one-command invariant for all ARMED-state dispatches. The edge-trigger is enabled by default and configurable; disabling it reverts to the prior cooldown-only guard. The extended 1000 ms cooldown provides a secondary safety margin in edge cases where temporal voting produces fast alternating accepted/unaccepted frames. No breaking changes to the public interface of `GestureActivationController` were introduced; `edge_trigger_enabled` defaults to `True` and the prior boolean semantics of all other parameters are preserved.
