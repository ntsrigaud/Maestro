# Work Log - Gesture Activation and Command Dispatch

## Context

This work establishes the end-to-end gesture activation and command dispatch path for the runtime system, turning stabilized gesture predictions into application-specific presentation and meeting controls. The branch delivers the dispatch layer foundation, migrates the runtime to the IPN-derived gesture vocabulary, hardens the activation state machine for real presentation use, and eliminates repeated command execution from overlapping inference windows via edge-triggered dispatch.

## Scope of Work

**Dispatch Layer Foundation**

- Introduced a dedicated dispatch adapter layer under `src/gesture_control/infrastructure/dispatch/app_adapters`.
- Added `IApplicationAdapter` protocol to formalize application-specific command execution contracts.
- Implemented `KeyboardAdapter` as the only OS-level key delivery path, backed by `pyautogui` and guarded by a strict safe-key allowlist.
- Implemented `PowerPointAdapter` and `ZoomAdapter` to translate logical command IDs into application-specific keyboard shortcuts.
- Added `CommandDispatcher` to enforce command allowlisting, apply deque-based sliding-window rate limiting, isolate adapter exceptions, and return structured `DispatchResult` values.
- Added gesture-to-command configuration under `configs/gestures/command_map.yaml` and `configs/gestures/command_map_ipn.yaml`.

**IPN Vocabulary Migration**

- Migrated bootstrap configuration to `vocabulary_ipn.yaml`, `command_map_ipn.yaml`, and `production_ipn.yaml` as the canonical runtime sources.
- Replaced the original custom-dataset gesture vocabulary (`fist_hold`, `thumbs_up`, `v_sign`, `pinch_hold`) with the IPN-derived command gestures (`swipe_up`, `swipe_down`, `zoom_in`, `zoom_out`).
- Updated `CommandID` to retire legacy dispatch commands and align with the IPN command set (`toggle_pointer_mode`, `exit_presentation`).
- Deprecated `vocabulary_v1.yaml`, `command_map.yaml`, and `production.yaml` with explicit forwarding content to prevent silent misconfiguration.

**Activation Controller Hardening**

- Corrected activation hold timing so the hold timer begins on the first above-threshold activation gesture frame, even before temporal voting marks the prediction as `accepted=True`.
- Corrected ARMED idle-timeout semantics by refreshing activity on any non-unknown gesture, preventing the system from dropping back to IDLE during natural pauses.
- Extended `idle_timeout_ms` from 3 seconds to 5 minutes and raised the schema upper bound accordingly.
- Added structured observability logs across stabilization, activation, command routing, and dispatch result reporting.

**Edge-Triggered Dispatch**

- Implemented edge-triggered command dispatch in `GestureActivationController` using `_last_dispatched_label` rather than frame-by-frame previous-label tracking.
- Added `edge_trigger_enabled` to inference configuration and enabled it by default.
- Increased `command_cooldown_ms` from 500 ms to 1000 ms as a secondary guard.
- Reset dispatch memory on `unknown` predictions and on `IDLE → ARMED` transition so repeated identical gestures dispatch correctly only after a neutral interval or re-arming.

**Build and Dependency Updates**

- Added `pyautogui>=0.9.54` to project dependencies.
- Broadened the `pyautogui` import exception handling path so headless and CI environments degrade safely when the dependency is unavailable.

**Documentation**

- Updated `ARCHITECTURE_SPECIFICATION.md` to describe the dispatch architecture, IPN gesture vocabulary, activation semantics, and edge-trigger model.
- Updated `docs/IPN_INTEGRATION_GUIDE.md` to clarify `open_palm_hold` mapping to the IPN open-twice gesture.

## Technical Changes

- `src/gesture_control/infrastructure/dispatch/app_adapters/base.py`: Added `IApplicationAdapter` protocol with `target_app` and `execute(command_id)` contract.
- `src/gesture_control/infrastructure/dispatch/app_adapters/keyboard_adapter.py`: Added safe-key validation before calling `pyautogui.hotkey`; centralized all OS keystroke delivery in one adapter; added lazy import fallback for test and CI environments.
- `src/gesture_control/infrastructure/dispatch/app_adapters/powerpoint_adapter.py` and `zoom_adapter.py`: Added logical-command-to-shortcut translation with adapter-level exception capture and structured `DispatchResult` output.
- `src/gesture_control/infrastructure/dispatch/command_dispatcher.py`: Implemented three guard stages in order: allowlist enforcement, rolling-window rate limiting, and adapter execution. Successful dispatch timestamps are stored in a deque and pruned by time window.
- `src/gesture_control/pipeline/activation.py`: Evolved `GestureActivationController` from basic armed/cooldown routing into a production-oriented state machine. Added activation-hold fix, ARMED persistence fix, structured logging, and later edge-trigger suppression using `_last_dispatched_label` plus configurable `edge_trigger_enabled`.
- `src/gesture_control/app/bootstrap.py`: Switched runtime config loading to IPN config files, fixed the command map payload key to `gesture_commands`, and threaded `edge_trigger_enabled` into `GestureActivationController` construction.
- `src/gesture_control/utils/schema_validation.py`: Extended `InferenceConfig` with higher `idle_timeout_ms` upper bound and `edge_trigger_enabled: bool = True`.
- `configs/inference.yaml`: Extended idle timeout to `300000`, increased cooldown to `1000`, and enabled `edge_trigger_enabled` by default.
- `src/gesture_control/domain/enums.py`: Replaced legacy command IDs with the IPN-aligned command set.

## Testing and Validation

- Added `tests/unit/test_adapters.py` covering keyboard adapter allowlist behaviour, application adapter shortcut routing, dispatch failures for unmapped commands, and dispatcher-level allowlist and rate-limit guards.
- Added `tests/integration/test_gesture_dispatch_pipeline.py` as an end-to-end validation suite for:
  - activation hold and IDLE → ARMED transition,
  - ARMED → COOLDOWN → ARMED lifecycle,
  - secure command dispatch through `CommandDispatcher` and `PowerPointAdapter`,
  - allowlist rejection and rate-limit suppression,
  - IPN gesture vocabulary alignment,
  - activation hold persistence and ARMED-state timeout behaviour,
  - edge-trigger single-dispatch guarantee and same-gesture reset after `unknown`.
- Updated `tests/unit/test_dataset_builder.py` to align dataset builder expectations with the IPN vocabulary.
- Executed `pytest`: 150 tests pass, 0 failures. One warning remains expected in the synthetic training smoke test due to group-split fallback behaviour.
- `ruff check .` passes.
- No test suppressions were added or removed.

## Outcome

The system now has a complete and production-oriented gesture activation and command dispatch architecture. Stabilized gesture predictions are translated into secure, rate-limited, application-specific commands for PowerPoint and Zoom; the runtime operates exclusively on the IPN gesture vocabulary; the activation controller behaves correctly under realistic presentation timing; and edge-triggered dispatch enforces a one-gesture-one-command invariant. This branch introduces intentional domain-level command vocabulary changes by retiring legacy dispatch commands in favour of the IPN-aligned command set. Outside that intentional contract update, no additional breaking interface changes were introduced.
