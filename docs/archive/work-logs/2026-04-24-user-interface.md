# Work Log — PySide6 User Interface: Runtime Control, Telemetry, and GUI Architecture

**Date:** 2026-04-24  
**Branch:** `feat/user-interface`  
**Author:** @ntsrigaud

---

## Context

The runtime had become operationally complete but remained CLI-driven. Model source
selection, runtime start/stop, preview-window toggling, and live telemetry were only
accessible through configuration edits and terminal execution. This branch adds a
PySide6 control surface on top of the existing runtime without changing the underlying
inference pipeline contract. The goal is to make the presentation-controller runtime
launchable and observable from a desktop GUI while preserving the existing CLI path and
keeping Qt dependencies behind a dedicated import boundary.

---

## Scope of Work

### GUI Application and Runtime Bridge

- Added a dedicated UI entry point in `scripts/run_ui.py` and the Qt application boot
  path in `src/gesture_control/app/ui/app_ui.py`.
- Introduced a new `src/gesture_control/app/ui/` package containing the GUI control
  layer: `controller.py`, `main_window.py`, `config_panel.py`, `model_panel.py`,
  `launcher_panel.py`, and `status_monitor.py`.
- Extended `src/gesture_control/app/runtime.py` with an optional
  `on_state: Callable[[RuntimeState], None] | None` callback so runtime state can be
  emitted to the GUI once per processed frame without affecting the CLI path.
- Wrapped runtime execution in a private worker `QThread` via `RuntimeController`,
  isolating `build_container()` and `Runtime.run()` from the Qt event loop.

### User Interface Components

- Implemented `MainWindow` as the root orchestration surface: owns the
  `RuntimeController`, wires panel signals, updates a session log, and manages a system
  tray icon.
- Implemented `LauncherPanel` as a four-state start/stop widget (`idle`, `building`,
  `running`, `error`) with a `Show preview window` toggle.
- Implemented `StatusMonitor` to display live runtime telemetry derived from
  `RuntimeState`.
- Implemented `ConfigPanel` as a form-based editor for `AppConfig` and
  `InferenceConfig` with in-memory validation and atomic YAML writes.
- Implemented `ModelPanel` as a two-tab model selector supporting Hugging Face source
  metadata and optional local artifact selection.

### Runtime and Preview Behavior

- Disabled the OpenCV preview window by default in the UI flow and in runtime-facing
  defaults.
- Added `_imshow()` in `src/gesture_control/app/runtime.py` to gracefully disable the
  preview window when `cv2.imshow()` is unavailable, such as in headless builds or Qt
  thread-conflict scenarios.
- Preserved preview support behind the launcher checkbox rather than removing it from
  the runtime.

### Configuration, Dependencies, and Documentation

- Added a `ui` optional dependency group in `pyproject.toml` for PySide6 and aligned
  development dependencies for Qt-based testing.
- Updated `configs/app.yaml` and `configs/inference.yaml` formatting and runtime
  defaults to match the new GUI launch posture.
- Added `docs/architecture/ui-view.md` and updated the architecture suite
  (`data-flow.md`, `development-view.md`, `logical-view.md`, `physical-view.md`,
  `process-view.md`, `system-overview.md`) to document the PySide6 control layer and
  dual-thread execution model.
- Updated `README.md` to reflect the existence of the GUI application.

---

## Technical Changes

**`src/gesture_control/app/runtime.py`** now accepts an optional `on_state` callback.
The callback is invoked exactly once per processed frame in both the gaze-blocked path
and the main inference path, allowing the GUI to observe `RuntimeState` without pulling
from the runtime. This preserves the existing CLI behavior because the non-GUI entry path
passes no callback.

The same module adds `_imshow(window_name, frame) -> bool`, which catches `cv2.error`
and logs a warning once when OpenCV display support is unavailable. On failure,
`show_window` is disabled dynamically rather than terminating runtime execution.

**`src/gesture_control/app/ui/controller.py`** introduces a two-layer controller model:
`RuntimeController` in the Qt main thread and a private `_RuntimeThread` worker. The
worker constructs the application container, instantiates `Runtime(on_state=...)`, and
emits throttled `RuntimeState` updates via Qt signals. The emit cadence is capped by
`emit_every` so UI refresh does not track camera FPS one-for-one.

**`src/gesture_control/app/ui/main_window.py`** is the composition root of the GUI
subsystem. It owns the controller, launcher, monitor, config form, model selector, log
panel, and tray icon. All communication from panels to runtime goes through `MainWindow`;
no widget imports runtime or bootstrap directly. Accepted gestures update the tray icon,
surface a transient notification, and append a timestamped log entry.

**`src/gesture_control/app/ui/config_panel.py`** builds a reflective form editor from
`AppConfig` and `InferenceConfig` Pydantic metadata. The panel validates draft values in
memory, exposes `current_config()`, disables apply while runtime is active, and writes
YAML atomically through a temp-file swap.

**`src/gesture_control/app/ui/model_panel.py`** adds a two-tab source selector: Hugging
Face metadata fields on the first tab and a local `.pt` picker on the second. The panel
returns `artifact_path()` and `hf_revision_override()` for `RuntimeController.start()`.

**`src/gesture_control/app/ui/launcher_panel.py`** implements an explicit state machine
for runtime lifecycle control and makes preview display opt-in through a checkbox that
defaults to `False`.

**Documentation updates** introduce `docs/architecture/ui-view.md` as a dedicated
architectural view for the GUI layer and revise the other architecture views to explain
state delivery, UI deployment, and the dual-thread Qt + runtime execution model.

---

## Testing and Validation

- Added `tests/unit/test_ui_runtime_on_state.py` to verify `Runtime.on_state`
  invocation behavior.
- Added `tests/unit/test_ui_runtime_controller.py` to verify worker-thread lifecycle,
  stop behavior, runtime start signaling, and error propagation.
- Added widget-level unit suites for `ConfigPanel`, `ModelPanel`, `LauncherPanel`,
  `StatusMonitor`, and system tray / preview toggle behavior:
  `test_ui_config_panel.py`, `test_ui_model_panel.py`, `test_ui_launcher_panel.py`,
  `test_ui_status_monitor.py`, `test_ui_tray.py`.
- Updated `tests/conftest.py` to skip UI test collection automatically when PySide6 is
  not installed, mirroring the existing optional-dependency handling for training tests.
- Applied Black formatting across UI and touched core modules; removed unused imports and
  corrected Qt import ordering to satisfy linting.

---

## Outcome

The system now supports a graphical control plane for the existing presentation runtime
without forking the runtime architecture or coupling Qt into the CLI path. Operators can
launch and stop the system, inspect live state, toggle the preview window, edit runtime
configuration, and select model source behavior from a desktop window and system tray.
The runtime remains single-source and testable, while the GUI is isolated behind a
dedicated package, script entry point, and optional dependency group. No breaking changes
were introduced to the existing CLI execution path. Net change: 32 files changed,
2,321 insertions, 101 deletions.
