# UI View

## 1. Overview

### 1.1 Purpose

The UI layer provides a graphical control surface for Maestro. Its
responsibilities are confined to three concerns:

- **Lifecycle management**: starting and stopping the inference runtime.
- **Configuration editing**: presenting `AppConfig` and `InferenceConfig` fields as an editable
  form with live Pydantic validation, and writing validated YAML only after the runtime stops.
- **State observation**: displaying per-frame telemetry (`RuntimeState`) and dispatched gesture
  events via signal-driven widgets and a system tray icon.

The UI is explicitly not part of the inference pipeline. It never reads from or writes to any
pipeline component during a running session. All interaction with the runtime is mediated through
a single controller object.

### 1.2 Design Goals

| Goal                        | Mechanism                                                                |
| --------------------------- | ------------------------------------------------------------------------ |
| Non-blocking inference loop | `Runtime.run()` executes in a `QThread` worker; Qt event loop in main    |
| Thread-safe state delivery  | Qt queued signal connections; no shared mutable state in the hot path    |
| Configuration safety        | YAML written only when runtime is stopped; Pydantic validates all drafts |
| Minimal inference coupling  | One optional `on_state` callback parameter added to `Runtime.__init__`   |
| Ambient feedback            | `QSystemTrayIcon` with colour-coded icons; visible when window minimised |
| Single deployment unit      | PySide6 native; no web server, port, or browser dependency               |

### 1.3 Entry Points

| Script                    | Purpose                                              |
| ------------------------- | ---------------------------------------------------- |
| `scripts/run_ui.py`       | Qt GUI entry point — launches `run_ui()` from app_ui |
| `scripts/run_realtime.py` | CLI-only entry point — no PySide6 import; unchanged  |

The inference CLI path (`run_realtime.py` → `app/main.py`) never imports PySide6. The two
entry points are completely isolated at the import boundary.

---

## 2. Architectural Design

### 2.1 Pattern

The UI layer follows an **event-driven, signal-slot architecture** on top of Qt's queued
connection mechanism. The pattern maps cleanly to the system's existing structure:

- `Runtime` is a pure computation object with no UI dependency.
- `RuntimeController` acts as the **presenter**: it owns the worker thread, translates lifecycle
  events into Qt signals, and throttles state delivery.
- Panels are **passive views**: they receive signals and emit user-intent signals; they never
  call into the runtime or controller directly.
- `MainWindow` acts as the **mediator**: it wires signals between panels and the controller
  without exposing either side to the other.

### 2.2 Layer Diagram

```mermaid
graph TD
    subgraph "Main Thread — Qt Event Loop"
        MW["MainWindow\n(Mediator)"]
        LP["LauncherPanel\n(Start/Stop/Checkbox)"]
        SM["StatusMonitor\n(Telemetry display)"]
        CP["ConfigPanel\n(Form editor)"]
        MP["ModelPanel\n(HF / Local selector)"]
        TI["QSystemTrayIcon\n(Ambient indicator)"]
        LOG["Session Log\n(QTextEdit)"]
        MW --- LP
        MW --- SM
        MW --- CP
        MW --- MP
        MW --- TI
        MW --- LOG
    end

    subgraph "Worker Thread — QThread"
        RT["_RuntimeThread\nrun() → build_container() + Runtime.run()"]
    end

    subgraph "Inference Layer (no UI dependency)"
        BC["build_container()"]
        RM["Runtime"]
    end

    LP -->|"start_requested signal"| MW
    LP -->|"stop_requested signal"| MW
    MW -->|"controller.start()"| RC["RuntimeController"]
    MW -->|"controller.stop()"| RC
    RC -->|"spawns"| RT
    RT -->|"build_container()"| BC
    BC -->|"AppContainer"| RM
    RM -->|"on_state callback"| RT
    RT -->|"state_ready signal (throttled)"| RC
    RC -->|"state_updated signal"| SM
    RC -->|"state_updated signal"| TI
    RC -->|"state_updated signal"| LOG
    RC -->|"runtime_started signal"| LP
    RC -->|"runtime_started signal"| CP
    RC -->|"runtime_stopped signal"| LP
    RC -->|"runtime_stopped signal"| CP
    RC -->|"runtime_stopped signal"| TI
    RC -->|"error_occurred signal"| LP
```

### 2.3 Package Structure

```
src/maestro/app/
├── bootstrap.py          # Unchanged — composition root
├── runtime.py            # Minimal change: on_state optional callback
├── main.py               # Unchanged — CLI entry point
└── ui/
    ├── __init__.py
    ├── app_ui.py          # run_ui() — QApplication bootstrap
    ├── controller.py      # RuntimeController + _RuntimeThread
    ├── main_window.py     # MainWindow — mediator and signal router
    ├── launcher_panel.py  # Start/Stop button + status pill + checkbox
    ├── status_monitor.py  # Per-frame telemetry widgets
    ├── config_panel.py    # Pydantic-driven config form
    └── model_panel.py     # HF / Local model source selector
```

**Dependency rule enforced by implementation**: No file outside `app/ui/` imports from
`app/ui/`. `app/runtime.py` and `app/main.py` contain no PySide6 imports.

---

## 3. Thread-Safety and Concurrency Model

### 3.1 Threading Architecture

```
┌─────────────────────────────────────────────────────────────┐
│  Main Thread                                                │
│  QApplication.exec() — Qt event loop                        │
│  Owns: MainWindow, all panels, RuntimeController            │
└────────────────────┬────────────────────────────────────────┘
                     │ QThread.start()
                     ▼
┌─────────────────────────────────────────────────────────────┐
│  Worker Thread (_RuntimeThread)                             │
│  Executes: build_container() → Runtime.run()                │
│  Owns: Runtime instance, frame counter                      │
│  Emits: state_ready, build_succeeded, error_emitted         │
└────────────────────┬────────────────────────────────────────┘
                     │ on_state callback (in worker thread)
                     ▼
┌─────────────────────────────────────────────────────────────┐
│  Runtime.run() inference loop                               │
│  Calls self._on_state(state) every frame                    │
│  on_state → _RuntimeThread._handle_state() → emit signal    │
└─────────────────────────────────────────────────────────────┘
```

The two threads share **no mutable state** except:

- `Runtime._running` — a single `bool` set by `Runtime.stop()`. Under the Python GIL, a
  single boolean write is atomic. No lock is required.
- `_RuntimeThread._runtime` — written once in `run()` before `stop()` is ever called;
  the reference is published to the main thread only via the Qt signal-slot mechanism.

### 3.2 Signal Delivery

Qt's queued connection delivers every emitted signal through the receiving thread's event queue.
When the worker thread emits `state_ready.emit(state)`, the `RuntimeState` object is placed in
the main thread's event queue and processed during the next event-loop iteration. No locking,
no polling, no `Queue.get()`.

All cross-thread signal connections in `RuntimeController.start()` use the default auto
connection, which Qt automatically promotes to a queued connection when source and destination
are in different threads.

### 3.3 State Throttling

`Runtime.run()` calls `on_state` on every frame (30 Hz). The `_RuntimeThread._handle_state`
method maintains a private frame counter and emits `state_ready` only every `emit_every` frames
(default 3). This caps UI update traffic at approximately 10 Hz, preventing signal flooding
while preserving interactive responsiveness.

The frame counter is private to the worker thread; no synchronization is needed to read or
increment it.

### 3.4 Shutdown Sequence

```
User clicks Stop (or closeEvent fires)
    │
    ▼
RuntimeController.stop()
    │
    └─→ _RuntimeThread.stop()
            │
            └─→ Runtime.stop()          ← sets _running = False (GIL-safe)
                        │
                        ▼
                Runtime.run() while loop exits on next iteration
                        │
                        ▼
                _RuntimeThread.run() returns
                        │
                        ▼
                QThread.finished emitted (worker thread)
                        │
                        ▼ queued connection
                RuntimeController.runtime_stopped emitted (main thread)
                        │
                        ▼
                LauncherPanel, ConfigPanel, tray icon updated
```

`RuntimeController.wait(timeout_ms)` is available for deterministic teardown in `closeEvent`
and test fixtures. It blocks the main thread until the worker exits or the timeout elapses.

---

## 4. Integration with the Inference Pipeline

### 4.1 Change Surface

The only modification to pre-existing inference code is a single optional constructor parameter
in `Runtime`:

```python
class Runtime:
    def __init__(
        self,
        container: RuntimeDependencies,
        on_state: Callable[[RuntimeState], None] | None = None,
    ) -> None:
```

When `on_state` is `None` (the CLI path), `Runtime` behaves identically to its pre-UI
implementation. No inference-path code was changed to support the UI.

### 4.2 RuntimeState as the UI Data Contract

`RuntimeState` is produced per-frame by `Runtime.run()` and is the sole data type crossing the
thread boundary. It is a frozen dataclass defined in `domain/entities.py` and carries all
information the UI requires:

| Field                      | Type          | Consumer               |
| -------------------------- | ------------- | ---------------------- |
| `mode`                     | `str`         | StatusMonitor, tray    |
| `gaze_permitted`           | `bool`        | StatusMonitor, tray    |
| `last_label`               | `str \| None` | StatusMonitor, log     |
| `last_confidence`          | `float`       | StatusMonitor          |
| `info["fps"]`              | `float`       | StatusMonitor          |
| `info["latency_e2e_ms"]`   | `float`       | StatusMonitor          |
| `info["latency_infer_ms"]` | `float`       | StatusMonitor          |
| `info["valid_ratio"]`      | `float`       | StatusMonitor          |
| `info["accepted"]`         | `bool`        | Tray notification, log |

The UI never queries any pipeline object directly. All telemetry arrives through `state_updated`.

### 4.3 Container Construction

`build_container()` is invoked once per session start, inside the worker thread. It reads YAML
configs, validates them via Pydantic, downloads or loads the model artifact, and opens the
webcam. This operation takes approximately 2–5 seconds and is intentionally not called from the
main thread. The `build_succeeded` signal notifies the UI when construction completes, at which
point the launcher transitions from "Starting…" to "Running".

### 4.4 cv2.imshow Behaviour

When the user enables "Show preview window", the runtime calls `cv2.imshow()` from the worker
thread. Because the Qt event loop is running in the main thread, this call may fail on headless
OpenCV builds or when a Qt/OpenCV display conflict arises. The `_imshow()` helper in
`runtime.py` wraps `cv2.imshow` in a `try/except cv2.error` block, logs a warning, and returns
`False`. On failure, `container.show_window` is set to `False` to prevent repeated error logging
on subsequent frames.

The "Show preview window" checkbox in `LauncherPanel` defaults to unchecked; preview is opt-in.

---

## 5. Component Breakdown

### 5.1 `app_ui.py` — Application Bootstrap

```
run_ui(config_path: str = "configs") → int
```

Creates or reuses the `QApplication` singleton, instantiates `MainWindow`, calls `window.show()`,
and enters the Qt event loop with `app.exec()`. Returns the exit code. This is the only
function called from `scripts/run_ui.py`.

### 5.2 `controller.py` — RuntimeController and \_RuntimeThread

**`_RuntimeThread(QThread)`** — private worker:

| Signal            | Emitted When                                     |
| ----------------- | ------------------------------------------------ |
| `state_ready`     | Every `emit_every` frames during `Runtime.run()` |
| `build_succeeded` | `build_container()` returned successfully        |
| `error_emitted`   | Unhandled exception in `run()`                   |

The thread owns the `Runtime` reference. It is constructed with all session parameters
(`config_path`, `artifact_path`, `hf_revision_override`, `show_window`, `emit_every`) and
does not accept changes after `start()`.

**`RuntimeController(QObject)`** — public bridge:

| Signal            | Emitted When                                         |
| ----------------- | ---------------------------------------------------- |
| `state_updated`   | Forwarded from `_RuntimeThread.state_ready`          |
| `runtime_started` | Forwarded from `_RuntimeThread.build_succeeded`      |
| `runtime_stopped` | `_RuntimeThread.finished` (QThread lifecycle signal) |
| `error_occurred`  | Forwarded from `_RuntimeThread.error_emitted`        |

Public methods: `start()`, `stop()`, `wait()`, `is_running`.

`start()` is a no-op if a thread is already running. `stop()` is a no-op if no thread is live.
Both are safe to call from the main thread at any time.

### 5.3 `main_window.py` — MainWindow

`MainWindow(QMainWindow)` is the composition root of the UI layer. It:

- Instantiates all panels and `RuntimeController`.
- Pre-populates `ModelPanel` from `model_source.yaml` if available.
- Wires all signals between panels and the controller.
- Owns the `QSystemTrayIcon` and session log `QTextEdit`.
- Implements minimize-to-tray via `changeEvent` override.
- Implements graceful shutdown via `closeEvent` override (stops runtime before closing).

`MainWindow` does not contain any domain or pipeline logic. All business decisions are
delegated to `RuntimeController` and the panel signals.

**Tray icon colour convention** (mirrors `OverlayRenderer` border colours):

| Colour | Hex       | Meaning                               |
| ------ | --------- | ------------------------------------- |
| Grey   | `#888888` | Runtime stopped or gaze absent        |
| Green  | `#44AA44` | Runtime running, gaze active          |
| Amber  | `#FFA500` | Gesture accepted / command dispatched |

A brief amber flash (300 ms) followed by a return to green accompanies each dispatched command.
Tray notifications display the dispatched gesture label for 800 ms.

**Session log**: A read-only `QTextEdit` (maximum height 120 px) appends timestamped entries on
mode transitions and dispatched commands. It never receives raw telemetry values — only
state-change events derived from `RuntimeState`.

### 5.4 `launcher_panel.py` — LauncherPanel

`LauncherPanel(QWidget)` presents a single button whose label and enabled state reflect the
current system state, a colour-coded status pill, and an opt-in "Show preview window" checkbox.

| System State | Button Label | Pill Colour |
| ------------ | ------------ | ----------- |
| `idle`       | Start        | Grey        |
| `building`   | Starting…    | Amber       |
| `running`    | Stop         | Green       |
| `error`      | Retry        | Red         |

State transitions are driven by slots (`on_runtime_started`, `on_runtime_stopped`, `on_error`,
`on_start_clicked`) connected from `MainWindow`. The panel emits `start_requested` and
`stop_requested` signals; it never calls `RuntimeController` directly.

The `show_window` property reads the checkbox state and is consumed by `MainWindow._on_start_requested()`.

### 5.5 `status_monitor.py` — StatusMonitor

`StatusMonitor(QWidget)` subscribes to `RuntimeController.state_updated` via its `on_state(RuntimeState)` slot. It displays eight fields:

| Row             | Widget         | Source                           |
| --------------- | -------------- | -------------------------------- |
| Mode            | `QLabel`       | `state.mode.upper()`             |
| Gaze            | `QLabel`       | `state.gaze_permitted`           |
| Gesture         | `QLabel`       | `state.last_label`               |
| Confidence      | `QProgressBar` | `state.last_confidence × 100`    |
| FPS             | `QLabel`       | `state.info["fps"]`              |
| Latency (e2e)   | `QLabel`       | `state.info["latency_e2e_ms"]`   |
| Latency (infer) | `QLabel`       | `state.info["latency_infer_ms"]` |
| Valid ratio     | `QLabel`       | `state.info["valid_ratio"]`      |

The widget contains no state beyond what Qt widget internals require. It never calls into the
runtime or stores `RuntimeState` objects.

### 5.6 `config_panel.py` — ConfigPanel

`ConfigPanel(QWidget)` generates a form from `AppConfig` and `InferenceConfig` Pydantic model
fields. Widget type is selected by field annotation:

| Annotation     | Widget           | Notes                                    |
| -------------- | ---------------- | ---------------------------------------- |
| `bool`         | `QCheckBox`      | Checked if default is `True`             |
| `int`          | `QSpinBox`       | `ge`/`le` constraints applied as min/max |
| `float`        | `QDoubleSpinBox` | `ge`/`le` constraints applied as min/max |
| `str`          | `QLineEdit`      |                                          |
| `Literal[...]` | `QComboBox`      | All literal values as selectable options |
| Nested/complex | Skipped          | `command_map`, `app_name`, `GazeConfig`  |

On every field-change event, the panel builds a draft dict from all widget values and attempts
`RuntimeConfig(**draft)` construction. If Pydantic raises `ValidationError`, the draft is
marked invalid and the Apply button is disabled. If valid and the runtime is stopped, Apply
is enabled.

The "Apply & Restart" button triggers `apply_requested`, which:

1. Writes the validated config to `configs/app.yaml` and `configs/inference.yaml` using atomic
   `.yaml.tmp` → `.replace(path)` writes (safe against partial-write corruption).
2. Signals `MainWindow._on_apply_requested()`, which stops the runtime, waits for the worker
   to exit, and calls `_on_start_requested()` to restart with the new config.

YAML files are **never** written while `_runtime_active` is `True`.

Signals emitted: `config_changed(bool)` on every validation cycle, `apply_requested` on click.

### 5.7 `model_panel.py` — ModelPanel

`ModelPanel(QWidget)` presents two tabs backed by a `QTabWidget`:

**Tab 0 — Hugging Face**: `repo_id`, `revision`, and `filename` `QLineEdit` fields, pre-populated
from `model_source.yaml` via `ModelSourceConfig`. Revision is passed as `hf_revision_override`
to `RuntimeController.start()`; an empty string is treated as `None` (use the default revision).

**Tab 1 — Local Artifact**: A file label and "Browse…" button open a `QFileDialog` filtered to
`.pt` / `.pth` files. The selected path is passed as `artifact_path` to `RuntimeController.start()`.

When the Hugging Face tab is active, `artifact_path()` returns `None` (HF download path used by
bootstrap). When the Local tab is active, `hf_revision_override()` returns `None`.

---

## 6. Data Flow

### 6.1 Start Session Data Flow

```mermaid
sequenceDiagram
    participant User
    participant LP as LauncherPanel
    participant MW as MainWindow
    participant RC as RuntimeController
    participant WT as _RuntimeThread (worker)
    participant RM as Runtime

    User->>LP: clicks Start
    LP->>MW: start_requested signal
    MW->>LP: on_start_clicked() → "Starting…"
    MW->>RC: start(config_path, artifact_path, hf_override, show_window)
    RC->>WT: QThread.start()
    WT->>WT: build_container()
    WT-->>RC: build_succeeded signal (queued)
    RC-->>LP: runtime_started signal → "Running"
    RC-->>CP: set_runtime_active(True)
    WT->>RM: Runtime.run()
    loop every frame
        RM->>WT: on_state(state) [every 3rd frame]
        WT-->>RC: state_ready signal (queued)
        RC-->>SM: state_updated → on_state(state)
        RC-->>MW: state_updated → _on_tray_update(state)
        RC-->>MW: state_updated → _on_log_event(state)
    end
```

### 6.2 Stop Session Data Flow

```mermaid
sequenceDiagram
    participant User
    participant LP as LauncherPanel
    participant MW as MainWindow
    participant RC as RuntimeController
    participant WT as _RuntimeThread
    participant RM as Runtime

    User->>LP: clicks Stop
    LP->>MW: stop_requested signal
    MW->>RC: stop()
    RC->>WT: stop()
    WT->>RM: runtime.stop() [_running = False]
    RM->>RM: while loop exits
    WT-->>RC: QThread.finished (queued)
    RC-->>LP: runtime_stopped → "Idle"
    RC-->>CP: set_runtime_active(False)
    RC-->>MW: runtime_stopped → tray grey
```

### 6.3 Config Apply Data Flow

```mermaid
sequenceDiagram
    participant User
    participant CP as ConfigPanel
    participant MW as MainWindow
    participant RC as RuntimeController

    User->>CP: edits field
    CP->>CP: Pydantic validation
    CP-->>MW: config_changed(valid)
    User->>CP: clicks Apply & Restart
    CP->>CP: atomic YAML write (tmp → replace)
    CP-->>MW: apply_requested signal
    MW->>RC: stop() + wait(3000 ms)
    MW->>MW: _on_start_requested()
    note over MW: Runtime restarts with new YAML
```

---

## 7. Design Decisions and Trade-offs

### 7.1 PySide6 over Alternatives

| Option                     | Reason Rejected                                          |
| -------------------------- | -------------------------------------------------------- |
| FastAPI + browser frontend | HTTP stack, port conflicts, Node dependency, IPC latency |
| Tkinter                    | Poor widget quality, no system tray support              |
| Electron / Tauri           | Heavy runtime, complex deployment                        |
| **PySide6 (selected)**     | Native, zero-IPC, cross-platform, system tray support    |

### 7.2 `QThread` over `threading.Thread`

Qt's `QThread` integrates directly with the signal-slot system. Signals emitted from a `QThread`
are automatically delivered to the main thread via Qt's event queue, without any explicit locking
or `Queue`. `threading.Thread` would require a manual thread-safe queue and a polling timer.

### 7.3 Worker Thread owns `Runtime`, not `RuntimeController`

`RuntimeController` lives in the main thread. If it owned `Runtime`, calling `run()` would
block the event loop. Moving `Runtime` into `_RuntimeThread.run()` is the only correct placement.

### 7.4 Throttled State Emission

`Runtime.run()` runs at 30 Hz; emitting a Qt signal every frame would post 30 cross-thread
deliveries per second, each incurring event-queue overhead. The 3-frame throttle caps UI signal
traffic at ~10 Hz with negligible impact on perceived responsiveness.

### 7.5 No In-Session Config Mutation

The config form is locked (`set_runtime_active(True)`) while the runtime is active. YAML files
are written only after the runtime stops. This eliminates the risk of `build_container()` reading
a partially-written config and avoids the need for a live config hot-reload mechanism.

### 7.6 Minimize-to-Tray

`changeEvent` intercepts `WindowStateChange` when `isMinimized()` is true, hides the window
(`self.hide()`), and ignores the event. Left-clicking the tray icon calls `show()`, `raise_()`,
and `activateWindow()`. This keeps the runtime alive with no open windows, providing ambient
gesture feedback via the tray icon without requiring the user to keep the control panel in focus.

### 7.7 Atomic YAML Writes

The `ConfigPanel._safe_write()` method writes to a `.yaml.tmp` sibling file and atomically
renames it over the target. On POSIX systems, `Path.replace()` is guaranteed atomic at the
filesystem level. This prevents config file corruption if the process is killed mid-write.

---

## 8. Configuration and Deployment

### 8.1 Dependencies

PySide6 is declared in the `dev` optional extra of `pyproject.toml` (alongside `pytest-qt`).
To run the UI:

```bash
pip install -e ".[dev,training,huggingface]"
python scripts/run_ui.py
```

### 8.2 CI Guard

`tests/conftest.py` includes a collection guard that skips all `test_ui_*` files when PySide6
is not importable. This allows the standard CI pipeline (which runs without a display server) to
collect and execute all non-UI tests without modification.

### 8.3 Test Coverage

| Test file                       | Scope                                               | Count |
| ------------------------------- | --------------------------------------------------- | ----- |
| `test_ui_runtime_on_state.py`   | `on_state` callback integration in `Runtime`        | 5     |
| `test_ui_runtime_controller.py` | `RuntimeController` signal contract and lifecycle   | 6     |
| `test_ui_launcher_panel.py`     | `LauncherPanel` state machine and signals           | 8     |
| `test_ui_status_monitor.py`     | `StatusMonitor` field rendering from `RuntimeState` | 6     |
| `test_ui_config_panel.py`       | Form validation, apply-button gating, atomic write  | 6     |
| `test_ui_model_panel.py`        | HF / Local tab switching and path accessors         | 6     |
| `test_ui_tray.py`               | System tray creation and `show_window` checkbox     | 4     |

---

**Document**: UI View  
**Suite**: Architecture Documentation (`docs/architecture/`)  
**Last Updated**: April 2026
