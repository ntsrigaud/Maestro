# Process View

## 1. Overview

The process view describes the runtime behavior of the system, including the real-time frame processing pipeline, concurrency model, state management, and timing characteristics. The system uses a **dual-thread execution model** launched via `scripts/run_ui.py`.

## 2. Threading Model

The system uses a **dual-thread execution model**:

| Thread            | Runs                                         | Responsibility                            |
| ----------------- | -------------------------------------------- | ----------------------------------------- |
| **Main thread**   | `QApplication.exec()` (Qt event loop)        | UI rendering, signal dispatch, user input |
| **Worker thread** | `_RuntimeThread` (QThread) → `Runtime.run()` | Full inference pipeline, unmodified       |

Cross-thread communication uses Qt **queued signal connections** — the safe mechanism for passing data across thread boundaries in Qt. `_RuntimeThread._handle_state()` is called by `Runtime` every frame via the `on_state` callback. The worker increments a frame counter and emits the `state_ready` Qt signal every `emit_every` frames (default 3), capping UI traffic at approximately 10 Hz at 30 FPS. The queued connection delivers each `RuntimeState` (`dataclass(slots=True)`) to the main thread via Qt's event queue. No mutex, explicit lock, or shared mutable state is used.

The inference pipeline is entirely decoupled from the UI thread; `on_state` is the only addition to `runtime.py`.

## 3. Main Event Loop

The runtime event loop processes one frame per iteration:

```mermaid
graph TD
    Start([Loop Start]) --> Capture["1. Frame Capture<br/>OpenCVFrameSource.read()"]
    Capture --> Gaze["2. Gaze Estimation<br/>HeadPoseGazeDetector.update()"]
    Gaze --> GazeOk{Gaze Permitted?}
    GazeOk -->|No| Render
    GazeOk -->|Yes| Extract["3. Landmark Extraction<br/>MediaPipeHandsExtractor.extract()"]
    Extract --> Select["4. Hand Selection<br/>HandSelector.select()"]
    Select --> Transform["5. Feature Transform<br/>LandmarkFeatureTransformer.transform()"]
    Transform --> Buffer["6. Buffer Push<br/>SlidingWindowSequenceBuffer.push()"]
    Buffer --> Ready{Buffer Ready?}
    Ready -->|No| Render
    Ready -->|Yes| Infer["7. Enhanced Two-Stream Inference<br/>EnhancedTwoStreamLSTM.predict()"]
    Infer --> Stabilize["8. Stabilize<br/>PredictionStabilizer.update()"]
    Stabilize --> Activate["9. Activation Routing<br/>GestureActivationController.route()"]
    Activate --> HasCmd{Command?}
    HasCmd -->|Yes| Dispatch["10. Dispatch<br/>CommandDispatcher.dispatch()"]
    HasCmd -->|No| Continuous
    Dispatch --> Continuous["11. Continuous Trackers<br/>LaserPointer/AnnotationPen"]
    Continuous --> Render["12. Render Overlay<br/>OverlayRenderer.render()"]
    Render --> Callback["on_state callback<br/>(if registered; throttled ~10 Hz)"]
    Callback --> Start
```

### Iteration Timing

At 30 FPS target, each loop iteration has a ~33 ms budget. The pipeline stages consume approximately:

| Stage               | Typical Latency | Notes                                                |
| ------------------- | --------------- | ---------------------------------------------------- |
| Frame Capture       | 1–2 ms          | Blocks until frame available; buffer size = 1        |
| Gaze Estimation     | 1–2 ms          | MediaPipe BlazeFace amortized; runs at ⅓ frame rate  |
| Landmark Extraction | 15–25 ms        | MediaPipe CPU inference; skipped if gaze absent      |
| Hand Selection      | < 0.1 ms        | In-memory preference filtering over 0–2 hand entries |
| Feature Transform   | < 1 ms          | NumPy array operations                               |
| Buffer Push         | < 0.1 ms        | Deque append                                         |
| Enhanced Inference  | 20–30 ms        | PyTorch CPU forward pass (when buffer ready)         |
| Stabilization       | < 0.1 ms        | EMA + deque lookup                                   |
| Activation Routing  | < 0.1 ms        | State machine transition                             |
| Command Dispatch    | < 1 ms          | pyautogui call (when command emitted)                |
| Continuous Trackers | < 1 ms          | Pointer/pen update, mouse drag while gesture active  |
| Overlay Render      | 1–2 ms          | OpenCV text/rectangle drawing                        |

**Total end-to-end (gaze active)**: ~40–60 ms per frame (without model inference), ~60–80 ms when inference triggers. Gaze estimation adds ~1–2 ms amortized per frame because BlazeFace is sampled at 10 Hz (every 3rd frame) and the result is cached. When gaze is absent, the per-frame cost drops to ~3–5 ms (capture + gaze check + render).

## 4. Gaze-Aware Activation Design

### 4.1 Gaze Intent Detector State Machine

`HeadPoseGazeDetector` maintains an independent four-state machine that acts as the outermost pipeline gate:

```mermaid
stateDiagram-v2
    [*] --> ABSENT

    ABSENT --> ACQUIRING : Head pose within threshold
    ACQUIRING --> ACTIVE : Stable gaze for arm_ms (default 500 ms)
    ACQUIRING --> ABSENT : Gaze broken before threshold
    ACTIVE --> LOSING : Head pose leaves threshold
    LOSING --> ACTIVE : Gaze recovered within grace period
    LOSING --> ABSENT : Grace period elapsed (grace_ms, default 5 s)
```

**ABSENT**: No face detected or head pose exceeds angular thresholds. All hand landmark extraction, LSTM inference, and activation routing are bypassed. Any ongoing gesture activation hold timer is reset.

**ACQUIRING**: A qualifying head pose has been detected. An internal stability timer starts. The gesture pipeline remains inactive until the state advances to ACTIVE, preventing instantaneous arming when the user glances briefly toward the camera.

**ACTIVE**: Head pose has been within thresholds continuously for `arm_ms` (default 500 ms). The downstream gesture pipeline is fully operational.

**LOSING**: Head pose has left the acceptance window. A grace timer starts. If the user looks back before the grace period elapses (`grace_ms`, default 5000 ms), no state change in the gesture controller occurs. This prevents disarming during natural head movements mid-presentation.

### 4.2 Gesture Activation State Machine

The `GestureActivationController` implements a two-state finite state machine that governs command emission:

```mermaid
stateDiagram-v2
    [*] --> ARMED

    ARMED --> COOLDOWN : Command dispatched (edge trigger)
    COOLDOWN --> ARMED : Cooldown timer expires (1500 ms)
```

### State Descriptions

**ARMED**:

- The system accepts command gestures and dispatches them to the target application.
- The controller **starts in ARMED state** at session startup with no external trigger required.
- **Edge-trigger logic**: A command is emitted only when:
  1. The stabilized prediction is `accepted` (temporal voting passed).
  2. The label differs from the last dispatched label (rising-edge detection).
  3. The label exists in the command map.
- The `_last_dispatched_label` resets to `"unknown"` whenever the prediction returns to the unknown class (hand in neutral position), enabling the same gesture to be dispatched again after a neutral interval.

**COOLDOWN**:

- Entered immediately after a command is dispatched.
- All dispatch is blocked for `command_cooldown_ms` (default 1500 ms in current config).
- Provides a secondary guard against rapid re-dispatch (the primary guard is edge-triggering).
- After the cooldown timer expires, the system returns to ARMED.

## 5. Prediction Stabilization Pipeline

The prediction stabilizer applies three sequential filtering stages to reduce noise in raw model output:

```mermaid
graph LR
    Raw["Raw Prediction<br/>(class probabilities)"] --> EMA["Stage 1: EMA Smoothing<br/>α = 0.4"]
    EMA --> Threshold["Stage 2: Per-Class<br/>Confidence Threshold"]
    Threshold --> Vote["Stage 3: Temporal<br/>Majority Voting"]
    Vote --> Output["StabilizedPrediction<br/>(label, accepted, reason)"]
```

**Stage 1 — EMA Smoothing**: Each class probability is smoothed over time using exponential moving average: `p'[c] = α · p_current[c] + (1 - α) · p_previous[c]`. This dampens frame-to-frame fluctuations.

**Stage 2 — Confidence Thresholding**: The smoothed top-class probability is compared against the class-specific threshold from `production_hybrid.yaml`. If below threshold, the candidate is demoted to `no_gesture` with reason `"below_threshold_or_no_gesture"`.

**Stage 3 — Temporal Majority Voting**: The last N predictions (where N = `class_vote_windows[label]`) must all agree on the same class. Dynamic gestures (swiping_right, swiping_left) use N=1 for immediate response. Static holds (stop_sign) use N=3 for stability. Disagreement results in rejection with reason `"temporal_inconsistency"`.

## 6. Command Dispatch Pipeline

When the activation controller emits an `ExecutableCommand`, it passes through three validation stages:

```mermaid
graph TD
    Cmd["ExecutableCommand"] --> Allow{Allowlist Check}
    Allow -->|Rejected| Fail1["DispatchResult<br/>(rejected_by_allowlist)"]
    Allow -->|Passed| Rate{Rate Limit Check}
    Rate -->|Exceeded| Fail2["DispatchResult<br/>(rate_limited)"]
    Rate -->|Passed| Adapter["PresentationAdapter"]
    Adapter --> KeyValidate{Key Name Validation}
    KeyValidate -->|Invalid| Fail3["DispatchResult<br/>(invalid_key)"]
    KeyValidate -->|Valid| Execute["KeyboardAdapter action\n(hotkey/zoom/mouse)"]
    Execute --> Success["DispatchResult<br/>(success=True)"]
```

**Allowlist check**: The command ID must exist in the `allowed_commands` frozenset configured at startup.

**Rate limit check**: A sliding-window deque tracks recent dispatch timestamps. If `rate_limit_per_window` commands have occurred within `rate_limit_window_ms`, the command is rejected.

**Key name validation**: The `KeyboardAdapter` validates every key in the shortcut tuple against `_SAFE_KEY_NAMES`, a hardcoded frozenset of known-safe pyautogui key names. No user-supplied strings reach the OS keyboard layer.

## 7. Training Pipeline Process

The offline training pipeline is a batch process separate from the real-time inference loop:

```mermaid
graph TD
    JSONL["JSONL Landmark Files"] --> Load["Load Labeled Frames"]
    Load --> FeatTrain["Feature Transform<br/>(same as inference)"]
    FeatTrain --> Window["Sliding Window<br/>(same as inference)"]
    Window --> NPZ["NPZ Dataset<br/>(N × T × 147)"]
    NPZ --> Split["Recording-level Group Split<br/>(70/15/15)"]
    Split --> Augment["On-the-Fly Augmentation<br/>(temporal scale, spatial jitter)"]
    Augment --> Train["EnhancedTwoStreamLSTM Training Loop<br/>(AdamW, class-weighted CE)"]
    Train --> Eval["Evaluation<br/>(precision, recall, F1)"]
    Eval --> Export["Export Artifact<br/>(.pt + metadata)"]
    Export --> HFHub["Publish to HF Hub<br/>(ntsrigaud/maestro-lstm-hybrid)"]
```

The training pipeline reuses `LandmarkFeatureTransformer` and `SlidingWindowSequenceBuffer` from the inference pipeline to guarantee feature parity. The `GestureDataModule` manages dataset loading, group-aware recording-level splits, and on-the-fly augmentation for the training partition. Trained artifacts are published to the Hugging Face Hub (`ntsrigaud/maestro-lstm-hybrid`) via `training/hf_publish.py` and are not tracked in VCS. At runtime, the artifact is downloaded on first use via `hf_hub_download()` and served from the local HF cache.

## 8. Graceful Shutdown

The runtime loop supports three shutdown triggers:

1. **Keyboard interrupt**: Interrupting the process from terminal (headless CLI path).
2. **Frame limit**: Reaching `--max-frames` count (used in testing and benchmarking).
3. **UI Stop button**: Clicking Stop in the PySide6 control panel calls `RuntimeController.stop()`, which sets the runtime stop flag. `MainWindow` then waits for worker completion during shutdown.

On shutdown, the runtime:

1. Releases the camera resource (`IFrameSource.close()`).
2. Closes the MediaPipe model (`IHandLandmarkExtractor.close()`).
3. Deactivates continuous trackers (laser pointer / annotation pen) to restore safe input state.
4. Optionally writes a latency benchmark report to the specified JSON path.

---

**Document**: Process View  
**Suite**: Architecture Documentation (`docs/architecture/`)  
**Last Updated**: May 2026
