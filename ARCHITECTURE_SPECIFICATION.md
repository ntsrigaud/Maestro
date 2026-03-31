# Production-Grade Architecture Specification

## MediaPipe Hands + LSTM Gesture Control for Presentations and Meetings

## 0. Executive Summary

This document defines a production-ready architecture for a webcam-based gesture control system that targets presentation and meeting workflows. The system uses MediaPipe Hands for landmark extraction and an LSTM classifier for temporal gesture recognition. The design is CPU-first, modular, testable, and intended for long-term maintainability.

Target command set:

- Next slide
- Previous slide
- Mute/unmute microphone
- Raise hand
- Start presentation
- Toggle presenter tools

Design goals:

- Reliable real-time interaction on consumer laptops
- Small and highly separable gesture vocabulary (6 to 8 gestures)
- Low false-trigger rate through explicit safety mechanisms
- Clear separation between training, inference, and application integration

---

## 1. Research-Driven Design Principles

### 1.1 Recurrent findings from recent MediaPipe + temporal gesture systems

Across 2019-2024 literature and production-like implementations, the most repeatable architecture pattern is:

1. Landmark extraction
2. Landmark normalization
3. Temporal windowing
4. Sequence model inference
5. Decision stabilization
6. Command gating and dispatch

Observed practical lessons:

- Landmark-based input is efficient and robust enough for small command vocabularies.
- Temporal models (LSTM/GRU/TCN) consistently improve dynamic gesture classification versus frame-only classifiers.
- Production reliability depends heavily on post-model logic (thresholding, temporal voting, activation state), not only model accuracy.
- No-gesture/unknown modeling is critical to reduce accidental activations.

### 1.2 Preprocessing patterns supported by research and deployments

Commonly effective techniques:

- Wrist-relative normalization for translation invariance
- Scale normalization with palm reference distance
- Optional orientation normalization when camera pose varies
- Temporal derivatives (velocity features) for dynamic gestures
- Basic temporal smoothing to handle noisy landmark jitter

### 1.3 Real-time design patterns validated in practice

- Sliding windows with overlap outperform disjoint windows for responsiveness.
- Ring buffers with bounded memory prevent latency drift.
- Batched offline evaluation must be separated from online inference mode.
- Trigger logic should require temporal consistency, not single-frame confidence spikes.

### 1.4 Architectural implications for this project

The architecture should enforce:

- Deterministic feature transformation
- Explicit inference state machine
- Config-driven thresholds and mappings
- Pluggable command adapters for meeting/presentation apps

---

## 2. Standard Project Directory Structure

Proposed repository tree:

```text
project_root/
  README.md
  pyproject.toml
  requirements/
    base.txt
    dev.txt
  configs/
    app.yaml
    inference.yaml
    logging.yaml
    model/
      feature_pipeline.yaml
      lstm_default.yaml
    gestures/
      vocabulary_ipn.yaml
      command_map_ipn.yaml
    thresholds/
      production_ipn.yaml
  src/
    gesture_control/
      __init__.py
      app/
        main.py
        bootstrap.py
        runtime.py
      domain/
        entities.py
        enums.py
        events.py
        contracts.py
      interfaces/
        frame_source.py
        landmark_extractor.py
        feature_transformer.py
        sequence_buffer.py
        gesture_model.py
        prediction_stabilizer.py
        activation_controller.py
        command_dispatcher.py
        feedback_renderer.py
      infrastructure/
        camera/
          opencv_source.py
        mediapipe/
          hands_extractor.py
        model/
          checkpoint_loader.py
        dispatch/
          keyboard_dispatcher.py   # legacy stub (superseded)
          command_dispatcher.py    # production dispatcher with allowlist + rate limiting
          app_adapters/
            base.py                # IApplicationAdapter Protocol
            keyboard_adapter.py   # low-level pyautogui key delivery with allowlist
            powerpoint_adapter.py  # PowerPoint shortcut table
            zoom_adapter.py        # Zoom shortcut table
        feedback/
          overlay_renderer.py
      pipeline/
        feature_engineering.py
        buffering.py
        stabilization.py
        activation.py
      training/
        augmentation.py
        dataset_builder.py
        datamodule.py
        train_lstm.py
        evaluate.py
        export_checkpoint.py
      evaluation/
        benchmark_latency.py
      utils/
        config_loader.py
        feature_validation.py
        time_utils.py
        schema_validation.py
        reproducibility.py
  data/
    raw/
    interim/
    processed/
    labels/
  models/
    checkpoints/
    exported/
    registry.json
  experiments/
    runs/
  scripts/
    collect_data.py
    build_dataset.py
    build_ipn_dataset.py
    train_model.py
    run_realtime.py
    benchmark_system.py
  tests/
    unit/
    integration/
    system/
    fixtures/
  docs/
    IPN_INTEGRATION_GUIDE.md
    architecture/
      system-overview.md
      logical-view.md
      development-view.md
      process-view.md
      physical-view.md
      data-flow.md
  tools/
    lint.sh
    test.sh
    profile.sh
```

### 2.1 Directory responsibilities

- configs: versioned runtime and model configuration; no hardcoded thresholds in code.
- src/gesture_control/app: entrypoints and dependency wiring.
- src/gesture_control/domain: pure business concepts; no framework code.
- src/gesture_control/interfaces: abstract contracts for module decoupling.
- src/gesture_control/infrastructure: concrete implementations (OpenCV, MediaPipe, Torch, OS dispatch).
- src/gesture_control/pipeline: orchestration logic for online inference path.
- src/gesture_control/training: data generation, training, validation, export.
- src/gesture_control/evaluation: reproducible benchmarks and acceptance metrics.
- data: immutable separation between raw and processed artifacts.
- models: checkpoints and model registry metadata.
- experiments: exploratory outputs separated from production code.
- scripts: operational CLIs for common workflows.
- tests: layered tests from unit to end-to-end.
- docs: architecture decision records, runbooks, integration docs.

---

## 3. Core System Architecture

### 3.1 End-to-end runtime flow

1. Video Input Module reads webcam frames with timestamps.
2. Hand Landmark Extraction returns 21 landmarks and confidence metrics.
3. Feature Engineering converts landmarks to normalized feature vectors.
4. Temporal Buffer Manager builds sliding windows.
5. LSTM Gesture Classification produces per-class probabilities.
6. Prediction Stabilization applies smoothing, voting, and rejection logic.
7. Gesture Activation Controller verifies state, cooldown, and idle behavior.
8. Command Dispatcher triggers OS/app actions.
9. Real-Time Feedback overlays state, prediction, and confidence.

### 3.2 Module specifications

#### A. Video Input Module

Responsibilities:

- Capture frames from webcam
- Enforce frame-rate target (for example 20 to 30 FPS)
- Optional resize/color conversion for downstream stability

Inputs:

- Camera index, resolution, target FPS

Outputs:

- FramePacket(frame, timestamp, frame_id)

Failure handling:

- Camera reconnect retry
- Graceful degraded state when capture fails

#### B. Hand Landmark Extraction

Responsibilities:

- Run MediaPipe Hands
- Select active hand (single-user assumption)
- Validate detection and tracking confidence

Inputs:

- FramePacket

Outputs:

- LandmarkPacket(21x3 landmarks, handedness, detect_conf, track_conf)

Failure handling:

- Emit no-landmark event without crashing pipeline

#### C. Feature Engineering Layer

Responsibilities:

- Wrist-relative normalization
- Scale normalization
- Optional orientation canonicalization
- Compute velocity features and frame validity flags

Inputs:

- LandmarkPacket plus previous frame feature context

Outputs:

- FeatureFrame(vector, valid_flag, timestamp)

#### D. Temporal Buffer Manager

Responsibilities:

- Maintain ring buffer of FeatureFrame
- Build fixed-length windows with overlap
- Gate inference when valid ratio is below threshold

Inputs:

- FeatureFrame stream

Outputs:

- SequenceWindow(T x F features, metadata)

Recommended defaults:

- Window length: 24 frames
- Step size: 3 frames
- Minimum valid ratio: 0.8

#### E. Gesture Classification Model (LSTM)

Responsibilities:

- Load versioned checkpoint and metadata
- Run low-latency sequence inference
- Return calibrated class probabilities

Inputs:

- SequenceWindow

Outputs:

- Prediction(class_probs, top_class, confidence)

Model baseline:

- 2-layer LSTM, hidden size 128, dropout 0.3
- Class set includes unknown/no_gesture

#### F. Prediction Stabilization Layer

Responsibilities:

- Apply confidence thresholds per class
- Temporal majority vote on recent predictions
- Confidence EMA smoothing
- Unknown/reject handling

Inputs:

- Raw Prediction stream

Outputs:

- StabilizedPrediction(label, confidence, accepted)

#### G. Gesture Activation Controller

Responsibilities:

- Manage state machine: IDLE, ARMED, COOLDOWN
- Activation gesture hold detection
- Idle timeout and command cooldown

Inputs:

- StabilizedPrediction and timestamps

Outputs:

- ExecutableCommand or None

#### H. Command Dispatch Layer

Responsibilities:

- Enforce a **command allowlist**: only command IDs explicitly listed in the
  allowlist are forwarded to adapters; unknown IDs are rejected before any
  OS interaction occurs.
- Apply **adapter-level rate limiting**: a sliding-window counter caps the
  number of dispatches within a configurable time window, providing a second
  layer of protection independent of the activation controller cooldown.
- Delegate actual key delivery to an **application adapter** selected at
  startup based on `active_app` config (powerpoint, zoom, …).
- Return a structured `DispatchResult` for every call — the runtime loop
  is never interrupted by adapter failure.

**Class hierarchy**:

```
ICommandDispatcher         (interfaces/command_dispatcher.py)
    └── CommandDispatcher  (infrastructure/dispatch/command_dispatcher.py)
            └── IApplicationAdapter  (app_adapters/base.py  — Protocol)
                    ├── PowerPointAdapter  (uses KeyboardAdapter)
                    └── ZoomAdapter        (uses KeyboardAdapter)
                            └── KeyboardAdapter  (wraps pyautogui with _SAFE_KEY_NAMES allowlist)
```

**Safety design — KeyboardAdapter**:

- `_SAFE_KEY_NAMES` frozenset validates every key string before `pyautogui.hotkey` is called.
- No user-supplied strings or command IDs ever reach `pyautogui` directly — the adapter
  layer resolves command IDs to predefined key tuples first.
- Raising `ValueError` for any unlisted key name prevents injection of arbitrary OS commands.
- `pyautogui` is imported at module level with a `None` fallback, so CI / headless
  environments can run without side-effects and the module remains patchable.

**Gesture-to-command mapping**:

- Loaded from `configs/gestures/command_map_ipn.yaml` at startup by `bootstrap.py`.
- Passed to `GestureActivationController` (gesture → command_id resolution) and
  used to build the `CommandDispatcher` allowlist (frozenset of command_id values).
- Fallback to hardcoded defaults if the YAML is absent.

Inputs:

- ExecutableCommand

Outputs:

- DispatchResult(success, command_id, target_app, detail)

#### I. Real-Time Feedback System

Responsibilities:

- Overlay active state and recognized gesture
- Show confidence and rejection reason
- Emit structured telemetry for debugging

Inputs:

- Runtime events from pipeline

Outputs:

- Rendered frame and log records

---

## 4. Software Architecture Principles

### 4.1 SOLID mapping

- Single Responsibility: each module owns one concern (capture, extraction, feature transform, inference, dispatch).
- Open-Closed: new gestures or dispatch targets are added via config or adapters without modifying core pipeline.
- Liskov Substitution: alternative implementations (for example mock extractor) satisfy the same interface contracts.
- Interface Segregation: small focused interfaces prevent monolithic dependencies.
- Dependency Inversion: orchestration depends on abstractions in interfaces, not concrete infrastructure classes.

### 4.2 DRY and YAGNI application

- Shared feature normalization utility used by both training and inference.
- One config schema for thresholds and vocabulary used everywhere.
- Avoid multi-model routing in v1; ship one LSTM path first, keep extension points ready.

### 4.3 Separation of training and inference

- Training pipeline lives under training module and scripts.
- Inference runtime is isolated in pipeline plus infrastructure.
- Shared code is limited to deterministic feature transformations and schema definitions.

### 4.4 Interface-based service architecture

Core contracts:

- IFrameSource
- IHandLandmarkExtractor
- IFeatureTransformer
- ISequenceBuffer
- IGestureModel
- IPredictionStabilizer
- IActivationController
- ICommandDispatcher
- IFeedbackRenderer

Dependency wiring is centralized in bootstrap composition root.

---

## 5. Gesture Vocabulary Design (IPN Dataset — 8 Gestures)

### 5.1 Production vocabulary (8 gestures, IPN dataset)

1. open_palm_hold (IPN: G07 — "Open twice")

- Command: arm/disarm interaction (activation gesture)
- Note: `open_palm_hold` is the production vocabulary label for IPN G07
  ("Open twice"). During dataset assembly, G07 frames (the double open-hand
  motion) are labeled `open_palm_hold` because that motion most closely
  approximates the open-palm activation cue in a presentation context. The
  trained model (`lstm_v1_20260330T181558Z`) was built with this label;
  **no retraining is required**. The mapping is marked "Approximate" in
  `docs/IPN_INTEGRATION_GUIDE.md` § 2.1.

2. swipe_right (IPN: G06)

- Command: next_slide

3. swipe_left (IPN: G05)

- Command: previous_slide

4. swipe_up (IPN: G03)

- Command: start_presentation

5. swipe_down (IPN: G04)

- Command: exit_presentation

6. zoom_in (IPN: G10)

- Command: toggle_pointer_mode

7. zoom_out (IPN: G11)

- Command: raise_hand

8. unknown (IPN: D0X + B0A + B0B)

- Command: none (no-gesture / noise class)

### 5.2 Vocabulary design rules

- Prefer high visual distance between classes.
- Mix static and dynamic gestures carefully to reduce confusion.
- Reserve one activation gesture not used for command dispatch.
- Keep gesture execution duration below 700 ms for comfort.

---

## 6. Reliability and Safety Mechanisms

### 6.1 Required protections

- Activation gating: commands only in ARMED state.
- Unknown class: no forced classification.
- Temporal consistency: require stable class across consecutive windows.
- **Edge-trigger dispatch**: a command is emitted only when the predicted
  label _changes_ from the previous label (rising edge). Subsequent frames
  that carry the same accepted label are ignored, regardless of whether the
  cooldown has expired. The trigger resets as soon as the prediction returns
  to any different label (typically `unknown` between gestures).
- Cooldown timer: secondary guard that suppresses re-dispatch within a fixed
  window after each command (backs up edge-trigger in unusual conditions).
- Idle timeout: auto-return to IDLE.
- Confirmation rule for critical actions.

### 6.2 Suggested thresholds (initial production defaults)

- Activation hold: 600 ms
- Idle timeout: 300 000 ms (5 min) — configured via `idle_timeout_ms` in `configs/inference.yaml`
- Base acceptance threshold: 0.85
- Critical commands threshold: 0.90
- Stability rule: same class accepted in 3 consecutive decisions
- Command cooldown: 1 000 ms (secondary guard; primary protection is edge-trigger)
- Edge trigger: enabled by default (`edge_trigger_enabled: true` in `configs/inference.yaml`)

### 6.4 Single-dispatch guarantee (edge-trigger detail)

With `window_length=24`, `window_step=3`, and a 30 FPS stream, inference
windows arrive every ~100 ms. A single swipe lasting 0.8 s produces ~8
consecutive accepted predictions. Prior to the edge-trigger fix, every
time the 1 000 ms cooldown expired while the gesture was still active,
another command was dispatched.

The edge-trigger guard in `GestureActivationController.route()` resolves
this by tracking `_last_dispatched_label` — the label of the most recently
executed command. A command is dispatched only when:

```
prediction.label != _last_dispatched_label   ← rising-edge condition
AND prediction.accepted == True
AND state == ARMED
AND cooldown expired
```

`_last_dispatched_label` is updated **only when a command is actually
dispatched**, never on unaccepted frames. This is critical because the
temporal stabiliser (vote*window=3) emits 2–3 consecutive `accepted=False`
frames before the first dispatchable `accepted=True` frame. An earlier
implementation tracked `_prev_label` on \_every* call, which caused those
unaccepted buildup frames to consume the rising edge — silently dropping all
gestures after the first.

`_last_dispatched_label` is reset to `unknown` when:

- the prediction returns to `unknown` (hand goes neutral), or
- the controller re-enters the ARMED state from IDLE.

A new gesture fires exactly once — on its first `accepted=True` frame — after
either a different label was last dispatched, or the hand went neutral since
the last dispatch.

### 6.3 False-trigger control strategy

- Train with diverse no_gesture negatives.
- Explicitly log all rejected predictions with confidence.
- Tune thresholds from false-trigger benchmark, not only validation accuracy.

---

## 7. Implementation Roadmap (Multi-Phase)

### Phase 1: Project scaffolding and repository setup

Goal:

- Establish clean structure, tooling, and contracts.

Deliverables:

- Directory skeleton, interfaces, config schema, CI skeleton.

Validation:

- Lint/test pipeline green; import graph and style checks pass.

### Phase 2: MediaPipe integration and landmark extraction

Goal:

- Stable landmark stream from webcam.

Deliverables:

- Frame source, MediaPipe extractor, confidence-filtered output.

Validation:

- Frame throughput report and extraction reliability test.

### Phase 3: Feature engineering and dataset generation

Goal:

- Deterministic features for training and inference parity.

Deliverables:

- Normalization module, sequence builder, dataset CLI.

Validation:

- Unit tests for feature invariance and reproducibility snapshot tests.

### Phase 4: LSTM training pipeline

Goal:

- Train baseline LSTM and export reproducible checkpoint.

Deliverables:

- Training script, evaluation report, model registry metadata.

Validation:

- Accuracy, macro-F1, calibration, and confusion matrix thresholds.

### Phase 5: Real-time inference pipeline

Goal:

- End-to-end online prediction with bounded latency.

Deliverables:

- Sliding window inference, prediction stream, overlay diagnostics.

Validation:

- Latency benchmark under target hardware profile.

### Phase 6: Activation and command dispatch

Goal:

- Safe command execution integration.

Deliverables:

- Activation state machine, command mapper, adapter dispatchers.

Validation:

- Integration tests for gesture-to-command mapping.

### Phase 7: Reliability and safety hardening

Goal:

- Minimize accidental triggers.

Deliverables:

- Unknown handling, temporal voting, cooldown, rejection telemetry.

Validation:

- False-trigger benchmark on non-gesture sessions.

### Phase 8: Feedback UI and operational visibility

Goal:

- Improve operator trust and debuggability.

Deliverables:

- Overlay renderer, confidence indicators, structured runtime logs.

Validation:

- Usability walkthrough and operator error analysis.

### Phase 9: Testing, benchmarking, optimization

Goal:

- Production readiness certification.

Deliverables:

- Unit/integration/system test suites, performance profile, tuning report.

Validation:

- Meet release gates for latency, accuracy, and false-trigger metrics.

---

## 8. Testing Strategy

### 8.1 Unit tests

Focus:

- Landmark normalization
- Feature extraction
- Velocity and temporal derivatives
- Ring buffer behavior
- Threshold and voting logic

Tools:

- pytest
- hypothesis (property-based tests for invariance)
- pytest-cov

### 8.2 Integration tests

Focus:

- Frame source to extractor to feature pipeline
- Sequence inference to stabilization flow
- Activation controller plus dispatcher behavior

Approach:

- Use recorded landmark fixtures for deterministic playback
- Use fake adapters for external app integration

### 8.3 System tests

Focus:

- End-to-end gesture recognition loop
- Long-run stability and memory usage
- False-trigger behavior during idle/non-gesture sessions

Acceptance gates:

- End-to-end latency less than 200 ms
- In-vocabulary accuracy above 95% in controlled indoor setup
- False triggers less than 1 per 10 minutes non-gesture runtime

---

## 9. Configuration and Model Management

### 9.1 Configuration approach

- Use YAML for human-readable configs.
- Validate all runtime config using typed schema (for example pydantic models).
- Separate files by concern: model, vocabulary, thresholds, logging, app mappings.

### 9.2 Model checkpoint management

- Store checkpoints in models/checkpoints with immutable version IDs.
- Maintain models/registry.json with:
  - model_version
  - training_data_version
  - feature_schema_version
  - class_labels
  - window_length
  - checksum

### 9.3 Gesture vocabulary and threshold definitions

- Keep vocabulary in configs/gestures/vocabulary_ipn.yaml.
- Keep class-specific thresholds in configs/thresholds/production_ipn.yaml.
- Keep command mappings in configs/gestures/command_map_ipn.yaml by target integration adapter.

### 9.4 Reproducibility controls

- Fixed random seeds for training and evaluation.
- Dataset manifest files with hash and sampling metadata.
- Experiment artifacts stored in experiments/runs with config snapshot.

---

## 10. Interface Definitions (Implementation Guide)

```python
class IFrameSource(Protocol):
    def read(self) -> FramePacket: ...

class IHandLandmarkExtractor(Protocol):
    def extract(self, frame: FramePacket) -> LandmarkPacket: ...

class IFeatureTransformer(Protocol):
    def transform(self, packet: LandmarkPacket) -> FeatureFrame: ...

class ISequenceBuffer(Protocol):
    def push(self, feature: FeatureFrame) -> None: ...
    def ready(self) -> bool: ...
    def window(self) -> SequenceWindow: ...

class IGestureModel(Protocol):
    def predict(self, seq: SequenceWindow) -> Prediction: ...

class IPredictionStabilizer(Protocol):
    def update(self, pred: Prediction) -> StabilizedPrediction: ...

class IActivationController(Protocol):
    def route(self, pred: StabilizedPrediction) -> Optional[ExecutableCommand]: ...

class ICommandDispatcher(Protocol):
    def dispatch(self, cmd: ExecutableCommand) -> DispatchResult: ...

class IFeedbackRenderer(Protocol):
    def render(self, frame: FramePacket, state: RuntimeState) -> FramePacket: ...
```

Notes:

- Keep protocol contracts minimal and stable.
- Avoid leaking infrastructure dependencies into domain interfaces.

---

## 11. Production Release Criteria

Release only when all are true:

- Latency target met on defined baseline laptop profile
- Accuracy and false-trigger targets met in repeated sessions
- No critical failures in 60-minute endurance test
- Config validation and model integrity checks pass
- Integration adapters pass automated smoke tests

---

## 12. Final Recommendation

Implement this system as a single, well-instrumented LSTM production path first. Prioritize reliability logic, deterministic preprocessing, and command safety over adding more models or gestures. This yields faster time-to-value, better maintainability, and a clearer route to dependable real-world usage.
