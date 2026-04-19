# Work Log — Full Project: Presentation and Meeting Control System

**Date:** 2026-04-18  
**Branch:** `develop`  
**Authors:** @ntsrigaud · @blackbird410

---

## Context

This branch represents the complete build of the Gesture-Controlled Presentation and Meeting
Control system on top of the repository that previously contained a ViT/LSTM-based semantic
gesture prototype. The `develop` branch replaces that earlier direction with a purpose-built,
production-oriented system: a real-time webcam gesture pipeline that drives presentation and
meeting software through gaze-aware activation and secure keyboard dispatch.

The work proceeded in six phases over approximately five weeks (2026-03-16 → 2026-04-18).

**Total scope:** 404 files changed, 27 978 insertions(+), 26 486 deletions(-)  
**Total PRs:** #60 – #145 (86 PRs)

---

## Legacy Repository Cleanup

Before new development began, a large volume of deprecated material from the previous
prototype direction was removed:

| Category                                                                                  | Removed                       |
| ----------------------------------------------------------------------------------------- | ----------------------------- |
| Ad-hoc guide documents (`DEVELOPMENT_GUIDE.md`, `INSTALL.md`, `QUICK_REFERENCE.md`, etc.) | ~12 root-level markdown files |
| `docs/Project-Architecture/` subtree (old CLIP/ViT architecture docs)                     | ~10 files                     |
| `docs/Research-Notes/`, `docs/Topics/`, `docs/Imaging/` (research artifacts)              | ~15 files + images            |
| `logs/` directory (legacy work logs)                                                      | 6 files                       |
| `notebooks/` training artifacts and legacy notebooks                                      | ~25 files / images            |
| `scripts/` legacy data collection, HaGRID download, and demo scripts                      | ~30 scripts                   |
| `src/input_manager/`, `src/recognition_engine/`, `src/shared/`                            | Full legacy packages          |
| `environment.yml`, `setup.py`                                                             | Replaced by `pyproject.toml`  |

---

## Phase 1 — Project Foundation (PRs #73–#92, #85–#90; March 2026)

### Package Architecture

The repository was reorganized into a production-oriented package centered on
`src/gesture_control/` with strict layer boundaries:

| Layer          | Path                                  |
| -------------- | ------------------------------------- |
| Domain         | `src/gesture_control/domain/`         |
| Interfaces     | `src/gesture_control/interfaces/`     |
| Infrastructure | `src/gesture_control/infrastructure/` |
| Pipeline       | `src/gesture_control/pipeline/`       |
| Application    | `src/gesture_control/app/`            |
| Training       | `src/gesture_control/training/`       |
| Evaluation     | `src/gesture_control/evaluation/`     |
| Utilities      | `src/gesture_control/utils/`          |

**Domain layer** (`domain/entities.py`, `contracts.py`, `enums.py`, `events.py`):
Dataclass entities for `FrameData`, `LandmarkFrame`, `FeatureFrame`, `GesturePrediction`,
`RuntimeState`, and `CommandID`; lifecycle protocols `IRunnable` / `IClosable`.

**Interfaces layer** — protocol contracts for all nine pipeline components:
`IFrameSource`, `ILandmarkExtractor`, `IFeatureTransformer`, `ISequenceBuffer`,
`IGestureModel`, `IPredictionStabilizer`, `IActivationController`, `ICommandDispatcher`,
`IFeedbackRenderer`, and later `IGazeIntentDetector`.

**Utilities layer**:

- `config_loader.py` — YAML config loading with environment variable resolution
- `schema_validation.py` — Pydantic models for all config schemas
- `reproducibility.py` — deterministic seeding for training runs
- `time_utils.py` — epoch-millisecond timestamp helper
- `feature_validation.py` — runtime compatibility validation (schema, dim, normalization)

### Configuration Baseline

| File                                     | Content                                                         |
| ---------------------------------------- | --------------------------------------------------------------- |
| `configs/app.yaml`                       | Application runtime settings                                    |
| `configs/inference.yaml`                 | Temporal buffer parameters, stabilization, activation, dispatch |
| `configs/logging.yaml`                   | Structured logging configuration                                |
| `configs/model/lstm_default.yaml`        | LSTM architecture and training hyperparameters                  |
| `configs/model/feature_pipeline.yaml`    | Feature schema version, block dims, normalization               |
| `configs/gestures/vocabulary_ipn.yaml`   | IPN-derived gesture class definitions                           |
| `configs/gestures/command_map_ipn.yaml`  | Gesture → command ID mapping                                    |
| `configs/thresholds/production_ipn.yaml` | Per-class confidence thresholds                                 |

### Build System and CI

`pyproject.toml` — `setuptools` build, Python 3.11+, core runtime dependencies
(`pydantic`, `PyYAML`, `numpy`, `mediapipe`, `opencv-python`, `torch`).

`.github/workflows/ci.yml` — GitHub Actions: lint (`ruff`), type check (`mypy`), and test
across `runtime` and `training` job tracks.

`tools/lint.sh`, `tools/test.sh`, `tools/profile.sh` — local developer automation.

`requirements/base.txt` + `requirements/dev.txt` — pinned dependency sets.

### Architecture Documentation

Six architecture views added under `docs/architecture/`:

| Document              | Content                                                             |
| --------------------- | ------------------------------------------------------------------- |
| `system-overview.md`  | Technology stack, deployment topology, key design decisions         |
| `logical-view.md`     | Package decomposition and dependency directions                     |
| `data-flow.md`        | Per-stage data transformation, tensor shapes, feature vector layout |
| `process-view.md`     | Runtime event loop, thread model, state machine                     |
| `development-view.md` | Build system, dependency management, CI/CD                          |
| `physical-view.md`    | Deployment environments, hardware requirements                      |

---

## Phase 2 — Core Pipeline and Training Foundation (PRs #90–#103; March 2026)

### Frame Capture and Landmark Extraction

`src/gesture_control/infrastructure/camera/opencv_source.py` — `OpenCVFrameSource`:
OpenCV `VideoCapture` wrapper implementing `IFrameSource`.

`src/gesture_control/infrastructure/mediapipe/hands_extractor.py` —
`MediaPipeHandsExtractor`: calls MediaPipe Hands and normalises detected landmark
coordinates relative to the wrist; returns `LandmarkFrame`.

### Feature Engineering and Temporal Buffering

`src/gesture_control/pipeline/feature_engineering.py` — `LandmarkFeatureTransformer`:
constructs the feature vector from a `LandmarkFrame`. Initial implementation:
63 wrist-relative, scale-normalised positions + 63 velocity vectors = 126-dim
(`feature-schema-v1`).

`src/gesture_control/pipeline/buffering.py` — `SlidingWindowSequenceBuffer`:
FIFO ring buffer of the last `window_length` feature frames; emits a `(T, D)` tensor
when the fill-ratio threshold is met.

`src/gesture_control/utils/feature_validation.py` — `validate_runtime_compatibility()`:
verifies feature schema version, dimension count, and normalisation flags match the
loaded model artifact before the runtime loop starts.

### LSTM Model and Checkpoint Loader

`src/gesture_control/infrastructure/model/lstm_model.py` — `GestureLSTMClassifier`:
bidirectional LSTM with temporal attention and a linear classification head.

`src/gesture_control/infrastructure/model/checkpoint_loader.py` —
`load_inference_artifact()`: loads a `torch.jit.save`-compatible checkpoint; reads
`model_type`, `input_size`, `feature_schema_version` from embedded metadata.
`load_latest_inference_artifact()`: resolves the highest-priority artifact from
`models/registry.json`.

### Real-Time Inference Pipeline

`src/gesture_control/pipeline/stabilization.py` — `GesturePredictionStabilizer`:
temporal voting over a prediction window; returns `GesturePrediction` with
`accepted: bool`.

`src/gesture_control/pipeline/activation.py` — `GestureActivationController`:
IDLE → ARMED → COOLDOWN state machine; edge-triggered dispatch with configurable
`command_cooldown_ms`.

`src/gesture_control/app/runtime.py` — `GestureControlRuntime`: orchestrates the
full inference loop — frame acquisition, landmark extraction, feature engineering,
buffering, LSTM inference, stabilisation, activation, dispatch, and overlay rendering.

`src/gesture_control/app/bootstrap.py` — `build_container()`: wires all infrastructure
implementations to their interface contracts; loads config; returns a fully-assembled
`AppContainer`.

`src/gesture_control/app/main.py` — CLI entry point; parses `--config`, `--registry`,
`--model-artifact` flags.

`scripts/run_realtime.py` — forwards to `main.py`.

### LSTM Training Pipeline

`src/gesture_control/training/dataset_builder.py` — builds NPZ datasets from labeled
JSONL landmark files; sliding-window extraction; feature schema version propagation.

`src/gesture_control/training/datamodule.py` — `GestureDataModule`: deterministic
recording-level group split (`train_ratio`, `val_ratio`, `test_ratio`);
`torch.utils.data.DataLoader` construction.

`src/gesture_control/training/train_lstm.py` — training loop: early stopping,
cosine-annealing scheduler, label smoothing, checkpointing, structured run metrics.

`src/gesture_control/training/evaluate.py` — accuracy, macro-F1, confusion matrix,
calibration report.

`src/gesture_control/training/export_checkpoint.py` — serialises model weights,
metadata (architecture, schema version, class labels), and SHA256 checksum; appends
to `models/registry.json`.

`scripts/train_model.py` — full training CLI (validate → train → export → register).

### IPN Hand Dataset Integration

`scripts/build_ipn_dataset.py` — full IPN dataset builder: landmark extraction from
raw IPN video frames, gesture class filtering, feature schema application, sliding-window
NPZ construction. Added `--unknown-subsample-ratio` to reduce class imbalance.

`configs/gestures/vocabulary_ipn.yaml` — 10-class IPN-derived gesture vocabulary
replacing the legacy `v1` vocabulary.

### Prediction Stabilisation and Overlay

`src/gesture_control/infrastructure/feedback/overlay_renderer.py` — `OverlayRenderer`:
renders gesture label, confidence bar, state indicator, and later a gaze-aware colored
border onto the OpenCV frame.

`src/gesture_control/evaluation/benchmark_latency.py` + `scripts/benchmark_system.py` —
per-stage latency benchmarking and reporting.

### Testing

Baseline test coverage (≥ 200 tests passing by end of phase):

| Suite                                               | Coverage                               |
| --------------------------------------------------- | -------------------------------------- |
| `tests/unit/test_feature_engineering.py`            | 147-dim schema, normalisation          |
| `tests/unit/test_sequence_buffer.py`                | Fill ratio, FIFO ordering              |
| `tests/unit/test_mediapipe_extractor.py`            | Landmark normalisation                 |
| `tests/unit/test_opencv_source.py`                  | Frame source lifecycle                 |
| `tests/unit/test_checkpoint_loader.py`              | Artifact loading and schema guard      |
| `tests/unit/test_prediction_stabilization.py`       | Voting window                          |
| `tests/unit/test_dataset_builder.py`                | NPZ construction, vocabulary alignment |
| `tests/unit/test_training_datamodule.py`            | Group split determinism                |
| `tests/unit/test_training_export.py`                | Checksum and registry append           |
| `tests/unit/test_overlay_renderer.py`               | Rendering state machine                |
| `tests/unit/test_latency_benchmark.py`              | Benchmark data model                   |
| `tests/integration/test_phase2_pipeline.py`         | Frame → prediction integration         |
| `tests/integration/test_phase3_feature_pipeline.py` | Feature schema v5 alignment            |
| `tests/integration/test_training_pipeline_smoke.py` | End-to-end training smoke              |

---

## Phase 3 — Secure Dispatch and Application Control (PRs #103–#108; March 2026)

### Dispatch Adapter Layer

`src/gesture_control/infrastructure/dispatch/app_adapters/base.py` — `IApplicationAdapter`
protocol: `target_app`, `execute(command_id)`.

`src/gesture_control/infrastructure/dispatch/app_adapters/keyboard_adapter.py` —
`KeyboardAdapter`: OS keystroke delivery via `pyautogui`; strict safe-key allowlist;
lazy import fallback for CI/headless environments.

`src/gesture_control/infrastructure/dispatch/app_adapters/powerpoint_adapter.py` —
logical command ID → PowerPoint keyboard shortcut translation.

`src/gesture_control/infrastructure/dispatch/app_adapters/zoom_adapter.py` — logical
command ID → Zoom keyboard shortcut translation.

`src/gesture_control/infrastructure/dispatch/command_dispatcher.py` —
`CommandDispatcher`: allowlist enforcement, per-window deque rate limiting, adapter
exception isolation, structured `DispatchResult`.

### IPN Vocabulary Migration

The runtime vocabulary was migrated from the legacy v1 gesture set to the IPN-derived set:
`swipe_left`, `swipe_right`, `swipe_up`, `swipe_down`, `zoom_in`, `zoom_out`,
`open_palm_hold`, `pinch_hold`, `point_hold`, `unknown`.

Legacy `vocabulary_v1.yaml`, `command_map.yaml`, and `production.yaml` deprecated with
forwarding content.

### Edge-Triggered Dispatch

`GestureActivationController` extended with `_last_dispatched_label` suppression:
a repeated identical gesture emits only one command until a neutral (`unknown`) prediction
resets dispatch memory. Configurable via `edge_trigger_enabled` in `configs/inference.yaml`.

### Integration Tests

`tests/integration/test_gesture_dispatch_pipeline.py` — end-to-end suite covering:
IDLE → ARMED transition, ARMED → COOLDOWN → ARMED lifecycle, command dispatch through
`CommandDispatcher` and `PowerPointAdapter`, allowlist rejection, rate-limit suppression,
IPN vocabulary alignment, edge-trigger single-dispatch guarantee, and same-gesture reset.

`tests/unit/test_adapters.py` — adapter allowlist, shortcut routing, allowlist rejection,
rate-limit guards.

---

## Phase 4 — Gaze-Aware Activation (PRs #109–#121; April 2026)

### Gaze Infrastructure

`src/gesture_control/interfaces/gaze_detector.py` — `IGazeIntentDetector` protocol.

`src/gesture_control/infrastructure/gaze/head_pose_detector.py` —
`HeadPoseGazeDetector`: MediaPipe Face Mesh landmarks → OpenCV `solvePnP` head-pose
estimation → gaze-forward classification; hysteresis thresholds, engagement timing,
configurable frame subsampling. State-change callback interface.

`src/gesture_control/infrastructure/gaze/null_detector.py` — `NullGazeDetector`:
null-object fallback for disabled-gaze configurations.

### Domain and Pipeline Integration

`src/gesture_control/domain/entities.py` — `RuntimeState` extended with `gaze_permitted`.

`src/gesture_control/pipeline/activation.py` — `GestureActivationController` extended
with `force_idle()` (immediate gaze-loss reset) and `rearm()` (gaze-restoration recovery).

`src/gesture_control/app/runtime.py` — gaze detection gated ahead of hand-landmark
extraction; rising-edge rearm logic; detector teardown in all exit paths.

`src/gesture_control/infrastructure/feedback/overlay_renderer.py` — gaze-aware colored
border: red (gaze absent), amber (gaze present, not armed), green (gaze present, armed).

### Final Activation Model Refactor

- Gesture-based `open_palm_hold` activation removed from runtime logic.
- `activation_hold_ms` and `activation_gesture` removed from configuration and schema.
- Gaze restoration is now the sole re-arming mechanism.
- `open_palm_hold` retained as a model output class only (for classifier compatibility).

### Test Additions

`tests/unit/test_head_pose_gaze_detector.py` (428 lines) — state transitions, hysteresis,
callback behaviour, cached sampling, null-detector fallback.

`tests/unit/test_activation_controller.py` (248 lines) — `force_idle()`, gaze-aware
rearm, ARMED persistence semantics.

`tests/unit/test_runtime.py` — gaze-path coverage, detector teardown.

---

## Phase 5 — Dependency and CI Hardening (PRs #122–#130; April 2026)

### OpenMP / scikit-learn Isolation

`scikit-learn` moved from core runtime dependencies to the `[training]` optional extra
in `pyproject.toml`. CI split into two jobs:

- `runtime` — installs `.[dev]`, runs `pytest -m "not training"` and `mypy src/`
- `training` — installs `.[dev,training]`, runs the full test suite

`tests/conftest.py` — `pytest_ignore_collect()` hook skips test files whose name
contains `training` when `scikit-learn` is absent.

### Extras and Typing Cleanup

`dev` optional dependency group added: `pytest`, `ruff`, `mypy`, `types-PyYAML`,
`types-PyAutoGUI`. `requirements/dev.txt` reduced to a single `pip install -e .[dev,training]` line.

`AppContainer` fields typed against interface protocols rather than concrete
implementations.

`src/gesture_control/infrastructure/model/lstm_model.py` extracted as a shared module
importable without the training package.

### IPN / Jester Fusion (PR #129, reverted PR #130)

An attempt to fuse the IPN and Jester gesture datasets was introduced in PR #129 and
reverted in PR #130 after evaluation showed degraded classifier consistency. The revert
restored the IPN-only vocabulary baseline.

---

## Phase 6 — Model Improvement Campaign (PRs #131–#145; April 2026)

Full detail in `docs/archive/work-logs/2026-04-18-model-improvement-campaign.md`.
Summary:

### ML Workflow Operationalisation (PRs #131, #134, #135, #136)

- `src/gesture_control/training/hf_publish.py` (448 lines) — Hugging Face Hub
  publishing: two-stage commit (model card + config, then binary artifact);
  `build_model_card()` with per-class metrics and 147-dim feature schema table.
- `notebooks/01_dataset_preparation.ipynb` — Layer 1 dataset preparation workflow.
- `notebooks/02_model_training.ipynb` — Layer 2 training, evaluation, export, publishing.
- `GestureDataModule` extended with `train_indices`, `val_indices`, `test_indices`
  `@property` accessors (PR #134 by @blackbird410).
- `scripts/build_ipn_dataset.py` — `--unknown-subsample-ratio` flag added.
- Artifact path resolution corrected to be relative to registry file (PR #136).

### Feature Schema Evolution

| Version | Dims | Added block                                                  | PR      |
| ------- | ---- | ------------------------------------------------------------ | ------- |
| v1      | 126  | Positions + velocities                                       | Phase 2 |
| v2      | 126  | Wrist trajectory Δ + fingertip spread (replaces old v1 calc) | #138    |
| v3      | 144  | +10 MCP/PIP joint angles                                     | #138    |
| v4      | 144  | Baseline training round                                      | #135    |
| v5      | 147  | +global wrist velocity (`vwx, vwy, vwz`)                     | #138    |

Final 147-dim schema:

```
[x₀…z₂₀]              ← 63 dims  (positions)
[d₀₁…d₀₅]             ←  5 dims  (fingertip spread)
[Δwx, Δwy, Δwz]        ←  3 dims  (wrist trajectory delta)
[vx₀…vz₂₀]            ← 63 dims  (velocity)
[θ₀…θ₉]               ← 10 dims  (joint angles)
[vwx, vwy, vwz]        ←  3 dims  (wrist velocity, camera-normalised)
                        ─────────
                          147 dims total  (feature-schema-v5)
```

### TCN Evaluation and Retirement

`src/gesture_control/infrastructure/model/tcn_model.py` (120 lines) — TCN with dilated
causal convolutions, residual connections, weight normalisation. Two training runs
completed. Macro-F1 0.87 vs LSTM 0.94 at `feature-schema-v3`. TCN retired; all TCN
configs removed in PR #143.

### Horizontal Swipe Fix

Label-aware mirror augmentation: `SWIPE_LEFT` / `SWIPE_RIGHT` labels toggled when the
horizontal flip is applied; `vwx` (index 146) added to the flip axis list.

### HF Hub Migration (PRs #139–#144)

- 47 historical docs archived to `docs/archive/`; `docs/work-logs/` removed.
- `ModelSourceConfig` Pydantic model (`backend: local|huggingface`).
- `configs/model_source.yaml` — ships with `backend: huggingface`, pinned to
  `lstm_v1_20260418T023120Z_inference.pt`.
- `src/gesture_control/infrastructure/model/hf_loader.py` — `load_artifact_from_hub()`
  calls `hf_hub_download()` then delegates to the local loader.
- `bootstrap.py` routes on `model_source.backend`; `--hf-revision` CLI flag added;
  `--registry` deprecated.
- 21 `.pt` artifacts (~22 MB) + `models/registry.json` deleted from VCS.
- `huggingface_hub>=0.22.0` promoted to core dependency.
- 3 offline unit tests for the HF loader.

---

## Training Experiments

| Run                        | Schema | Macro-F1 | Notes                     |
| -------------------------- | ------ | -------- | ------------------------- |
| `lstm_v1_20260328T100850Z` | v1     | —        | Foundation baseline       |
| `lstm_v1_20260330T181558Z` | v1     | —        | IPN vocabulary training   |
| `lstm_v1_20260405T082546Z` | v3     | —        | v3 feature evaluation     |
| `lstm_v1_20260405T132108Z` | v3     | —        |                           |
| `lstm_v1_20260413T023049Z` | v3     | —        |                           |
| `lstm_v1_20260413T064039Z` | v3     | —        |                           |
| `tcn_v1_20260413T065716Z`  | v3     | 0.87     | TCN evaluation            |
| `tcn_v1_20260417T052818Z`  | v3     | —        | TCN second run            |
| `lstm_v1_20260417T*` (×5)  | v4     | —        | v4 notebook workflow      |
| `lstm_v1_20260418T002927Z` | v5     | —        | v5 training               |
| `lstm_v1_20260418T004149Z` | v5     | —        |                           |
| `lstm_v1_20260418T023120Z` | v5     | **0.96** | Production model → HF Hub |

16 experiment runs committed to `experiments/runs/` (config snapshots, dataset
metadata, per-epoch training metrics JSON).

---

## Final System State

| Metric                 | State                                                                                 |
| ---------------------- | ------------------------------------------------------------------------------------- |
| Test suite             | **≥ 250 tests, 0 failures**                                                           |
| CI                     | Runtime + training job tracks; `mypy src/` enforced                                   |
| Inference schema       | `feature-schema-v5`, 147-dim                                                          |
| Production model       | `ntsrigaud/gesture-control-lstm` on Hugging Face Hub (Macro-F1 **0.96**)              |
| Runtime deployment     | `pip install -e . && python scripts/run_realtime.py` — no prior training run required |
| Tracked binary weights | **0 MB** (all `.pt` and `registry.json` excluded from VCS)                            |
| Gaze activation        | `HeadPoseGazeDetector` as sole arming mechanism                                       |
| Secure dispatch        | `KeyboardAdapter` allowlist; `CommandDispatcher` rate limiting                        |
| Application targets    | Microsoft PowerPoint, Zoom (extensible via `IApplicationAdapter`)                     |
