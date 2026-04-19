# System Overview

## 1. Introduction

The Gesture Control System is a real-time gesture control application that enables hands-free desktop interaction. The system captures webcam video, extracts hand landmarks using MediaPipe Hands, classifies temporal gesture sequences with an LSTM neural network, and dispatches operating-system-level keyboard commands for navigation, scrolling, zoom, and desktop management.

The architecture follows a modular, protocol-driven design with strict separation between domain logic, infrastructure adapters, and pipeline orchestration. All runtime parameters are externalized to YAML configuration files. The system is CPU-first and operates at interactive frame rates on consumer hardware.

## 2. System Objective

Provide reliable, low-latency gesture-based desktop control using a small, highly separable gesture vocabulary. The system prioritizes safety (low false-trigger rate) over recognition speed through multi-stage prediction stabilization and an explicit activation state machine.

## 3. Gesture Vocabulary

The system recognizes eight gesture classes derived from the IPN Hand dataset:

| Gesture          | Type            | Mapped Command       |
| ---------------- | --------------- | -------------------- |
| `open_palm_hold` | Static          | Mission Control      |
| `swipe_right`    | Dynamic         | Navigate forward     |
| `swipe_left`     | Dynamic         | Navigate back        |
| `swipe_up`       | Dynamic         | Scroll back          |
| `swipe_down`     | Dynamic         | Scroll forward       |
| `zoom_in`        | Dynamic         | Zoom in view         |
| `zoom_out`       | Dynamic         | Zoom out view        |
| `unknown`        | Rejection class | No action            |

The activation gate is controlled entirely by gaze detection: the system starts in ARMED state and accepts command gestures as long as the user's gaze is directed at the camera. Turning away (gaze lost beyond the grace period) transitions the system to IDLE; looking back triggers `rearm()` and resumes ARMED state. `swipe_up`/`swipe_down` scroll through content, `swipe_left`/`swipe_right` navigate between virtual desktops or browser tabs (configurable via `navigation_mode`), `zoom_in`/`zoom_out` adjust view magnification, and `open_palm_hold` triggers Mission Control / Activities overview.

## 4. High-Level Architecture

The system is organized into four layers:

```
┌─────────────────────────────────────────────────────────┐
│                    Application Layer                    │
│         (Bootstrap, Runtime Loop, CLI Scripts)          │
├─────────────────────────────────────────────────────────┤
│                    Pipeline Layer                       │
│  (Feature Engineering, Buffering, Stabilization,        │
│   Activation Control)                                   │
├─────────────────────────────────────────────────────────┤
│                 Infrastructure Layer                    │
│  (OpenCV Camera, MediaPipe Extractor, LSTM Model,       │
│   Command Dispatcher, Keyboard Adapters, Overlay)       │
├─────────────────────────────────────────────────────────┤
│                    Domain Layer                         │
│  (Entities, Enums, Contracts, Events)                   │
└─────────────────────────────────────────────────────────┘
```

**Domain Layer**: Pure data structures and interface contracts with no framework dependencies. All pipeline data types (`FramePacket`, `LandmarkPacket`, `FeatureFrame`, `SequenceWindow`, `Prediction`, `StabilizedPrediction`, `ExecutableCommand`, `DispatchResult`) are frozen dataclasses.

**Infrastructure Layer**: Concrete implementations of domain interfaces using external libraries (OpenCV, MediaPipe, PyTorch, pyautogui). Each implementation is isolated behind a Protocol interface.

**Pipeline Layer**: Stateful processing components that transform data between pipeline stages. Includes feature normalization, temporal windowing, prediction smoothing, and the activation state machine.

**Application Layer**: Composition root (`bootstrap.py`) wires all dependencies from configuration. The runtime loop (`runtime.py`) orchestrates the sequential frame processing pipeline.

## 5. End-to-End Pipeline Summary

The real-time inference pipeline processes each camera frame through ten sequential stages:

```mermaid
graph LR
    A[Camera Frame] --> B[Gaze Estimation]
    B --> C[Landmark Extraction]
    C --> D[Feature Engineering]
    D --> E[Temporal Buffering]
    E --> F[LSTM Inference]
    F --> G[Prediction Stabilization]
    G --> H[Activation Controller]
    H --> I[Command Dispatch]
    I --> J[Keyboard Execution]
```

1. **Frame Capture**: OpenCV reads a BGR frame from the webcam at 30 FPS (1280×720).
2. **Gaze Estimation**: MediaPipe Face Mesh estimates head pose from facial landmarks. If the user's gaze is not directed toward the camera, all downstream gesture processing is bypassed for that frame.
3. **Landmark Extraction**: MediaPipe Hands detects 21 hand landmarks with confidence filtering.
4. **Feature Engineering**: Landmarks are wrist-centered, scale-normalized, and augmented with velocity, joint angles, fingertip spread, and wrist trajectory features to produce a 147-dimensional feature vector (feature-schema-v5).
5. **Temporal Buffering**: A sliding window buffer accumulates 24 consecutive feature vectors with a step size of 3 frames.
6. **LSTM Inference**: The buffered sequence (24×147) is classified by a 2-layer LSTM into one of 8 gesture classes.
7. **Prediction Stabilization**: EMA smoothing, per-class confidence thresholding, and temporal majority voting filter transient predictions.
8. **Activation Control**: A three-state machine (IDLE / ARMED / COOLDOWN) gates command emission. The system starts in ARMED state; gaze loss transitions to IDLE; gaze restoration calls `rearm()` to return to ARMED. Edge-triggered dispatch is enforced throughout.
9. **Command Dispatch**: The gesture-to-command mapping is resolved, validated against an allowlist, rate-limited, and routed to the appropriate application adapter.
10. **Keyboard Execution**: The adapter translates commands to keyboard shortcuts and invokes them via pyautogui with key-name allowlist validation.

## 6. Key Architectural Characteristics

### 6.1 Safety-First Command Dispatch

Multiple independent safety mechanisms prevent accidental command execution:

- **Gaze gating**: The entire gesture pipeline is bypassed unless the user's head pose indicates attention toward the camera. This is the outermost safety layer and eliminates unintentional activations when the user is not facing the display.
- **Activation gating**: Commands are only accepted while the system is in ARMED mode.
- **Edge-triggered dispatch**: A gesture triggers a command only on its first accepted occurrence; holding the same gesture does not re-fire.
- **Temporal voting**: 1–3 consecutive frames must agree on the same class before acceptance.
- **Per-class confidence thresholds**: Each gesture class has an independent acceptance threshold.
- **Command allowlist**: Only pre-approved command IDs reach the OS keyboard layer.
- **Rate limiting**: A sliding-window rate limiter caps dispatch frequency.
- **Key-name allowlist**: The keyboard adapter validates every key name against a safe set before invoking pyautogui.

### 6.2 Training–Inference Parity

The same `LandmarkFeatureTransformer` and `SlidingWindowSequenceBuffer` classes are used in both the training dataset builder and the real-time inference pipeline. Feature schema metadata embedded in the NPZ dataset and model checkpoint enables automated compatibility validation at runtime.

### 6.3 Configuration-Driven Operation

All operational parameters—camera settings, inference window geometry, confidence thresholds, gesture vocabulary, command mappings, model hyperparameters—are externalized to YAML files under `configs/`. No thresholds or behavioral constants are hardcoded in source.

### 6.4 Protocol-Based Modularity

Nine `Protocol` interfaces define service contracts. Implementations can be swapped without modifying consuming code. This enables isolated unit testing with mock implementations and supports future extension (additional application adapters, alternative model backends).

## 7. Technology Stack

| Component               | Technology        | Version                                                                |
| ----------------------- | ----------------- | ---------------------------------------------------------------------- |
| Language                | Python            | 3.11                                                                   |
| Hand Pose Estimation    | MediaPipe Hands   | 0.10.14 (pinned)                                                       |
| Sequence Classification | PyTorch LSTM      | ≥2.7.0                                                                 |
| Frame Capture           | OpenCV (headless) | ≥4.10.0                                                                |
| OS Keyboard Control     | pyautogui         | ≥0.9.54                                                                |
| Config Validation       | Pydantic          | ≥2.6.0                                                                 |
| Array Operations        | NumPy             | ≥1.26.0                                                                |
| Model Hosting           | Hugging Face Hub  | ≥0.22.0 (optional `[huggingface]` extra; required for default backend) |
| ML Metrics (training)   | scikit-learn      | ≥1.5.0 (optional `[training]` extra)                                   |

## 8. Training Infrastructure

The offline training pipeline converts labeled video data into a trained LSTM checkpoint:

1. **Data Collection**: The IPN Hand dataset (200 videos, 8 classes) is the canonical training corpus. `scripts/build_ipn_dataset.py` extracts landmarks from IPN video files and produces NPZ sequence datasets.
2. **Model Training**: `scripts/train_model.py` trains a 2-layer LSTM with class-weighted cross-entropy loss, Adam optimizer, learning rate scheduling, and early stopping.
3. **Model Export**: Trained checkpoints are exported as inference artifacts containing the state dictionary, class labels, and feature schema metadata, then published to the Hugging Face Hub (`ntsrigaud/gesture-control-lstm`). At runtime the artifact is downloaded on first use and served from the local Hugging Face cache.

---

**Document**: System Overview  
**Suite**: Architecture Documentation (`docs/architecture/`)  
**Last Updated**: April 2026
