# Presentation and Meeting Control System

Production-grade webcam-based gesture control system using MediaPipe Hands and an LSTM classifier for presentation and meeting workflows.

## System Overview

The system recognizes 8 hand gestures from a live webcam feed and maps them to application commands (slide navigation, raise hand, mute, etc.). It runs in real time on consumer laptops without a GPU.

**Core pipeline:**

```
Camera → MediaPipe Hands → Feature Engineering → Sliding Window Buffer
→ LSTM Inference → Prediction Stabilization → Activation Controller
→ Command Dispatch → Application Adapter → OS Keyboard Events
```

**Gesture vocabulary** (IPN Hand dataset, 8 classes):

| Gesture          | Command                 | IPN Source      |
| ---------------- | ----------------------- | --------------- |
| `open_palm_hold` | Activation (arm/disarm) | G07             |
| `swipe_right`    | Next slide              | G06             |
| `swipe_left`     | Previous slide          | G05             |
| `swipe_up`       | Raise hand              | G03             |
| `swipe_down`     | Start presentation      | G04             |
| `zoom_in`        | Toggle pointer mode     | G10             |
| `zoom_out`       | Exit presentation       | G11             |
| `unknown`        | None (noise class)      | D0X + B0A + B0B |

## Quick Start

1. Create or activate a Python 3.11 environment.
2. Install dependencies:

```bash
pip install -r requirements/dev.txt
pip install -e .
```

3. Run quality checks:

```bash
ruff check .
pytest
```

4. Launch real-time inference (requires a trained model in `models/`):

```bash
python scripts/run_realtime.py --config configs
```

## Project Architecture

See `ARCHITECTURE_SPECIFICATION.md` for the full system specification, module contracts, and design rationale.

Key directories:

| Directory              | Purpose                                                                |
| ---------------------- | ---------------------------------------------------------------------- |
| `src/gesture_control/` | Core package (domain, interfaces, infrastructure, pipeline, training)  |
| `configs/`             | YAML configuration (app, inference, model, gestures, thresholds)       |
| `scripts/`             | CLI entry points (data collection, dataset build, training, inference) |
| `models/`              | Checkpoints, exported artifacts, and model registry                    |
| `tests/`               | Unit, integration, and system tests                                    |
| `docs/`                | Architecture docs, guides, and decision records                        |

## Dataset Generation

The training dataset is built from the IPN Hand dataset (200 videos, 14 gesture classes). See `docs/IPN_INTEGRATION_GUIDE.md` for the full pipeline.

```bash
python scripts/build_ipn_dataset.py \
    --ipn-root data/IPN \
    --output-npz data/processed/ipn_dataset.npz \
    --num-workers 4
```

For custom data collection, see `docs/DATASET_RECORDING_WORKFLOW.md`:

```bash
python scripts/collect_data.py \
    --label swipe_right \
    --recording-id rec-swipe-right \
    --output-jsonl data/raw/labeled_landmarks.jsonl \
    --interactive

python scripts/build_dataset.py \
    --input-jsonl data/raw/labeled_landmarks.jsonl \
    --output-npz data/processed/gesture_dataset.npz \
    --config-dir configs \
    --label-strategy majority
```

## Model Training

```bash
python scripts/train_model.py \
    --dataset data/processed/ipn_dataset.npz \
    --config configs/model/lstm_default.yaml \
    --config-dir configs
```

The training pipeline includes group-aware data splitting (no recording leakage), data augmentation (temporal scaling + spatial jitter), early stopping, and class-weighted loss. See `docs/DATA_COLLECTION_GUIDE.md` for details.

## Implementation Status

All architectural phases are implemented and validated:

- Phase 1: Project scaffolding, interface contracts, configuration system
- Phase 2: OpenCV frame source, MediaPipe landmark extraction
- Phase 3: Feature engineering, temporal buffering, dataset generation
- Phase 4: LSTM training pipeline with augmentation and evaluation
- Phase 5: Real-time inference pipeline with latency benchmarking
- Phase 6: Activation state machine, command dispatch, application adapters
- Phase 7: Prediction stabilization, edge-trigger dispatch, safety hardening
