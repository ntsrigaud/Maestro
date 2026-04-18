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

| Gesture          | Command                      | IPN Source      |
| ---------------- | ---------------------------- | --------------- |
| `open_palm_hold` | No action (model class only) | G07             |
| `swipe_right`    | Next slide                   | G06             |
| `swipe_left`     | Previous slide               | G05             |
| `swipe_up`       | Raise hand                   | G03             |
| `swipe_down`     | Start presentation           | G04             |
| `zoom_in`        | Toggle pointer mode          | G10             |
| `zoom_out`       | Exit presentation            | G11             |
| `unknown`        | None (noise class)           | D0X + B0A + B0B |

## Quick Start

**End users** (runtime inference only):

```bash
pip install -e .
python scripts/run_realtime.py --config configs
```

**Developers** (linting, type-checking, and tests):

```bash
pip install -e ".[dev]"
ruff check .
mypy src/
pytest -m "not training"
```

## Training Setup

Install the `training` extra (scikit-learn) alongside the `dev` tools:

```bash
pip install -e ".[dev,training]"
```

Then run the full test suite (including training tests):

```bash
pytest
```

To build a dataset and train a model:

```bash
python scripts/build_ipn_dataset.py --ipn-root data/IPN --output-npz data/processed/ipn_dataset.npz
python scripts/train_model.py --dataset data/processed/ipn_dataset.npz --config-dir configs
```

## Project Architecture

See `docs/architecture/` for the full system specification, module contracts, and design rationale.

Key directories:

| Directory              | Purpose                                                                |
| ---------------------- | ---------------------------------------------------------------------- |
| `src/gesture_control/` | Core package (domain, interfaces, infrastructure, pipeline, training)  |
| `configs/`             | YAML configuration (app, inference, model, gestures, thresholds)       |
| `scripts/`             | CLI entry points (data collection, dataset build, training, inference) |
| `models/`              | Local training artifacts (`.gitignore`d); production model hosted on [Hugging Face](https://huggingface.co/ntsrigaud/gesture-control-lstm) |
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

Legacy custom webcam collection material has been removed. Contributors should follow `docs/IPN_INTEGRATION_GUIDE.md` for the canonical dataset workflow.

## Model Training

```bash
python scripts/train_model.py \
    --dataset data/processed/ipn_dataset.npz \
    --config configs/model/lstm_default.yaml \
    --config-dir configs
```

The training pipeline includes group-aware data splitting (no recording leakage), data augmentation (temporal scaling + spatial jitter), early stopping, and class-weighted loss.

## Implementation Status

All architectural phases are implemented and validated:

- Phase 1: Project scaffolding, interface contracts, configuration system
- Phase 2: OpenCV frame source, MediaPipe landmark extraction
- Phase 3: Feature engineering, temporal buffering, dataset generation
- Phase 4: LSTM training pipeline with augmentation and evaluation
- Phase 5: Real-time inference pipeline with latency benchmarking
- Phase 6: Activation state machine, command dispatch, application adapters
- Phase 7: Prediction stabilization, edge-trigger dispatch, safety hardening
