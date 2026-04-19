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
| `swipe_up`       | Start presentation           | G03             |
| `swipe_down`     | Exit presentation            | G04             |
| `zoom_in`        | Toggle pointer mode          | G10             |
| `zoom_out`       | Raise hand                   | G11             |
| `unknown`        | None (noise class)           | D0X + B0A + B0B |

## Quick Start

**End users** (runtime inference only):

```bash
pip install -e ".[huggingface]"
python scripts/run_realtime.py --config configs
```

The default configuration (`configs/model_source.yaml`) loads the production model from [Hugging Face Hub](https://huggingface.co/ntsrigaud/gesture-control-lstm) on first run. The artifact is cached locally by `huggingface_hub` and requires no re-download on subsequent runs.

**Developers** (linting, type-checking, and tests):

```bash
pip install -e ".[dev]"
ruff check .
mypy src/
pytest -m "not training"
```

## Training Setup

Install the `training` and `dev` extras (adds scikit-learn and dev tools):

```bash
pip install -e ".[dev,training,huggingface]"
```

Then run the full test suite (including training tests):

```bash
pytest
```

The recommended training workflow uses the Jupyter notebooks under `notebooks/`:

| Notebook                       | Purpose                                                  |
| ------------------------------ | -------------------------------------------------------- |
| `01_dataset_preparation.ipynb` | Build the NPZ dataset from IPN Hand video landmarks      |
| `02_model_training.ipynb`      | Train, evaluate, export, and publish to Hugging Face Hub |

Install the notebook extra and launch Jupyter:

```bash
pip install -e ".[notebooks,training,huggingface]"
jupyter lab
```

Alternatively, use the CLI scripts:

```bash
python scripts/build_ipn_dataset.py --ipn-root data/IPN --output-npz data/processed/ipn_dataset.npz
python scripts/train_model.py --dataset data/processed/ipn_dataset.npz --config-dir configs
```

## Project Architecture

See `docs/architecture/` for the full system specification, module contracts, and design rationale.

Key directories:

| Directory              | Purpose                                                                                                                                    |
| ---------------------- | ------------------------------------------------------------------------------------------------------------------------------------------ |
| `src/gesture_control/` | Core package (domain, interfaces, infrastructure, pipeline, training)                                                                      |
| `configs/`             | YAML configuration (app, inference, model, gestures, thresholds)                                                                           |
| `scripts/`             | CLI entry points (data collection, dataset build, training, inference)                                                                     |
| `models/`              | Local training artifacts (`.gitignore`d); production model hosted on [Hugging Face](https://huggingface.co/ntsrigaud/gesture-control-lstm) |
| `tests/`               | Unit, integration, and system tests                                                                                                        |
| `docs/`                | Architecture docs, guides, and decision records                                                                                            |

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

## Current System State

| Metric                   | Value                                                                                                         |
| ------------------------ | ------------------------------------------------------------------------------------------------------------- |
| Inference feature schema | `feature-schema-v5` (147-dim)                                                                                 |
| Production model         | [`ntsrigaud/gesture-control-lstm`](https://huggingface.co/ntsrigaud/gesture-control-lstm) — Macro-F1 **0.96** |
| Model loading            | Hugging Face Hub (cached, no training required)                                                               |
| Gaze activation          | `HeadPoseGazeDetector` — sole arming mechanism                                                                |
| Test suite               | ≥ 250 tests, 0 failures                                                                                       |
| Tracked binary weights   | **0 MB** — all `.pt` artifacts excluded from VCS                                                              |

The complete development history is recorded in [docs/architecture/](docs/architecture/) and [docs/archive/work-logs/](docs/archive/work-logs/).
