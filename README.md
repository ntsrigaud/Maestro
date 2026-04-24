# Maestro

Production-grade webcam-based gesture recognition system for hands-free presentation control, powered by MediaPipe Hands and a two-stream LSTM classifier.

## System Overview

The system recognizes 8 hand gestures from a live webcam feed and maps them to application commands (slide navigation, view zoom, presentation start/stop). It runs in real time on consumer laptops without a GPU.

**Core pipeline:**

```
Camera → MediaPipe Hands → Feature Engineering → Sliding Window Buffer
→ LSTM Inference → Prediction Stabilization → Activation Controller
→ Command Dispatch → Application Adapter → OS Keyboard Events
```

A **PySide6 graphical interface** provides runtime control, live telemetry monitoring, configuration editing, and model selection — all without interfering with the inference pipeline. The inference loop runs in a background `QThread`; the UI receives state updates through Qt signal-slot connections at approximately 10 Hz.

**Two launch modes:**

| Mode    | Script                    | Description                                             |
| ------- | ------------------------- | ------------------------------------------------------- |
| **GUI** | `scripts/run_ui.py`       | PySide6 control panel with monitoring and config editor |
| **CLI** | `scripts/run_realtime.py` | Headless inference only; no PySide6 dependency          |

**Gesture vocabulary** (IPN Hand dataset, 8 classes):

| Gesture          | Command                   | IPN Source      |
| ---------------- | ------------------------- | --------------- |
| `open_palm_hold` | Start / stop presentation | G07             |
| `swipe_right`    | Scroll forward            | G06             |
| `swipe_left`     | Scroll back               | G05             |
| `swipe_up`       | Previous slide            | G03             |
| `swipe_down`     | Next slide                | G04             |
| `zoom_in`        | Zoom in view              | G10             |
| `zoom_out`       | Zoom out view             | G11             |
| `unknown`        | No action (noise class)   | D0X + B0A + B0B |

## Quick Start

**GUI (recommended)** — control panel with live monitoring and configuration editor:

```bash
pip install -e ".[dev,training,huggingface]"
python scripts/run_ui.py
```

**CLI** — headless inference only:

```bash
pip install -e ".[huggingface]"
python scripts/run_realtime.py --config configs
```

The default configuration (`configs/model_source.yaml`) loads the production model from [Hugging Face Hub](https://huggingface.co/ntsrigaud/maestro-lstm) on first run. The artifact is cached locally by `huggingface_hub` and requires no re-download on subsequent runs.

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

| Directory      | Purpose                                                                                                                            |
| -------------- | ---------------------------------------------------------------------------------------------------------------------------------- |
| `src/maestro/` | Core package (domain, interfaces, infrastructure, pipeline, training, app/ui)                                                      |
| `configs/`     | YAML configuration (app, inference, model, gestures, thresholds)                                                                   |
| `scripts/`     | CLI entry points (`run_ui.py`, `run_realtime.py`, data, training)                                                                  |
| `models/`      | Local training artifacts (`.gitignore`d); production model hosted on [Hugging Face](https://huggingface.co/ntsrigaud/maestro-lstm) |
| `tests/`       | Unit, integration, and system tests (269 total)                                                                                    |
| `docs/`        | Architecture docs, guides, and decision records                                                                                    |

### GUI Architecture

The PySide6 UI (`src/maestro/app/ui/`) is a non-blocking control surface:

- `RuntimeController` owns a `QThread` worker that runs `build_container()` and `Runtime.run()`.
- `Runtime` receives an optional `on_state` callback; the worker thread throttles callback emissions to ~10 Hz and forwards them as Qt signals to the main thread.
- `MainWindow` mediates all signal routing between panels and the controller; no panel communicates with the controller directly.
- Configuration changes are validated in-memory by Pydantic; YAML files are written only after the runtime stops and only on explicit user action.
- A `QSystemTrayIcon` provides ambient gesture feedback (colour-coded: grey / green / amber) when the window is minimised.

See [docs/architecture/ui-view.md](docs/architecture/ui-view.md) for the complete UI architecture documentation.

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

| Metric                   | Value                                                                                         |
| ------------------------ | --------------------------------------------------------------------------------------------- |
| Inference feature schema | `feature-schema-v5` (147-dim)                                                                 |
| Production model         | [`ntsrigaud/maestro-lstm`](https://huggingface.co/ntsrigaud/maestro-lstm) — Macro-F1 **0.96** |
| Model loading            | Hugging Face Hub (cached, no training required)                                               |
| Gaze activation          | `HeadPoseGazeDetector` — sole arming mechanism                                                |
| Test suite               | 269 tests, 0 failures                                                                         |
| Tracked binary weights   | **0 MB** — all `.pt` artifacts excluded from VCS                                              |

The complete development history is recorded in [docs/architecture/](docs/architecture/) and [docs/archive/work-logs/](docs/archive/work-logs/).
