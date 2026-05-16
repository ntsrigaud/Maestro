# Maestro

Production-grade webcam-based gesture recognition system for hands-free presentation control, powered by MediaPipe Hands and a two-stream LSTM classifier.

## System Overview

The system recognizes a 10-class runtime vocabulary from a live webcam feed (9 active gestures + `unknown`) and executes validated presentation controls in real time on consumer laptops without a GPU.

**Core pipeline:**

```
Camera → MediaPipe Hands → Hand Selection → Feature Engineering → Sliding Window Buffer
→ LSTM Inference → Prediction Stabilization → Activation Controller
→ Command Dispatch → Application Adapter → OS Keyboard/Pointer Events
```

A **PySide6 graphical interface** provides runtime control, live telemetry monitoring, configuration editing, and model selection — all without interfering with the inference pipeline. The inference loop runs in a background `QThread`; the UI receives state updates through Qt signal-slot connections at approximately 10 Hz.

**Two launch modes:**

| Mode    | Entry point  | Description                                             |
| ------- | ------------ | ------------------------------------------------------- |
| **GUI** | `maestro-ui` | PySide6 control panel with monitoring and config editor |
| **CLI** | `maestro`    | Headless inference only; no PySide6 dependency          |

For developers who prefer script-based invocation, `scripts/run_ui.py` and `scripts/run_realtime.py` remain available as thin wrappers.

**Gesture vocabulary** (hybrid Jester+IPN, 10 classes):

| Gesture                 | Command                   | Source  |
| ----------------------- | ------------------------- | ------- |
| `fist`                  | Erase annotations         | LD-CONG |
| `swiping_right`         | Next slide                | Jester  |
| `swiping_left`          | Previous slide            | Jester  |
| `swiping_up`            | Start presentation        | Jester  |
| `swiping_down`          | Stop presentation         | Jester  |
| `zooming_in_full_hand`  | Zoom in view              | Jester  |
| `zooming_out_full_hand` | Zoom out view             | Jester  |
| `point_one`             | Continuous laser pointer  | IPN     |
| `point_two`             | Continuous annotation pen | IPN     |
| `unknown`               | No action (noise class)   | IPN     |

Discrete commands are routed through `GestureActivationController` + `CommandDispatcher`.
`point_one` and `point_two` are continuous gestures handled directly by runtime trackers.

## Installation

### Option A — Standalone application (recommended for end users)

Download the pre-built bundle for your platform from the [GitHub Releases](https://github.com/ntsrigaud/Maestro/releases) page:

| Platform    | Download              | Action                                                             |
| ----------- | --------------------- | ------------------------------------------------------------------ |
| **macOS**   | `Maestro-macos.dmg`   | Open the DMG, drag Maestro to Applications, double-click to launch |
| **Windows** | `Maestro-windows.zip` | Extract the archive, run `Maestro.exe` inside the folder           |

No Python installation required. The model is downloaded from Hugging Face Hub on first launch and cached locally.

### Option B — pip install (Python users)

```bash
pip install -e ".[ui]"   # installs PySide6, opencv-python, and all runtime deps
maestro-ui               # launch the GUI
maestro --no-display     # launch headless (CLI)
```

On first launch the application copies its factory-default configuration to:

| Platform | Location                                         |
| -------- | ------------------------------------------------ |
| macOS    | `~/Library/Application Support/Maestro/configs/` |
| Windows  | `%APPDATA%\Maestro\configs\`                     |
| Linux    | `~/.config/Maestro/configs/`                     |

Edit the YAML files there to change camera, hand preference, or gesture mappings. Set `MAESTRO_CONFIG_DIR` to override the location entirely.

## Quick Start (development)

**GUI:**

```bash
pip install -e ".[ui,dev]"
maestro-ui
# or: python scripts/run_ui.py
```

**CLI (headless):**

```bash
pip install -e ".[ui]"
maestro --no-display
# or: python scripts/run_realtime.py --no-display
```

The model is loaded from [Hugging Face Hub](https://huggingface.co/ntsrigaud/maestro-lstm) on first run and cached locally. No re-download on subsequent runs.

**Developers** (linting, type-checking, and tests):

```bash
pip install -e ".[dev]"
ruff check .
mypy src/
pytest -m "not training"
```

## Building from Source

Build a standalone application bundle (no Python required by the end user):

```bash
make build
# Output: dist/Maestro.app (macOS) or dist/Maestro/Maestro.exe (Windows)
```

This uses [PyInstaller](https://pyinstaller.org/) with the included `maestro.spec` to bundle MediaPipe models, config files, and all dependencies into a self-contained executable. Pre-built binaries for macOS and Windows are also produced by the GitHub Actions [build workflow](.github/workflows/build.yml) on every tagged release.

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

| Notebook                                       | Purpose                                                  |
| ---------------------------------------------- | -------------------------------------------------------- |
| `01_dataset_preparation_two-stream-lstm.ipynb` | Build the NPZ dataset from IPN Hand video landmarks      |
| `02_model_training_two-stream-lstm.ipynb`      | Train, evaluate, export, and publish to Hugging Face Hub |

Install the notebook extra and launch Jupyter:

```bash
pip install -e ".[notebooks,training,huggingface]"
jupyter lab
```

Alternatively, use the CLI scripts:

```bash
python scripts/build_ipn_dataset.py --ipn-root data/IPN --output-npz data/processed/ipn_dataset.npz
python scripts/train_model.py --dataset-path data/processed/ipn_dataset.npz --config configs/model/two_stream_attn.yaml --config-dir configs
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
| `tests/`       | Unit, integration, and system test suites                                                                                          |
| `docs/`        | Architecture docs, guides, and decision records                                                                                    |

### GUI Architecture

The PySide6 UI (`src/maestro/app/ui/`) is a non-blocking control surface:

- `RuntimeController` owns a `QThread` worker that runs `build_container()` and `Runtime.run()`.
- `Runtime` receives an optional `on_state` callback; the worker thread throttles callback emissions to ~10 Hz and forwards them as Qt signals to the main thread.
- `MainWindow` mediates all signal routing between panels and the controller; no panel communicates with the controller directly.
- Configuration changes are validated in-memory by Pydantic; YAML files are written only after the runtime stops and only on explicit user action.
- The main UI window is the cross-platform control surface, and a `QSystemTrayIcon` supplements it with ambient gesture feedback (colour-coded: grey / green / amber) when available.

See [docs/architecture/ui-view.md](docs/architecture/ui-view.md) for the complete UI architecture documentation.

## Dataset Generation

The training dataset is built from the IPN Hand dataset (200 videos, 14 gesture classes). See `docs/IPN_INTEGRATION_GUIDE.md` for the full pipeline.

```bash
python scripts/build_ipn_dataset.py \
    --ipn-root data/IPN \
    --format flat \
    --output-npz data/processed/ipn_dataset.npz \
    --num-workers 4
```

Legacy custom webcam collection material has been removed. Contributors should follow `docs/IPN_INTEGRATION_GUIDE.md` for the canonical dataset workflow.

## Model Training

```bash
python scripts/train_model.py \
    --dataset-path data/processed/ipn_dataset.npz \
    --config configs/model/two_stream_attn.yaml \
    --config-dir configs
```

The training pipeline includes group-aware data splitting (no recording leakage), data augmentation (temporal scaling + spatial jitter), early stopping, and scheduler-based learning-rate control.

## Current System State

| Metric                   | Value                                                                                         |
| ------------------------ | --------------------------------------------------------------------------------------------- |
| Inference feature schema | `feature-schema-v5` (147-dim)                                                                 |
| Production model         | [`ntsrigaud/maestro-lstm`](https://huggingface.co/ntsrigaud/maestro-lstm) — Macro-F1 **0.96** |
| Model loading            | Hugging Face Hub (cached, no training required)                                               |
| Gaze activation          | `HeadPoseGazeDetector` supported; can be disabled via `inference.yaml`                        |
| Test suite               | Unit / integration / system coverage under `tests/`                                           |
| Tracked binary weights   | **0 MB** — all `.pt` artifacts excluded from VCS                                              |

The complete development history is recorded in [docs/architecture/](docs/architecture/) and [docs/archive/work-logs/](docs/archive/work-logs/).
