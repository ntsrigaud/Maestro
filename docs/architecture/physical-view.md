# Physical View

## 1. Overview

The physical view describes the system's deployment topology, hardware dependencies, runtime environment, and communication between physical components. The system is designed as a single-node desktop application that runs entirely on the user's local machine with no network dependencies.

## 2. Deployment Topology

```mermaid
graph TB
    subgraph Host ["User Workstation"]
        subgraph HW ["Hardware"]
            Webcam["USB/Built-in Webcam<br/>1280×720 @ 30 FPS"]
            CPU["CPU<br/>(x86_64 or ARM64)"]
            Display["Display Monitor"]
            Keyboard["OS Keyboard Bus"]
        end

        subgraph SW ["Software Environment"]
            Python["Python 3.11 Runtime"]
            subgraph App ["Gesture Control Application"]
                Capture["Frame Capture<br/>(OpenCV)"]
                MP["Landmark Extraction<br/>(MediaPipe)"]
                LSTM["Gesture Classification<br/>(PyTorch LSTM)"]
                Dispatch["Command Dispatch<br/>(pyautogui)"]
                Overlay["Visual Feedback<br/>(OpenCV)"]
            end
            TargetApp["Target Application<br/>(PowerPoint / Zoom)"]
        end
    end

    Webcam -->|USB / V4L2 / AVFoundation| Capture
    Capture --> MP
    MP --> LSTM
    LSTM --> Dispatch
    Dispatch -->|OS Keyboard Events| Keyboard
    Keyboard -->|Key Press| TargetApp
    Overlay -->|Window Frame| Display
```

## 3. Hardware Requirements

### 3.1 Minimum Hardware

| Component   | Requirement      | Notes                                                                                 |
| ----------- | ---------------- | ------------------------------------------------------------------------------------- |
| **Webcam**  | 720p @ 30 FPS    | Built-in laptop cameras are sufficient; USB webcams supported via OpenCV device index |
| **CPU**     | 4-core, 2.0+ GHz | MediaPipe and PyTorch inference are CPU-bound; no GPU required                        |
| **RAM**     | 4 GB available   | MediaPipe model (~50 MB), LSTM model (~1 MB), frame buffers (~10 MB)                  |
| **Display** | Any resolution   | Required for visual feedback overlay; optional with `--no-display` flag               |

### 3.2 Tested Configurations

The system has been developed and tested on macOS with Apple Silicon (ARM64). The architecture is platform-independent at the application level, with platform-specific behavior isolated to:

- **pyautogui**: Uses Quartz API on macOS, X11/Wayland on Linux, Win32 API on Windows.
- **OpenCV VideoCapture**: Uses AVFoundation on macOS, V4L2 on Linux, DirectShow on Windows.

### 3.3 GPU Support

GPU acceleration is **optional and not required**. If a CUDA-capable GPU or Apple MPS device is available, PyTorch will use it for LSTM inference and training when `--device auto` is specified. The default configuration targets CPU execution.

## 4. Runtime Environment

### 4.1 Python Environment

| Component       | Version      | Constraint                                                 |
| --------------- | ------------ | ---------------------------------------------------------- |
| Python          | 3.11         | `>=3.11,<3.12` (locked to match MediaPipe compatibility)   |
| Package Manager | pip          | Installed via `pip install -e .` from `pyproject.toml`     |
| Environment     | Conda / venv | Recommended: `conda create -n gesture-control python=3.11` |

### 4.2 Runtime Dependencies

| Library                | Version   | Role                                                               |
| ---------------------- | --------- | ------------------------------------------------------------------ |
| mediapipe              | ==0.10.14 | Hand landmark extraction (pinned for reproducibility)              |
| torch                  | ≥2.7.0    | LSTM inference and training                                        |
| opencv-python-headless | ≥4.10.0   | Webcam capture and image rendering                                 |
| pyautogui              | ≥0.9.54   | OS keyboard event injection                                        |
| pydantic               | ≥2.6.0    | Configuration schema validation                                    |
| numpy                  | ≥1.26.0   | Array operations                                                   |
| PyYAML                 | ≥6.0.1    | Configuration file parsing                                         |
| huggingface_hub        | ≥0.22.0   | Model download from HF Hub (`[huggingface]` optional extra; required for the default `backend: huggingface` in `model_source.yaml`) |
| scikit-learn           | ≥1.5.0    | Dataset splitting and training metrics (`[training]` optional extra; not required at inference runtime) |

### 4.3 OS Permissions

The system requires the following OS-level permissions:

| Permission                       | Platform              | Purpose                                   |
| -------------------------------- | --------------------- | ----------------------------------------- |
| Camera access                    | macOS, Linux, Windows | Webcam frame capture via OpenCV           |
| Accessibility / Input Monitoring | macOS                 | pyautogui keyboard event injection        |
| No special permissions           | Linux (X11)           | X11 allows keyboard simulation by default |

## 5. Communication Interfaces

### 5.1 Webcam → Application

- **Protocol**: OS-native video capture API (AVFoundation / V4L2 / DirectShow)
- **Mediated by**: OpenCV `VideoCapture`
- **Data**: BGR 8-bit frames at configured resolution and FPS
- **Buffering**: OpenCV buffer size set to 1 to minimize capture latency

### 5.2 Application → Target Software

- **Protocol**: OS keyboard event injection
- **Mediated by**: pyautogui → Quartz / X11 / Win32
- **Data**: Key press/release events (e.g., Right Arrow, Alt+Y)
- **Safety**: Key names validated against hardcoded allowlist before injection

### 5.3 Application → Display

- **Protocol**: OpenCV `imshow()` window
- **Data**: Annotated BGR frame with diagnostic overlay
- **Optional**: Can be disabled with `--no-display` for headless operation

## 6. File System Layout

The system reads from and writes to the local file system:

| Path                         | Access     | Purpose                                                                   |
| ---------------------------- | ---------- | ------------------------------------------------------------------------- |
| `configs/`                   | Read       | YAML configuration files loaded at startup                                |
| `models/checkpoints/`        | Write      | Training checkpoints saved during model training (.gitignored)            |
| `data/raw/`                  | Read/Write | JSONL landmark recordings from data collection                            |
| `data/processed/`            | Read/Write | NPZ sequence datasets built from raw data                                 |
| `experiments/runs/`          | Write      | Training run logs and metrics                                             |
| `~/.cache/huggingface/hub/`  | Read/Write | Hugging Face Hub download cache (first-use download of production model)  |

Trained inference artifacts are **not tracked in VCS**. The production model (`lstm_v1_20260418T023120Z_inference.pt`) is hosted on the Hugging Face Hub (`ntsrigaud/gesture-control-lstm`) and downloaded on first use by `hf_hub_download()`. The backend is controlled by `configs/model_source.yaml`.

## 7. Deployment Procedure

### 7.1 Installation

```bash
# Clone repository
git clone <repository-url>
cd Semantic-HGR-System

# Create and activate environment
conda create -n gesture-control python=3.11
conda activate gesture-control

# Install package with dependencies
# Include [huggingface] extra for the default Hugging Face Hub model backend
pip install -e ".[huggingface]"
```

### 7.2 Inference Execution

```bash
# Run real-time gesture recognition (default: model downloaded from Hugging Face Hub)
python scripts/run_realtime.py --config configs

# Override HF revision
python scripts/run_realtime.py --config configs --hf-revision main

# Use an explicit local artifact instead of HF backend
python scripts/run_realtime.py --config configs --model-artifact path/to/artifact.pt
```

### 7.3 Training Execution

```bash
# Train LSTM model
python scripts/train_model.py \
    --dataset data/processed/gesture_dataset_v2.npz \
    --config configs/model/lstm_default.yaml \
    --config-dir configs
```

## 8. Scalability Considerations

The current system is designed for single-user, single-camera operation. The architecture supports the following extension points without structural changes:

| Extension                      | Mechanism                                                                   |
| ------------------------------ | --------------------------------------------------------------------------- |
| Additional target applications | Implement `IApplicationAdapter` with application-specific shortcut mappings |
| Alternative cameras            | Change `camera_index` in `app.yaml` or implement a custom `IFrameSource`    |
| GPU inference                  | Set `--device cuda` or `--device mps` for PyTorch acceleration              |
| Headless deployment            | Use `--no-display` flag to skip OpenCV window rendering                     |
| Alternative models             | Implement `IGestureModel` with a different classifier (Transformer, TCN)    |

---

**Document**: Physical View  
**Suite**: Architecture Documentation (`docs/architecture/`)  
**Last Updated**: April 2026
