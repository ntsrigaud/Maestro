# Complete Environment Setup Guide

**Last Updated:** January 17, 2026  
**Purpose:** Reproducible environment setup from a clean machine  
**Validated:** ✅ Tested on macOS (M3), Linux (Ubuntu 22.04)

---

## Quick Start (One-Command Setup)

```bash
# 1. Create and activate Conda environment
conda env create -f environment.yml
conda activate gesture

# 2. Install the gesture recognition package
pip install -e .

# 3. Verify installation
python scripts/validate_hand_detector.py
```

**Expected result:** All validation tests pass with ✓ marks.

---

## Dependency Rationale

### Why Conda + Pip Hybrid?

This project uses **both** Conda and pip because:

1. **Conda strengths:**
   - Platform-optimized binaries (PyTorch with CUDA/Metal acceleration)
   - Better handling of native C/C++ libraries (OpenCV, NumPy)
   - Consistent environment across platforms

2. **Pip necessity:**
   - MediaPipe 0.10.x unavailable/broken in Conda
   - Latest sentence-transformers requires pip
   - Hugging Face ecosystem not well-maintained in Conda

### Package Classification

| Package | Install Method | Justification |
|---------|---------------|---------------|
| **PyTorch** | Conda | Platform binaries (CUDA/Metal), faster installation |
| **OpenCV** | Conda | System library integration, better compatibility |
| **NumPy** | Conda | Optimized BLAS/LAPACK linking |
| **scikit-learn** | Conda | SciPy stack dependencies |
| **MediaPipe** | Pip | Conda version outdated (0.9.x), requires 0.10.21+ |
| **sentence-transformers** | Pip | Not in Conda, actively maintained on PyPI |
| **huggingface-hub** | Pip | Official distribution only on PyPI |

---

## Complete Dependency List

### Core Runtime (Required)

These are installed automatically with `pip install -e .`:

- **opencv-python** (4.8.0+) - `import cv2`
- **numpy** (1.24.0+) - `import numpy`
- **mediapipe** (0.10.21+) - Hand detection API
- **torch** (2.0.0+) - Deep learning framework
- **torchvision** (0.15.0+) - Vision transforms
- **timm** (0.9.0+) - ViT-B/16 model
- **pillow** (9.0.0+) - Image I/O

### Training & Notebooks (Optional)

Included in `environment.yml` for full development:

- **scikit-learn** (1.2.0+) - Metrics, confusion matrices
- **matplotlib** (3.5.0+) - Plotting
- **seaborn** (0.12.0+) - Statistical visualization
- **pandas** (1.5.0+) - Data manipulation
- **tqdm** (4.65.0+) - Progress bars
- **jupyter** (1.0.0+) - Notebook environment
- **ipykernel** (6.20.0+) - Jupyter kernel

### Research Features (Optional)

Install with `pip install -e ".[semantic,huggingface,datasets]"`:

- **sentence-transformers** (2.2.0+) - Semantic gesture learning
- **huggingface-hub** (0.16.0+) - Model hosting
- **kagglehub** (0.1.0+) - Dataset downloads
- **gdown** (4.7.0+) - Google Drive downloads

---

## Detailed Setup Instructions

### Prerequisites

- **Conda/Miniconda** installed ([download](https://docs.conda.io/en/latest/miniconda.html))
- **Python 3.10** (specified in environment.yml)
- **10 GB disk space** (for packages + models)

### Step-by-Step Setup

#### 1. Clone Repository

```bash
git clone <repository-url>
cd CSU-Capstone-Project
```

#### 2. Create Conda Environment

```bash
# Create environment from specification
conda env create -f environment.yml

# This installs:
# - Python 3.10
# - PyTorch (with Metal/CUDA if available)
# - OpenCV, NumPy, scikit-learn
# - Jupyter, matplotlib, seaborn
# - MediaPipe 0.10.21+ (via pip)
```

**Expected output:**
```
Collecting package metadata (repodata.json): done
Solving environment: done
...
Installing pip dependencies: done
```

#### 3. Activate Environment

```bash
conda activate gesture
```

#### 4. Install Gesture Recognition Package

```bash
# Install in editable mode (links to src/ directory)
pip install -e .
```

**This installs the package so you can use:**
```python
from input_manager import InputManager
from recognition_engine.hand_detection import HandDetector, HandCropper
from shared.logger import get_logger
```

#### 5. Verify Installation

```bash
# Test MediaPipe integration
python scripts/validate_hand_detector.py
```

**Expected output:**
```
✓ HandDetector instantiation successful
✓ API compatibility validated
✓ Detection output validation passed
✓ VIDEO mode works
✓ IMAGE mode works
🎉 ALL VALIDATIONS PASSED
```

---

## Validation Test Suite

### Test 1: Import Test

```bash
python -c "
import cv2
import numpy as np
import torch
import timm
import mediapipe as mp
from input_manager import InputManager
from recognition_engine.hand_detection import HandDetector, HandCropper
print('✓ All imports successful')
"
```

### Test 2: Hand Detection Test

```bash
python scripts/validate_hand_detector.py
```

Validates:
- MediaPipe 0.10.x API compatibility
- Hand detection pipeline
- Video/Image mode switching

### Test 3: Gesture Recognition Demo

```bash
# Requires webcam
python scripts/demo_gesture_recognition.py
```

Expected: Real-time gesture recognition with bounding boxes

### Test 4: Model Inference Test

```bash
python scripts/verify_clean_baseline.py
```

Expected: ViT model loads and performs inference

---

## Troubleshooting

### Issue: PyTorch symbol error (`iJIT_NotifyEvent`)

**Symptom:**
```
ImportError: undefined symbol: iJIT_NotifyEvent
```

**Solution:**
```bash
pip uninstall -y torch torchvision torchaudio
conda install pytorch torchvision torchaudio -c pytorch
```

### Issue: MediaPipe import error

**Symptom:**
```
ImportError: cannot import name 'solutions' from 'mediapipe'
```

**Solution:**
Ensure MediaPipe 0.10.21+:
```bash
pip install --upgrade mediapipe>=0.10.21
```

### Issue: OpenCV not found (`import cv2` fails)

**Symptom:**
```
ModuleNotFoundError: No module named 'cv2'
```

**Solution:**
```bash
# Option 1: Conda (preferred)
conda install -c conda-forge opencv

# Option 2: Pip (fallback)
pip install opencv-python>=4.8.0
```

### Issue: CUDA not available (NVIDIA GPUs)

**Symptom:**
```python
>>> torch.cuda.is_available()
False
```

**Solution:**
Reinstall PyTorch with CUDA:
```bash
conda install pytorch torchvision torchaudio pytorch-cuda=11.8 -c pytorch -c nvidia
```

Check compatibility: https://pytorch.org/get-started/locally/

---

## Platform-Specific Notes

### macOS (Apple Silicon M1/M2/M3)

- **Metal acceleration**: Enabled automatically via `device = torch.device('mps')`
- **Rosetta 2**: Not required for this environment
- **Known issue**: Some conda packages may be x86_64; use `CONDA_SUBDIR=osx-arm64`

### Linux (Ubuntu/Debian)

- **System dependencies**: Conda handles most, but may need:
  ```bash
  sudo apt-get install libgl1-mesa-glx libglib2.0-0
  ```
- **CUDA**: Install NVIDIA drivers first, then use `pytorch-cuda` channel

### Windows

- **Long path support**: Enable before cloning repository
- **Visual C++ Redistributable**: Required for PyTorch (auto-installed)
- **Git Bash recommended**: For Unix-like shell experience

---

## Environment File Maintenance

### When to update environment.yml

1. **New dependency added** to any Python file/notebook
2. **Version compatibility issue** discovered
3. **Platform-specific requirement** identified

### How to export current environment

```bash
# Export full environment (for debugging)
conda env export > environment-full.yml

# Export explicit dependencies only (preferred)
conda env export --from-history > environment.yml
```

### How to update existing environment

```bash
# Update from modified environment.yml
conda env update -f environment.yml --prune

# Or recreate from scratch
conda env remove -n gesture
conda env create -f environment.yml
```

---

## Clean Room Validation Checklist

Use this to verify a fresh setup works:

- [ ] Conda installed and in PATH
- [ ] `conda env create -f environment.yml` completes without errors
- [ ] `conda activate gesture` succeeds
- [ ] `pip install -e .` completes
- [ ] `python scripts/validate_hand_detector.py` passes all tests
- [ ] `python -c "import cv2, torch, mediapipe"` succeeds
- [ ] `python scripts/demo_gesture_recognition.py` runs (with webcam)
- [ ] No manual `pip install` commands were needed

---

## Dependency Version Pinning Strategy

- **Minimum versions** (>=): Core libraries (PyTorch, NumPy)
- **Exact versions** (==): Only when known compatibility issues exist
- **No pins**: Standard library, utilities (tqdm, argparse)

This balances reproducibility with flexibility for security updates.

---

## Alternative: Docker Setup (Future)

For maximum reproducibility, consider:

```dockerfile
FROM continuumio/miniconda3
COPY environment.yml .
RUN conda env create -f environment.yml
RUN echo "conda activate gesture" >> ~/.bashrc
COPY . /app
WORKDIR /app
RUN pip install -e .
```

Not currently implemented; manual conda setup is sufficient.

---

## Support

If you encounter issues not covered here:

1. Check `environment.yml` matches this documentation
2. Verify Python version: `python --version` should be 3.10.x
3. Check MediaPipe version: `pip show mediapipe` should be 0.10.21+
4. Review [HAND_DETECTOR_REFACTORING.md](HAND_DETECTOR_REFACTORING.md) for MediaPipe API changes

---

**Summary:** This environment specification has been tested and validated to work from a clean machine with zero manual installations. If you encounter `ModuleNotFoundError`, the environment file is incomplete and should be updated.
