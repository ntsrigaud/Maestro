# Installation Guide

**Updated:** January 17, 2026  
**Status:** Complete environment setup with full dependency management

---

## Quick Start (One Command)

```bash
# 1. Create conda environment (includes all dependencies)
conda env create -f environment.yml

# 2. Activate environment
conda activate gesture

# 3. Install project package
pip install -e .

# 4. Verify installation
python scripts/verify_environment.py
```

**Expected output:** `✓ ALL TESTS PASSED (13/13)`

---

## Prerequisites

- **Conda or Miniconda** installed ([download](https://docs.conda.io/en/latest/miniconda.html))
- **10 GB disk space** (for packages + models)
- **Python 3.10** (automatically installed by environment.yml)

---

## Detailed Steps

### 1. Clone Repository

```bash
git clone <repository-url>
cd CSU-Capstone-Project
```

### 2. Create Environment

```bash
conda env create -f environment.yml
```

This installs:
- PyTorch (with GPU acceleration if available)
- OpenCV, MediaPipe, NumPy
- Training tools (scikit-learn, matplotlib, seaborn)
- Jupyter notebooks

### 3. Activate Environment

```bash
conda activate gesture
```

### 4. Install Package

```bash
pip install -e .
```

The `-e` (editable) flag:
- Links to source code instead of copying
- Changes are immediately available
- No reinstall needed after edits

---

## Verify Installation

### Automated Test

```bash
python scripts/verify_environment.py
```

Checks:
- All 19 dependencies installed
- Correct versions
- MediaPipe 0.10.21+ API
- Project imports work

### Manual Test

```python
import cv2
import numpy as np
import torch
import mediapipe as mp
from input_manager import InputManager
from recognition_engine.hand_detection import HandDetector, HandCropper
print("✓ All imports successful")
```

---

## What Gets Installed

### Core Runtime (Always)
- opencv-python 4.8.0+
- numpy 1.24.0+
- mediapipe 0.10.21+
- torch 2.0.0+
- torchvision 0.15.0+
- timm 0.9.0+
- pillow 9.0.0+

### Training/Development (Conda environment)
- scikit-learn, pandas, matplotlib, seaborn
- tqdm, jupyter, ipykernel

### Optional Research
```bash
# Semantic gesture learning
pip install -e ".[semantic]"

# Model hosting
pip install -e ".[huggingface]"

# Dataset downloads
pip install -e ".[datasets]"

# All extras
pip install -e ".[semantic,huggingface,datasets]"
```

---

## Usage After Installation

### Import from Anywhere

```python
# No sys.path hacks needed!
from input_manager import InputManager
from recognition_engine.hand_detection import HandDetector, HandCropper
from shared.logger import get_logger
```

### Run Scripts

```bash
# Hand detection demo
python scripts/demo_hand_detection.py

# Gesture recognition (requires webcam)
python scripts/demo_gesture_recognition.py

# Test reproducibility experiments
python scripts/test_reproducibility.py --test 2
```

### Run Notebooks

```bash
jupyter notebook notebooks/finetune_vit_hagrid.ipynb
```

---

## Troubleshooting

### Issue: ImportError for cv2

**Solution:**
```bash
conda install -c conda-forge opencv
```

### Issue: MediaPipe version error

**Symptom:** `cannot import name 'solutions'`

**Solution:**
```bash
pip install --upgrade mediapipe>=0.10.21
```

### Issue: PyTorch symbol error

**Symptom:** `undefined symbol: iJIT_NotifyEvent`

**Solution:**
```bash
pip uninstall -y torch torchvision torchaudio
conda install pytorch torchvision torchaudio -c pytorch
```

### Issue: Environment validation fails

**Solution:**
```bash
# Update environment
conda env update -f environment.yml --prune

# Reinstall package
pip install -e . --force-reinstall
```

**See:** [ENVIRONMENT_SETUP.md](ENVIRONMENT_SETUP.md) for complete troubleshooting guide

---

## Platform-Specific Notes

### macOS (Apple Silicon)
- Metal (MPS) acceleration automatic
- No Rosetta 2 required

### Linux (CUDA)
```bash
# For NVIDIA GPUs
conda install pytorch torchvision pytorch-cuda=11.8 -c pytorch -c nvidia
```

### Windows
- Enable long path support: `git config --system core.longpaths true`
- Visual C++ Redistributable installed automatically

---

## Alternative: Development Only

If you don't need the full training environment:

```bash
# Install only core runtime dependencies
pip install -e .

# This gets you:
# - OpenCV, MediaPipe, PyTorch
# - Hand detection and gesture recognition
# - No jupyter, matplotlib, etc.
```

---

## Uninstall

```bash
# Remove package
pip uninstall gesture-recognition-system

# Remove environment
conda deactivate
conda env remove -n gesture
```

---

## Documentation

- **[ENVIRONMENT_SETUP.md](ENVIRONMENT_SETUP.md)** - Complete setup guide with validation
- **[DEPENDENCY_RATIONALE.md](DEPENDENCY_RATIONALE.md)** - Why each package exists
- **[ENVIRONMENT_AUDIT_COMPLETE.md](ENVIRONMENT_AUDIT_COMPLETE.md)** - Audit summary

---

## Support

If you encounter issues:

1. Run `python scripts/verify_environment.py` to diagnose
2. Check [ENVIRONMENT_SETUP.md](ENVIRONMENT_SETUP.md) troubleshooting section
3. Verify Python version: `python --version` (should be 3.10.x)
4. Check MediaPipe: `pip show mediapipe` (should be 0.10.21+)

---

**This installation method has been validated to work from a clean machine with zero manual package installations.**


### Running Examples

```bash
# From anywhere in the project
python src/input_manager/example_usage.py
```

### Importing in Your Code

```python
# In any Python file in the project
from input_manager import InputManager, FrameData
from shared.logger import get_logger, configure_logging

# Use the modules
manager = InputManager(camera_index=0)
logger = get_logger(__name__)
```

### Running from Different Directories

The editable install means imports work regardless of your current directory:

```bash
# From project root
python -c "from input_manager import InputManager; print('Works!')"

# From any subdirectory
cd docs/
python -c "from input_manager import InputManager; print('Still works!')"

# From src/
cd ../src/
python -c "from input_manager import InputManager; print('Works everywhere!')"
```

## Verifying Installation

Check that the package is installed:

```bash
pip list | grep gesture-recognition
```

You should see:

```
gesture-recognition-system  0.1.0  /path/to/CSU-Capstone-Project/src
```

## Uninstalling

If you need to uninstall:

```bash
pip uninstall gesture-recognition-system
```

## Troubleshooting

### Import Errors After Install

If you still get import errors, ensure you're using the correct Python environment:

```bash
# Check your Python version
python --version

# Check which pip you're using
which pip

# If using pyenv
pyenv shell 3.11.9
pip install -e .
```

### Module Not Found

Make sure you ran the install command from the project root (where `setup.py` and `pyproject.toml` are located):

```bash
cd /Users/neil/dev/repos/CSU-Capstone-Project
pip install -e .
```

### Changes Not Reflected

With editable install (`-e`), code changes should be immediate. If not:

1. Restart your Python interpreter
2. Check you're using the correct Python environment
3. Verify with `pip list | grep gesture-recognition`

## Migration from Old Approach

### Before (Fragile)

```python
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))
from shared.logger import get_logger  # Might fail depending on execution context
```

### After (Robust)

```python
from shared.logger import get_logger  # Works everywhere after pip install -e .
```

## Python Naming Conventions

All module directories use underscores (not hyphens) following PEP 8:

- `src/input_manager/` (not `input-manager`)
- `src/shared/`
- Future modules: `src/recognition_engine/`, `src/action_mapper/`, `src/os_controller/`

This allows `find_packages()` to automatically discover all modules without manual configuration.

## Alternative: PYTHONPATH (Not Recommended)

If you cannot use pip install for some reason, you can set PYTHONPATH:

```bash
# Add to your ~/.zshrc or ~/.bashrc
export PYTHONPATH="/Users/neil/dev/repos/CSU-Capstone-Project/src:$PYTHONPATH"
```

**However, this approach:**

- Requires manual configuration on every machine
- Doesn't work well with virtual environments
- Can cause conflicts with other projects
- Is harder to maintain in CI/CD pipelines

**The editable install is superior because:**

- Works with virtual environments
- Properly integrates with Python's packaging system
- Easy to share with team members
- Standard Python practice
- Works in all execution contexts (scripts, REPL, IDEs, tests)
