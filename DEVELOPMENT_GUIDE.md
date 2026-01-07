# Development Guide

**Last Updated**: January 7, 2026

This guide provides everything you need to understand, set up, and contribute to the Hand Gesture Recognition System project.

---

## Table of Contents

1. [Project Overview](#project-overview)
2. [Quick Start](#quick-start)
3. [Architecture](#architecture)
4. [Development Workflow](#development-workflow)
5. [Training Models](#training-models)
6. [Testing & Validation](#testing--validation)
7. [Common Tasks](#common-tasks)
8. [Troubleshooting](#troubleshooting)

---

## Project Overview

### What We're Building

A **semantic gesture recognition system** that understands the *meaning* of gestures, not just their visual patterns. The system uses:
- **Vision Transformer (ViT)** for visual understanding
- **LSTM** for temporal modeling
- **BERT/SentenceTransformer** for semantic alignment
- **Zero-shot learning** capabilities via shared embedding spaces

### Core Innovation

Traditional approach:
```
Gesture → Classifier → Label (e.g., "thumbs_up") → Hard-coded action
```

Our approach:
```
Gesture → Visual Embedding → Semantic Space → Text Command → Action
```

This enables:
- ✅ Zero-shot recognition of new gesture variations
- ✅ Extensibility via text descriptions (no retraining)
- ✅ Semantic understanding of user intent
- ✅ Natural language command mapping

### Implementation Status

| Component | Status | Location |
|-----------|--------|----------|
| **Input Manager** | ✅ Complete | `src/input_manager/` |
| **Hand Detection** | ✅ Complete | `src/recognition_engine/hand_detection/` |
| **Gesture Classification** | 🔬 Research | `notebooks/finetune_vit_hagrid.ipynb` |
| **Temporal Modeling** | 🔬 Experimental | `src/recognition_engine/temporal/` |
| **Semantic Understanding** | 🔬 PoC | `notebooks/semantic_gesture_learning.ipynb` |
| **Action Mapper** | ❌ Not started | - |
| **OS Controller** | ❌ Not started | - |

---

## Quick Start

### 1. Environment Setup

**Option A: Conda (Recommended)**
```bash
# Clone and navigate to repository
cd /path/to/CSU-Capstone-Project

# Create environment
conda env create -f environment.yml
conda activate gesture

# Install project
pip install -e .
```

**Option B: Manual Setup**
```bash
# Create environment
conda create -n gesture python=3.10 -y
conda activate gesture

# Install PyTorch
conda install pytorch torchvision -c pytorch -y

# Install dependencies
conda install -c conda-forge timm pillow matplotlib seaborn pandas tqdm scikit-learn jupyter -y
pip install transformers datasets accelerate sentence-transformers mediapipe opencv-python

# Install project
pip install -e .
```

### 2. Verify Installation

```bash
python scripts/verify_install.py
```

Expected output: All checks pass ✓

### 3. Download Dataset (Optional, for training)

**Recommended method** (fast, 5-15 minutes):
```bash
pip install kaggle
python scripts/download_hagrid_kaggle.py --save_path ./data/hagrid_subset
```

This downloads ~1.5 GB with 30K images across 18 gesture classes.

**Verify dataset**:
```bash
python -c "import pandas as pd; df = pd.read_csv('data/hagrid_subset/annotations.csv'); print(df['label'].value_counts())"
```

### 4. Test Hand Detection

```bash
python scripts/demo_hand_detection.py
```

This opens a window showing real-time hand detection with landmarks and bounding boxes.

---

## Architecture

### System Pipeline

```
┌──────────┐    ┌──────────────┐    ┌──────────────────┐    ┌────────────┐    ┌──────────┐
│  Camera  │───▶│Input Manager │───▶│Recognition Engine│───▶│Action Mapper│───▶│OS Control│
└──────────┘    └──────────────┘    └──────────────────┘    └────────────┘    └──────────┘
                                            │
                                            ├── Hand Detection (MediaPipe)
                                            ├── Gesture Classification (ViT)
                                            ├── Temporal Modeling (LSTM)
                                            └── Semantic Understanding (BERT)
```

### Detailed Documentation

- **Architecture Overview**: [`docs/Project-Architecture/README.md`](docs/Project-Architecture/README.md)
- **Detailed System Design**: [`docs/Project-Architecture/DETAILED_SYSTEM_ARCHITECTURE.md`](docs/Project-Architecture/DETAILED_SYSTEM_ARCHITECTURE.md)
- **Input Manager**: [`docs/Project-Architecture/Input-Manager/README.md`](docs/Project-Architecture/Input-Manager/README.md)
- **Recognition Engine**: [`docs/Project-Architecture/Recognition-Engine/README.md`](docs/Project-Architecture/Recognition-Engine/README.md)

---

## Development Workflow

### Code Organization

```
src/
├── input_manager/          # Camera capture and frame management
│   ├── input_manager.py    # Main InputManager class
│   └── example_usage.py    # Usage examples
├── recognition_engine/     # Hand detection and gesture recognition
│   ├── hand_detection/     # MediaPipe hand detection
│   │   ├── hand_detector.py
│   │   ├── hand_cropper.py
│   │   └── hand_data.py
│   └── temporal/           # Temporal modeling (LSTM, Mamba)
│       └── temporal_heads.py
└── shared/                 # Utilities
    └── logger.py           # Logging system
```

### Import Best Practices

After `pip install -e .`, import modules directly:

```python
# ✅ Correct
from input_manager import InputManager, FrameData
from recognition_engine.hand_detection import HandDetector, HandCropper
from shared.logger import get_logger

# ❌ Don't do this
import sys
sys.path.insert(0, ...)  # No longer needed!
```

### Logging

The project uses a centralized logging system with automatic rotation:

```python
from shared.logger import get_logger

logger = get_logger(__name__)
logger.info("Processing frame")
logger.debug("Detailed debugging info")
logger.error("Something went wrong", exc_info=True)
```

Logs are written to `logs/application.log` with automatic rotation at 10MB.

See [`LOG_ROTATION_STRATEGY.md`](LOG_ROTATION_STRATEGY.md) for details.

---

## Training Models

### ViT Fine-tuning for Gesture Recognition

**Training notebook**: [`notebooks/finetune_vit_hagrid.ipynb`](notebooks/finetune_vit_hagrid.ipynb)

**Quick reference**:
```bash
# 1. Activate environment
conda activate gesture

# 2. Launch Jupyter
jupyter notebook notebooks/finetune_vit_hagrid.ipynb

# 3. Run all cells or customize hyperparameters
```

**Key hyperparameters** (Cell 9 in notebook):
```python
LEARNING_RATE = 3e-5      # Try: 1e-5, 5e-5, 1e-4
EPOCHS = 10               # Try: 15, 20, 30
BATCH_SIZE = 32           # Reduce if OOM: 16, 8
```

**Expected results** (10 epochs):
- Test Accuracy: 85-90%
- Training time: 30-60 minutes on GPU, 2-3 hours on CPU
- Model size: ~330 MB

**Output**: Model checkpoint saved to `checkpoints/vit_hagrid_best.pt`

### Semantic Gesture Learning

**Training notebook**: [`notebooks/semantic_gesture_learning.ipynb`](notebooks/semantic_gesture_learning.ipynb)

This implements contrastive learning to align visual and text embeddings in a shared semantic space.

**Key research documents**:
- [`notebooks/SEMANTIC_GESTURE_RESEARCH.md`](notebooks/SEMANTIC_GESTURE_RESEARCH.md) - Approach overview
- [`notebooks/SEMANTIC_IMPROVEMENT_GUIDE.md`](notebooks/SEMANTIC_IMPROVEMENT_GUIDE.md) - Practical tips
- [`notebooks/GESTURE_SEMANTICS_ENHANCEMENT.md`](notebooks/GESTURE_SEMANTICS_ENHANCEMENT.md) - Semantic descriptions

### Temporal Modeling (LSTM)

**Training notebook**: [`notebooks/vit_lstm_gesture_training.ipynb`](notebooks/vit_lstm_gesture_training.ipynb)

Adds temporal understanding for dynamic gestures (swipes, waves, etc.).

---

## Testing & Validation

### Unit Tests

Test core functionality:
```bash
python scripts/test_hand_detection.py    # Hand detection tests
python scripts/validate_hand_detection.py # Validation suite
```

### Demo Scripts

Visual/interactive testing:
```bash
# Hand detection with visualization
python scripts/demo_hand_detection.py

# Full gesture recognition pipeline
python scripts/demo_gesture_recognition.py

# Semantic gesture understanding
python scripts/demo_semantic_gesture.py
```

Press `q` to quit demo windows.

### Verify Installation

After setup or making changes:
```bash
python scripts/verify_install.py
```

---

## Common Tasks

### Add a New Module

1. Create directory: `src/new_module/`
2. Add `__init__.py`: `src/new_module/__init__.py`
3. Reinstall: `pip install -e .` (auto-discovered by setuptools)
4. Import: `from new_module import MyClass`

### Capture Hand Samples

Collect training data:
```bash
python scripts/capture_hand_samples.py --output_dir ./samples --num_samples 100
```

### Organize Dataset

If you have HaGRID data in non-standard format:
```bash
python scripts/organize_hagrid_dataset.py \
    --input_dir ./raw_data \
    --output_dir ./data/hagrid_organized
```

See [`scripts/ORGANIZE_DATASET_README.md`](scripts/ORGANIZE_DATASET_README.md) for details.

### Check Disk Usage

```bash
# Dataset size
du -sh data/hagrid_subset

# Model checkpoints
du -sh checkpoints

# Logs
du -sh logs
```

### Clean Python Cache

If imports are misbehaving:
```bash
find . -type d -name __pycache__ -exec rm -rf {} + 2>/dev/null
pip install -e . --force-reinstall --no-deps
```

---

## Troubleshooting

### "Module not found" errors

**Solution**: Reinstall the project
```bash
pip install -e .
```

### MediaPipe installation issues

**macOS**:
```bash
pip install mediapipe --no-cache-dir
```

**Linux**:
```bash
pip install mediapipe opencv-python-headless
```

### CUDA/GPU not detected

**Check PyTorch installation**:
```python
import torch
print(torch.cuda.is_available())  # Should be True if GPU available
```

**Reinstall with CUDA**:
```bash
conda install pytorch torchvision torchaudio pytorch-cuda=11.8 -c pytorch -c nvidia
```

### Camera not accessible

**macOS**: Grant terminal camera permissions in System Preferences → Security & Privacy

**Linux**: Check device permissions
```bash
ls -l /dev/video*
sudo usermod -a -G video $USER  # Add user to video group
```

### Jupyter kernel crashes

**Reduce batch size** in training notebook:
```python
BATCH_SIZE = 16  # or 8
```

### Out of memory during training

**Solutions**:
1. Reduce batch size: `BATCH_SIZE = 16` or `8`
2. Use smaller model: `MODEL_NAME = 'vit_small_patch16_224'`
3. Reduce image size: `transforms.Resize(192)` instead of `224`
4. Use CPU: `device = 'cpu'` (slower but won't crash)

### Dataset download is slow

**Use the Kaggle method** instead of downloading full ZIPs:
```bash
python scripts/download_hagrid_kaggle.py --save_path ./data/hagrid_subset
```

See [`docs/HAGRID_DOWNLOAD_SOLUTIONS.md`](docs/HAGRID_DOWNLOAD_SOLUTIONS.md) for comparison.

---

## Additional Resources

### Documentation

- **Main README**: [`README.md`](README.md) - Project overview
- **Installation**: [`INSTALL.md`](INSTALL.md) - Detailed setup instructions
- **Training**: [`TRAINING_README.md`](TRAINING_README.md) - Model training guide
- **Package Management**: [`PACKAGE_MANAGEMENT.md`](PACKAGE_MANAGEMENT.md) - Import system guide

### Research References

- **HaGRID Dataset**: [`docs/HaGRID_DATASET_ASSESSMENT.md`](docs/HaGRID_DATASET_ASSESSMENT.md)
- **HAVIT Model**: [`docs/Topics/HAVIT_Gesture_Recognition.md`](docs/Topics/HAVIT_Gesture_Recognition.md)
- **Vision Transformers**: [`docs/Topics/ViT_and_self-attention.md`](docs/Topics/ViT_and_self-attention.md)
- **Contribution Explanation**: [`docs/Topics/contribution_explanation.md`](docs/Topics/contribution_explanation.md)

### Scripts Documentation

- **Scripts Overview**: [`scripts/README.md`](scripts/README.md)
- **Demo Guide**: [`scripts/GESTURE_RECOGNITION_DEMO.md`](scripts/GESTURE_RECOGNITION_DEMO.md)

---

## Getting Help

1. **Check existing documentation** in `docs/` and this guide
2. **Review error messages** carefully - they often point to the solution
3. **Check git history** for context: `git log --oneline --graph`
4. **Archived development logs** in `docs/archive/development-logs/` show how issues were resolved

---

## Contributing

### Before Making Changes

1. Create a feature branch: `git checkout -b feature-name`
2. Make sure tests pass: `python scripts/verify_install.py`
3. Keep commits focused and well-documented

### Code Style

- Follow PEP 8 conventions
- Use type hints where appropriate
- Add docstrings to classes and functions
- Keep functions focused and testable

### Documentation

- Update relevant documentation when changing functionality
- Add examples for new features
- Keep README files current with code changes

---

**Happy coding! 🚀**
