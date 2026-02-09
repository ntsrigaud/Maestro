# 🎯 Quick Orientation - Start Here

**Welcome to the Hand Gesture Recognition System project!**

This document provides a 5-minute orientation to get you up to speed quickly.

---

## What This Project Is

A **gesture recognition system** using Vision Transformers (ViT-B/16) for robust hand gesture classification. The system is trained on the HaGRID dataset and processes full-frame images to classify 18 different hand gestures.

**Core tech**: Vision Transformer (ViT-B/16) fine-tuned on HaGRID dataset

---

## Current State (January 2026)

### ✅ Completed
- Camera capture and frame management (Input Manager)
- ViT-B/16 gesture classifier trained on HaGRID (85-90% accuracy)
- Complete training pipeline with Hugging Face integration
- Trained model published to Hugging Face Hub
- Logging system with rotation

### 🔬 In Development
- Recognition Engine inference pipeline
- Temporal modeling experiments (LSTM for dynamic gestures)
- End-to-end system integration

### 📋 Planned
- Action Mapper (gesture → command translation)
- OS Controller (system command execution)
- Real-time demo application
- Semantic understanding research (vision-language alignment)

---

## Essential Documentation

Read in this order:

1. **[README.md](README.md)** - Project overview (5 min read)
2. **[DEVELOPMENT_GUIDE.md](DEVELOPMENT_GUIDE.md)** - Development workflow (15 min read)
3. **[docs/Project-Architecture/README.md](docs/Project-Architecture/README.md)** - System architecture (10 min read)
4. **[docs/MODEL_TRAINING_GUIDE.md](docs/MODEL_TRAINING_GUIDE.md)** - Model training details

---

## Quick Start (15 Minutes)

### 1. Setup Environment
```bash
conda env create -f environment.yml
conda activate gesture
pip install -e .
python scripts/verify_install.py  # Should pass all checks
```

### 2. Try Gesture Classification
```bash
# Use pre-trained model from Hugging Face
python -c "
from transformers import AutoModelForImageClassification
model = AutoModelForImageClassification.from_pretrained('neilrigaud/hagrid-vit-gesture')
print('✅ Model loaded successfully!')
"
```

### 3. Train Your Own Model
```bash
jupyter notebook notebooks/finetune_vit_hagrid.ipynb
```

The notebook walks through the complete training pipeline using the HaGRID dataset from Hugging Face. Training takes 30-60 minutes on GPU.

---

## Repository Structure (Simplified)
---

## Repository Structure (Simplified)

```
CSU-Capstone-Project/
├── README.md                      # Start here
├── DEVELOPMENT_GUIDE.md           # Development workflow
├── INSTALL.md                     # Setup details
│
├── src/                           # Core implementation
│   ├── input_manager/             # ✅ Camera capture
│   ├── recognition_engine/        # 🔬 Gesture recognition
│   └── shared/                    # ✅ Logging & utilities
│
├── notebooks/                     # Training & research
│   ├── finetune_vit_hagrid.ipynb  # ✅ ViT training pipeline
│   └── semantic_gesture_learning.ipynb # 🔬 Semantic research
│
├── scripts/                       # Utilities
│   ├── demo_gesture_recognition.py    # Gesture demo
│   ├── download_hagrid_subset.py      # Dataset downloader
│   └── verify_install.py              # Installation check
│
├── checkpoints/                   # Model checkpoints
│   └── best_model.pth            # Trained ViT model
│
└── docs/                          # Documentation
    ├── Project-Architecture/      # System design
    ├── Topics/                    # Research references
    └── archive/                   # Historical materials
```

---

## Common Tasks

### Train ViT Model
```bash
jupyter notebook notebooks/finetune_vit_hagrid.ipynb
# Run all cells (30-60 min on GPU)
# Dataset automatically downloaded from Hugging Face
```

### Use Pre-trained Model
```bash
python scripts/demo_gesture_recognition.py
```

### Check Installation
```bash
python scripts/verify_install.py
```

---

## Key Files to Know

### Code
- [src/input_manager/input_manager.py](src/input_manager/input_manager.py) - Camera capture implementation
- [src/recognition_engine/__init__.py](src/recognition_engine/__init__.py) - Recognition engine module
- [notebooks/finetune_vit_hagrid.ipynb](notebooks/finetune_vit_hagrid.ipynb) - Complete ViT training pipeline

### Documentation
- [docs/Project-Architecture/README.md](docs/Project-Architecture/README.md) - System architecture overview
- [docs/Topics/HAVIT_Gesture_Recognition.md](docs/Topics/HAVIT_Gesture_Recognition.md) - HAVIT research paper
- [docs/MODEL_TRAINING_GUIDE.md](docs/MODEL_TRAINING_GUIDE.md) - Training guide

---

## Need Help?

1. **Setup issues**: Check [INSTALL.md](INSTALL.md) and [DEVELOPMENT_GUIDE.md](DEVELOPMENT_GUIDE.md) troubleshooting sections
2. **Architecture questions**: See [docs/Project-Architecture/](docs/Project-Architecture/)
3. **Training issues**: Read [docs/MODEL_TRAINING_GUIDE.md](docs/MODEL_TRAINING_GUIDE.md)
4. **Import errors**: Run `pip install -e .` from project root

---

## Team

- Sley Hortes
- Neil Taison Rigaud
- Khishig Tuguldur
- Eder Guerrero

---

**You're ready to go! Start with the training notebook and explore the architecture docs.** 🚀
