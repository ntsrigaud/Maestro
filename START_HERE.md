# 🎯 Quick Orientation - Start Here

**Welcome back to the Hand Gesture Recognition System project!**

This document provides a 5-minute orientation to get you up to speed quickly.

---

## What This Project Is

A **semantic gesture recognition system** that understands gestures through vision-language alignment, enabling zero-shot recognition and natural command mapping. Think "Siri for hand gestures" - it learns what gestures *mean*, not just what they look like.

**Core tech**: Vision Transformer (ViT) + LSTM + BERT/SentenceTransformer

---

## Current State (January 2026)

### ✅ Working and Production-Ready
- Camera capture and frame management
- MediaPipe-based hand detection (21 landmarks)
- ViT gesture classifier (85-90% accuracy on HaGRID)

### 🔬 Active Research/Development
- Temporal modeling (LSTM for dynamic gestures)
- Semantic understanding (vision-language alignment)
- Zero-shot learning capabilities

### 📋 Planned but Not Started
- Action Mapper (gesture → command translation)
- OS Controller (system command execution)

---

## Essential Documentation

Read in this order:

1. **[README.md](README.md)** - Project overview (5 min read)
2. **[DEVELOPMENT_GUIDE.md](DEVELOPMENT_GUIDE.md)** - Complete workflow guide (15 min read)
3. **[docs/Project-Architecture/README.md](docs/Project-Architecture/README.md)** - System architecture (10 min read)

**For deep dive**: [REPOSITORY_ANALYSIS.md](REPOSITORY_ANALYSIS.md) - Comprehensive analysis of what exists and why

---

## Quick Start (15 Minutes)

### 1. Setup Environment
```bash
conda env create -f environment.yml
conda activate gesture
pip install -e .
python scripts/verify_install.py  # Should pass all checks
```

### 2. Try Hand Detection
```bash
python scripts/demo_hand_detection.py  # Press 'q' to quit
```

You should see real-time hand detection with landmarks overlaid on your video feed.

### 3. Explore Training (Optional)
```bash
jupyter notebook notebooks/finetune_vit_hagrid.ipynb
```

Review the notebook to understand the training pipeline. Running it requires downloading the dataset first (~1.5 GB).

---

## Repository Structure (Simplified)

```
CSU-Capstone-Project/
├── README.md                      # Start here
├── DEVELOPMENT_GUIDE.md           # Complete dev guide
├── INSTALL.md                     # Setup details
├── TRAINING_README.md             # Training workflow
│
├── src/                           # Core implementation
│   ├── input_manager/             # ✅ Camera capture
│   ├── recognition_engine/        # 🔬 Detection & recognition
│   └── shared/                    # ✅ Utilities
│
├── notebooks/                     # Training & research
│   ├── finetune_vit_hagrid.ipynb  # Main training
│   └── semantic_gesture_learning.ipynb # Semantic approach
│
├── scripts/                       # Demos & utilities
│   ├── demo_hand_detection.py     # Hand detection demo
│   └── download_hagrid_kaggle.py  # Dataset downloader
│
└── docs/                          # Documentation
    ├── Project-Architecture/      # System design
    ├── Topics/                    # Research references
    └── archive/                   # Historical materials
```

---

## Common Tasks

### Run Demos
```bash
python scripts/demo_hand_detection.py         # Hand detection
python scripts/demo_gesture_recognition.py    # Full pipeline
python scripts/demo_semantic_gesture.py       # Semantic understanding
```

### Download Dataset
```bash
pip install kaggle
python scripts/download_hagrid_kaggle.py --save_path ./data/hagrid_subset
```

### Train Model
```bash
jupyter notebook notebooks/finetune_vit_hagrid.ipynb
# Run all cells (30-60 min on GPU)
```

### Check Installation
```bash
python scripts/verify_install.py
```

---

## Key Files to Know

### Code
- [src/input_manager/input_manager.py](src/input_manager/input_manager.py) - Camera capture
- [src/recognition_engine/hand_detection/hand_detector.py](src/recognition_engine/hand_detection/hand_detector.py) - Hand detection
- [notebooks/finetune_vit_hagrid.ipynb](notebooks/finetune_vit_hagrid.ipynb) - Training pipeline

### Documentation
- [docs/Project-Architecture/DETAILED_SYSTEM_ARCHITECTURE.md](docs/Project-Architecture/DETAILED_SYSTEM_ARCHITECTURE.md) - Complete architecture
- [docs/Topics/HAVIT_Gesture_Recognition.md](docs/Topics/HAVIT_Gesture_Recognition.md) - Core research
- [notebooks/SEMANTIC_GESTURE_RESEARCH.md](notebooks/SEMANTIC_GESTURE_RESEARCH.md) - Semantic approach

---

## What Changed Recently?

The repository underwent a **deliberate cleanup** on January 7, 2026:
- ✅ Removed obsolete documentation (past change summaries, bug fixes)
- ✅ Archived historical materials (meeting notes, dev logs)
- ✅ Created comprehensive development guide
- ✅ Updated README to reflect actual state
- ✅ No code was deleted - all implementation preserved

**See [CLEANUP_REPORT.md](CLEANUP_REPORT.md)** for complete details.

---

## Need Help?

1. **Setup issues**: Check [INSTALL.md](INSTALL.md) and [DEVELOPMENT_GUIDE.md](DEVELOPMENT_GUIDE.md) troubleshooting sections
2. **Architecture questions**: See [docs/Project-Architecture/](docs/Project-Architecture/)
3. **Training issues**: Read [TRAINING_README.md](TRAINING_README.md)
4. **Import errors**: Run `pip install -e .` from project root

---

## Team

- Sley Hortes
- Neil Taison Rigaud
- Khishig Tuguldur
- Eder Guerrero

---

**You're ready to go! Start with the demos, then dive into the code.** 🚀
