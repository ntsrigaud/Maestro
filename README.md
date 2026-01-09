# Hand Gesture Recognition System

**Status:** Clean baseline after evaluation collapse  
**Focus:** Restore correctness before experimental features

---

## Overview

Vision-based gesture recognition using ViT-B/16 for human-computer interaction.

After accuracy collapsed from 89% to 11% with single-class dominance, the system was reset to a minimal, correct baseline.

**Current Implementation:**
- Hand detection (MediaPipe)
- ViT-B/16 classification (single linear head)
- No augmentation, no TTA
- Consistent preprocessing across all code paths

**Design:** Correctness > Performance, Interpretability > Cleverness

---

## Quick Start

```bash
# Install dependencies
pip install torch torchvision timm scikit-learn matplotlib seaborn pillow huggingface_hub

# Download dataset: https://www.kaggle.com/datasets/innominate817/hagrid-sample-30k-384p
# Extract to: ./data/hagrid-sample-30k-384p/hagrid_30k/

# Verify setup
python scripts/verify_clean_baseline.py

# Train (30-45 min on GPU)
jupyter notebook notebooks/finetune_vit_hagrid.ipynb

# Upload to Hugging Face (optional)
python scripts/upload_model_to_huggingface.py --repo-id username/repo-name

# Run demo
python scripts/demo_gesture_recognition.py
```

**Expected:** Test accuracy 85-92%, clear confusion matrix diagonal

**Details:** See [CLEAN_BASELINE.md](CLEAN_BASELINE.md)

## Documentation

**Clean Baseline:** [CLEAN_BASELINE.md](CLEAN_BASELINE.md) - Architecture, rationale, quick reference

**Additional:**
- [INSTALL.md](INSTALL.md) - Detailed setup
- [TRAINING_README.md](TRAINING_README.md) - Training details
- [docs/](docs/) - Architecture and research documentation

## Project Structure

```
├── notebooks/
│   └── finetune_vit_hagrid.ipynb                   # Training (clean baseline)
├── scripts/
│   ├── demo_gesture_recognition.py                 # Demo with HF auto-download
│   ├── upload_model_to_huggingface.py             # Upload to HF Hub
│   ├── verify_clean_baseline.py                   # Pre-training verification
│   └── download_hagrid_kaggle.py                  # Dataset downloader
├── src/
│   ├── input_manager/                             # Camera capture
│   ├── recognition_engine/                        # Hand detection
│   └── shared/                                    # Logging, utils
├── docs/                                          # Documentation
└── CLEAN_BASELINE.md                              # Current implementation
```

## Research Foundation

Our approach builds on:
- **HAVIT**: Hierarchical video-text understanding
- **ALBEF**: Vision-language alignment through contrastive learning
- **Vision Transformer (ViT)**: Patch-based image understanding
- **HaGRID Dataset**: 30K hand gesture images across 18 classes

See [docs/Topics/](docs/Topics/) for detailed research documentation.

## Current Status

### Completed ✅
- Input Manager: Camera capture, frame buffering, timestamp management
- Hand Detection: MediaPipe-based detection with 21 landmarks
- ViT Fine-tuning: Gesture classification model trained on HaGRID
- Logging system: Centralized logging with rotation

### In Progress 🔬
- Temporal modeling: LSTM-based dynamic gesture recognition
- Semantic understanding: Vision-language alignment for zero-shot learning
- End-to-end pipeline: Integrating all components

### Planned 📋
- Action Mapper: Gesture-to-command translation
- OS Controller: System command execution
- Real-time demo: Complete application

## Team

- Sley Hortes
- Neil Taison Rigaud
- Khishig Tuguldur
- Eder Guerrero

## License

This is an academic capstone project for National Dong Hwa University.
