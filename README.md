# Hand Gesture Recognition System

A semantic gesture recognition system that understands the _meaning_ of hand gestures through vision-language alignment, enabling zero-shot recognition and natural command mapping.

## Overview

This project implements a **semantic gesture recognition system** inspired by state-of-the-art vision-language models (HAVIT, ALBEF). Unlike traditional systems that rely on hardcoded gesture-to-action mappings, our approach learns a shared semantic space between visual gestures and text commands, enabling:

- **Zero-shot learning**: Recognize new gesture variations without retraining
- **Semantic understanding**: Map gestures to their _meaning_, not just visual patterns
- **Extensibility**: Add new commands via text descriptions, no model retraining needed
- **Natural interaction**: Bridge vision and language for intuitive control

### System Architecture

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

**Components**:

- **Input Manager** ✅ - Camera capture and frame preprocessing
- **Recognition Engine** 🔬 - Hand detection (complete), gesture classification (in research)
- **Action Mapper** 📋 - Planned
- **OS Controller** 📋 - Planned

### Core Innovation

**Traditional approach**:

```
Gesture → Classifier → Label → Hardcoded action
```

**Our approach**:

```
Gesture → Visual Embedding (ViT) → Semantic Space ↔ Text Embedding (BERT) → Action
```

This enables flexible, meaning-based gesture interpretation instead of rigid pattern matching.

## Quick Start

### Installation

```bash
# Create conda environment
conda env create -f environment.yml
conda activate gesture

# Install project in editable mode
pip install -e .

# Verify installation
python scripts/verify_install.py
```

See [INSTALL.md](INSTALL.md) for detailed instructions.

### Try Hand Detection

```bash
python scripts/demo_hand_detection.py
```

This opens a real-time window showing hand detection with landmarks and bounding boxes.

### Train a Gesture Recognition Model

```bash
# 1. Download dataset (~1.5 GB, 5-15 minutes)
pip install kaggle
python scripts/download_hagrid_kaggle.py --save_path ./data/hagrid_subset

# 2. Open training notebook
jupyter notebook notebooks/finetune_vit_hagrid.ipynb

# 3. Run all cells to train ViT model (~30-60 min on GPU)
```

Expected accuracy: 85-90% after 10 epochs.

## Documentation

### Setup & Development

- **[Development Guide](DEVELOPMENT_GUIDE.md)** - Complete development workflow and common tasks
- **[Installation Guide](INSTALL.md)** - Detailed setup instructions
- **[Architecture Overview](docs/Project-Architecture/README.md)** - System design
- **[API Documentation](docs/)** - Module-level documentation

### Model Training

- **[Training Guide](TRAINING_README.md)** - Model training details
- **[Retrain Quick Start](RETRAIN_QUICKSTART.md)** - ⚡ Fast guide to retrain clean baseline model
- **[Clean Baseline Refactoring](CLEAN_BASELINE_REFACTORING.md)** - 📋 Complete explanation of Version 3.0 changes
- **[Validation Checklist](CLEAN_BASELINE_VALIDATION.md)** - ✓ Step-by-step verification guide
- **[Before/After Comparison](CLEAN_BASELINE_BEFORE_AFTER.md)** - 🔄 Side-by-side configuration comparison

## Project Structure

```
├── src/                          # Core implementation
│   ├── input_manager/            # Camera capture (✅ Complete)
│   ├── recognition_engine/       # Gesture recognition (🔬 Research)
│   │   ├── hand_detection/       # MediaPipe integration (✅ Complete)
│   │   └── temporal/             # LSTM temporal modeling (🔬 Experimental)
│   └── shared/                   # Utilities (✅ Complete)
├── notebooks/                    # Training and research
│   ├── finetune_vit_hagrid.ipynb       # Main training workflow
│   ├── semantic_gesture_learning.ipynb # Semantic understanding
│   └── vit_lstm_gesture_training.ipynb # Temporal modeling
├── scripts/                      # Demos and utilities
│   ├── demo_hand_detection.py    # Hand detection demo
│   ├── demo_gesture_recognition.py # Full pipeline demo
│   └── download_hagrid_kaggle.py # Dataset downloader
├── docs/                         # Documentation
│   ├── Project-Architecture/     # Design documents
│   ├── Topics/                   # Research references
│   └── HaGRID_DATASET_ASSESSMENT.md # Dataset analysis
├── environment.yml               # Conda environment
├── pyproject.toml                # Package configuration
└── README.md                     # This file
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
