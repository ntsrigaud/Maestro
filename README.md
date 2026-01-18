# Hand Gesture Recognition System

A deep learning gesture recognition system using Vision Transformers (ViT-B/16) for robust hand gesture classification.

## Overview

This project implements a **gesture recognition system** trained on the HaGRID dataset using Vision Transformers. The system processes full-frame images through a pre-trained ViT-B/16 model fine-tuned for 18 gesture classes.

**Key Features**:

- **Vision Transformer**: Uses ViT-B/16 pre-trained on ImageNet-21k for rich visual feature extraction
- **HaGRID Dataset**: Trained on 30K+ hand gesture images across 18 gesture classes
- **Full-frame Processing**: Direct image classification without hand detection preprocessing
- **Transfer Learning**: Leverages ImageNet pre-training for efficient learning
- **Production Ready**: Trained model available on Hugging Face Hub

### System Architecture

```
┌──────────┐    ┌──────────────┐    ┌──────────────────┐    ┌────────────┐    ┌──────────┐
│  Camera  │───▶│Input Manager │───▶│Recognition Engine│───▶│Action Mapper│───▶│OS Control│
└──────────┘    └──────────────┘    └──────────────────┘    └────────────┘    └──────────┘
                                            │
                                            └── ViT-B/16 Gesture Classifier
```

**Components**:

- **Input Manager** ✅ - Camera capture and frame preprocessing
- **Recognition Engine** 🔬 - ViT-B/16 gesture classification (training complete, inference in development)
- **Action Mapper** 📋 - Planned
- **OS Controller** 📋 - Planned

### Technical Approach

**Architecture**:

```
Input Image (224x224) → ViT-B/16 Encoder → Classification Head → Gesture Label
```

The model processes images by:
1. Dividing input into 16×16 patches (196 patches total)
2. Applying patch embeddings with positional encoding
3. Processing through 12 transformer encoder layers
4. Classifying via a linear head to 18 gesture classes

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

### Train a Gesture Recognition Model

The training pipeline uses the [neilrigaud/hagrid-subset](https://huggingface.co/datasets/neilrigaud/hagrid-subset) dataset hosted on Hugging Face.

```bash
# 1. Install dependencies
conda env create -f environment.yml
conda activate gesture
pip install -e .

# 2. Open training notebook
jupyter notebook notebooks/finetune_vit_hagrid.ipynb

# 3. Run all cells to train ViT-B/16 model (~30-60 min on GPU)
```

**Training Configuration**:
- Model: ViT-B/16 (timm, ImageNet-21k pre-trained)
- Dataset: HaGRID subset (30K+ images, 18 classes)
- Expected accuracy: 85-90% after 10 epochs
- Trained model: Available at [neilrigaud/hagrid-vit-gesture](https://huggingface.co/neilrigaud/hagrid-vit-gesture)

### Use Pre-trained Model

```python
from transformers import AutoModelForImageClassification, AutoImageProcessor

model = AutoModelForImageClassification.from_pretrained("neilrigaud/hagrid-vit-gesture")
processor = AutoImageProcessor.from_pretrained("neilrigaud/hagrid-vit-gesture")

# Process image and classify
inputs = processor(images=image, return_tensors="pt")
outputs = model(**inputs)
predicted_class = outputs.logits.argmax(-1)
```

## Documentation

### Setup & Development

- **[Development Guide](DEVELOPMENT_GUIDE.md)** - Complete development workflow and common tasks
- **[Installation Guide](INSTALL.md)** - Detailed setup instructions
- **[Architecture Overview](docs/Project-Architecture/README.md)** - System design
- **[API Documentation](docs/)** - Module-level documentation

### Model Training

- **[Training Guide](docs/MODEL_TRAINING_GUIDE.md)** - ViT fine-tuning details
- **[Dataset Guide](docs/HAGRID_DOWNLOAD_GUIDE.md)** - HaGRID dataset access
- **[Training Notebook](notebooks/finetune_vit_hagrid.ipynb)** - Complete training workflow
- **[HaGRID Dataset Assessment](docs/HaGRID_DATASET_ASSESSMENT.md)** - Dataset analysis

## Project Structure

```
├── src/                          # Core implementation
│   ├── input_manager/            # Camera capture (✅ Complete)
│   ├── recognition_engine/       # Gesture recognition (🔬 In Development)
│   │   ├── preprocessing.py      # Image preprocessing utilities
│   │   └── temporal/             # LSTM temporal modeling (🔬 Experimental)
│   └── shared/                   # Logging and utilities (✅ Complete)
├── notebooks/                    # Training and research
│   ├── finetune_vit_hagrid.ipynb       # ViT training workflow (✅ Complete)
│   ├── semantic_gesture_learning.ipynb # Semantic understanding research
│   └── vit_lstm_gesture_training.ipynb # Temporal modeling experiments
├── scripts/                      # Demos and utilities
│   ├── demo_gesture_recognition.py     # Gesture inference demo
│   ├── download_hagrid_subset.py       # Dataset downloader
│   └── verify_install.py               # Installation verification
├── docs/                         # Documentation
│   ├── Project-Architecture/     # Design documents
│   ├── Topics/                   # Research references
│   └── HaGRID_DATASET_ASSESSMENT.md # Dataset analysis
├── checkpoints/                  # Model checkpoints
│   └── best_model.pth           # Trained ViT model
├── environment.yml               # Conda environment
├── pyproject.toml                # Package configuration
└── README.md                     # This file
```

## Research Foundation

This project explores gesture recognition using Vision Transformers, drawing inspiration from:

- **Vision Transformer (ViT)**: Patch-based image understanding with self-attention
- **Transfer Learning**: Leveraging ImageNet-21k pre-training for gesture classification
- **HaGRID Dataset**: Large-scale hand gesture dataset with 18 gesture classes
- **HAVIT & ALBEF**: Vision-language research inspiring future semantic understanding work

See [docs/Topics/](docs/Topics/) for detailed research documentation.

## Current Status

### Completed ✅

- Input Manager: Camera capture, frame buffering, timestamp management
- ViT Fine-tuning: Gesture classification model trained on HaGRID (85-90% accuracy)
- Training Pipeline: Complete notebook workflow with Hugging Face integration
- Model Deployment: Trained model published to Hugging Face Hub
- Logging System: Centralized logging with rotation

### In Progress 🔬

- Recognition Engine: Inference pipeline for real-time gesture classification
- Temporal Modeling: LSTM-based dynamic gesture recognition (experimental)
- End-to-end Integration: Connecting all system components

### Planned 📋

- Action Mapper: Gesture-to-command translation layer
- OS Controller: System command execution module
- Real-time Demo: Complete application with UI
- Semantic Understanding: Vision-language alignment for zero-shot learning (research)

## Team

- Sley Hortes
- Neil Taison Rigaud
- Khishig Tuguldur
- Eder Guerrero

## License

This is an academic capstone project for National Dong Hwa University.
