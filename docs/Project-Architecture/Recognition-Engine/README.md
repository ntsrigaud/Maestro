# Recognition Engine

The Recognition Engine is responsible for gesture recognition using Vision Transformers. It processes full-frame images to classify 18 hand gestures from the HaGRID dataset.

## Current Implementation Status

### Completed ✅

- **ViT-B/16 Training Pipeline** - Complete training workflow using Hugging Face dataset
  - Model: ViT-B/16 pre-trained on ImageNet-21k
  - Dataset: HaGRID subset (~30K images, 18 classes)
  - Training: 85-90% validation accuracy achieved
  - Deployment: Model published to [neilrigaud/hagrid-vit-gesture](https://huggingface.co/neilrigaud/hagrid-vit-gesture)

### In Development 🔬

- **Inference Pipeline** - Real-time gesture classification module
  - Load trained model
  - Preprocess frames
  - Run inference
  - Post-process predictions

### Planned 📋

- **Temporal Modeling** - LSTM-based dynamic gesture recognition (experimental)
- **Semantic Understanding** - Vision-language alignment research (ALBEF-inspired)

---

## Architecture Overview

### Current: ViT-B/16 Classifier

The system uses a **full-frame classification approach** with Vision Transformer:

```
Input Frame (224×224×3)
    ↓
Patch Embedding (16×16 patches)
    ↓
Transformer Encoder (12 layers)
    ↓
Classification Head
    ↓
Gesture Prediction (18 classes)
```

**Key Details**:

- No hand detection preprocessing required
- Processes entire frame through ViT
- Single-stage classification
- Leverages ImageNet-21k pre-training

---

## Gesture Classes

The model recognizes 18 hand gestures from the HaGRID dataset:

```
call            - Phone gesture
dislike         - Thumbs down
fist            - Closed fist
four            - Four fingers
like            - Thumbs up
mute            - Quiet/silence gesture
ok              - OK sign
one             - Index finger up
palm            - Open palm
peace           - Peace sign (V)
peace_inverted  - Peace sign inverted
rock            - Rock/metal sign
stop            - Stop hand
stop_inverted   - Stop hand inverted
three           - Three fingers
three2          - Three fingers (variant)
two_up          - Two fingers up
two_up_inverted - Two fingers up inverted
```

---

## Training

Training is performed via the Jupyter notebook: [notebooks/finetune_vit_hagrid.ipynb](../../../notebooks/finetune_vit_hagrid.ipynb)

**Training Configuration**:

```python
model = "vit_base_patch16_224"  # ViT-B/16
pretrained = True                # ImageNet-21k weights
num_classes = 18                 # HaGRID gestures
batch_size = 32
learning_rate = 1e-4
epochs = 10
optimizer = "AdamW"
```

**Dataset**: HuggingFace `neilrigaud/hagrid-subset`

- Train split: ~24K images
- Validation split: ~3K images
- Test split: ~3K images

**Results**:

- Training Accuracy: ~90%
- Validation Accuracy: 85-90%
- Training Time: 30-60 minutes on GPU

---

## Inference (In Development)

### Planned Usage

```python
from recognition_engine import GestureClassifier

# Load trained model
classifier = GestureClassifier(
    model_path="checkpoints/best_model.pth",
    device="cuda"
)

# Classify single frame
result = classifier.predict(frame)
print(f"Gesture: {result.label}, Confidence: {result.confidence:.2f}")

# Batch inference
results = classifier.predict_batch(frames)
```

### Current Status

The inference module is under development. Currently available:

- Trained model checkpoint: `checkpoints/best_model.pth`
- Model on Hugging Face Hub: `neilrigaud/hagrid-vit-gesture`
- Preprocessing utilities: `src/recognition_engine/preprocessing.py`

---

## Module Structure

```
src/recognition_engine/
├── __init__.py              # Module initialization
├── preprocessing.py         # Image preprocessing utilities
└── temporal/                # 🔬 LSTM temporal modeling (experimental)
    └── README.md
```

---

## Demo

Currently no demo script available (inference pipeline in development).

For training demonstration, use the notebook:

```bash
jupyter notebook notebooks/finetune_vit_hagrid.ipynb
```

---

## Future Enhancements

### Temporal Modeling (Experimental)

Research direction: Add LSTM for dynamic gesture recognition

```
Frame Sequence (T frames)
    ↓
ViT Feature Extraction (per frame)
    ↓
LSTM Temporal Modeling
    ↓
Dynamic Gesture Classification
```

**Status**: Experimental code in `src/recognition_engine/temporal/`

### Semantic Understanding (Research)

Long-term research direction: Vision-language alignment

**Inspiration**: HAVIT, ALBEF approaches

- Vision-text semantic space
- Zero-shot gesture recognition
- Natural language command mapping

**Status**: Conceptual stage, documented in research notebooks

---

## Performance

### Current System (ViT-B/16)

| Metric              | Value                |
| ------------------- | -------------------- |
| Model Size          | ~86M parameters      |
| Input Size          | 224×224×3            |
| Classes             | 18 gestures          |
| Validation Accuracy | 85-90%               |
| Training Time       | 30-60 min (GPU)      |
| Inference Speed     | TBD (in development) |

### Comparison with MediaPipe Approach

**Previous MediaPipe Approach** (no longer used):

- Hand landmark detection (21 points)
- Lightweight MLP classifier
- Pros: Fast, low compute
- Cons: Limited by landmark precision, loses visual context

**Current ViT Approach**:

- Full-frame deep learning
- Rich visual feature extraction
- Pros: Robust, accurate, transfer learning
- Cons: Higher compute, larger model

---

## References

- [Vision Transformer Paper](https://arxiv.org/abs/2010.11929) - "An Image is Worth 16x16 Words"
- [HaGRID Dataset](https://github.com/hukenovs/hagrid) - Hand Gesture Recognition Image Dataset
- [timm Library](https://github.com/huggingface/pytorch-image-models) - PyTorch Image Models
- [Trained Model](https://huggingface.co/neilrigaud/hagrid-vit-gesture) - Hugging Face Hub

---

## Related Documentation

- [System Architecture](../DETAILED_SYSTEM_ARCHITECTURE.md) - Complete system design
- [Training Guide](../../../docs/MODEL_TRAINING_GUIDE.md) - Model training details
- [HaGRID Dataset Assessment](../../../docs/HaGRID_DATASET_ASSESSMENT.md) - Dataset analysis
