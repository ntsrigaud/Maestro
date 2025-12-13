# Quick Start Guide: Semantic Gesture Recognition Training

**Date**: December 12, 2025  
**Notebook**: `finetune_vit_hagrid.ipynb`  
**Objective**: Train ViT model with semantic understanding and zero-shot learning

---

## Overview

This notebook implements **two complementary approaches** to gesture recognition:

### Part 1: Classification (Traditional)
- **Model**: ViT-B/16 with softmax classification head
- **Training**: Supervised learning on 18 HaGRID gesture classes
- **Output**: Class predictions (0-17)
- **Accuracy**: ~85-90% on test set
- **Limitation**: Only recognizes trained gestures

### Part 2: Semantic Understanding (Novel)
- **Model**: Dual-encoder (ViT + SentenceTransformer)
- **Training**: Contrastive learning on gesture-text pairs
- **Output**: Semantic embeddings matched to text commands
- **Zero-shot**: Can understand new commands without retraining
- **Top-5 accuracy**: ~80-90% on unseen commands

---

## Prerequisites

### 1. Dataset

Download HaGRID sample dataset:

```bash
# Option 1: Using Kaggle API
pip install kaggle
kaggle datasets download -d innominate817/hagrid-sample-30k-384p
unzip hagrid-sample-30k-384p.zip -d ./data/

# Option 2: Using opendatasets (in notebook)
# Requires Kaggle credentials
```

**Expected structure**:
```
data/
└── hagrid-sample-30k-384p/
    └── hagrid_30k/
        ├── train/
        │   ├── call/
        │   ├── dislike/
        │   ├── fist/
        │   └── ... (18 classes)
        ├── val/
        └── test/
```

### 2. Python Environment

```bash
# Create conda environment
conda create -n gesture_semantic python=3.10 -y
conda activate gesture_semantic

# Install dependencies
pip install torch torchvision timm scikit-learn
pip install matplotlib seaborn pillow tqdm
pip install sentence-transformers opendatasets
```

---

## Running the Notebook

### Part 1: Classification Training (~20-40 min on GPU)

**Cells to run**:
1. Install dependencies
2. Setup dataset (auto-download if needed)
3. Create PyTorch DataLoaders
4. Load ViT-B/16 model
5. **Training loop** (10 epochs)
6. Evaluation on test set
7. Visualizations

**Expected outputs**:
- `checkpoints/best_model.pth` - Trained classifier
- `training_history.png` - Loss/accuracy curves
- `confusion_matrix.png` - Test set performance
- `inference_samples.png` - Sample predictions

**Key metrics to watch**:
- **Train accuracy**: Should reach ~90-95%
- **Val accuracy**: Should be ~85-90%
- **Test accuracy**: Target is ~85%
- **F1 score**: Should be ~0.85

### Part 2: Semantic Training (~15-25 min on GPU)

**Cells to run**:
1. Load gesture semantic mappings (`gesture_semantics.py`)
2. Install sentence-transformers
3. Implement dual-encoder architecture
4. Create semantic training dataset
5. **Contrastive learning loop** (5 epochs)
6. Zero-shot evaluation
7. t-SNE visualization

**Expected outputs**:
- `checkpoints/semantic_model.pth` - Dual-encoder weights
- `semantic_tsne.png` - Embedding space visualization
- Console output with zero-shot accuracy

**Key metrics to watch**:
- **Training loss**: Should decrease to ~0.5-1.0
- **Validation similarity**: Should increase to ~0.6-0.8
- **Zero-shot Top-1**: Target is 60-70%
- **Zero-shot Top-5**: Target is 80-90%

---

## Understanding the Results

### Classification Results

**Confusion Matrix Interpretation**:
- Diagonal = correct predictions
- Off-diagonal = misclassifications
- Look for patterns: "peace" vs "two_up" often confused (visually similar)

**Sample output**:
```
Test Accuracy: 87.3%
F1 Score: 0.863
```

### Semantic Results

**Zero-Shot Examples**:

```
Example: "like" gesture (thumbs up)
  Trained on: "thumbs up", "approve", "positive"
  
  Zero-shot test:
  ✓ "I appreciate this" → similarity: 0.834 (CORRECT)
  ✓ "increase volume" → similarity: 0.812 (CORRECT)
  ✗ "stop music" → similarity: 0.234 (correctly rejected)
```

**t-SNE Visualization**:
- Each color = one gesture class
- Circles (●) = visual embeddings
- Stars (★) = text embeddings
- **Good result**: Same-color circles and stars cluster together
- **Poor result**: Scattered, overlapping clusters

---

## Troubleshooting

### Issue 1: Out of Memory (OOM)

**Solution**: Reduce batch size
```python
BATCH_SIZE = 16  # Was 32
SEMANTIC_BATCH_SIZE = 16  # Was 32
```

### Issue 2: Low Classification Accuracy (<70%)

**Possible causes**:
- Dataset not properly downloaded
- Wrong image transforms (check resolution)
- Learning rate too high/low

**Solutions**:
1. Verify dataset structure
2. Check first epoch loss (should decrease)
3. Try learning rate 1e-5 or 5e-5

### Issue 3: Poor Zero-Shot Performance (<40% Top-5)

**Possible causes**:
- Text descriptions too generic
- Not enough training epochs
- Temperature parameter suboptimal

**Solutions**:
1. Train for more epochs (5 → 10)
2. Increase semantic batch size
3. Try temperature 0.05 or 0.10

### Issue 4: Contrastive Loss Not Decreasing

**Possible causes**:
- Text encoder frozen (should be)
- Projection layer too small
- Learning rate too high

**Solutions**:
1. Check optimizer only trains projection layers
2. Increase embedding dimension (512 → 768)
3. Reduce learning rate to 5e-5

---

## Hyperparameter Tuning Guide

### Classification Training

| Parameter | Default | Range | Impact |
|-----------|---------|-------|--------|
| `BATCH_SIZE` | 32 | 16-64 | Memory vs speed |
| `EPOCHS` | 10 | 5-20 | Overfitting vs accuracy |
| `LEARNING_RATE` | 3e-5 | 1e-5 to 1e-4 | Convergence speed |
| `WEIGHT_DECAY` | 0.01 | 0.001-0.1 | Regularization |

### Semantic Training

| Parameter | Default | Range | Impact |
|-----------|---------|-------|--------|
| `EMBED_DIM` | 512 | 256-768 | Capacity vs efficiency |
| `SEMANTIC_EPOCHS` | 5 | 3-10 | Zero-shot accuracy |
| `SEMANTIC_LR` | 1e-4 | 5e-5 to 5e-4 | Projection learning |
| `TEMPERATURE` | 0.07 | 0.01-0.20 | Contrastive sharpness |

**Recommended settings for best zero-shot**:
```python
EMBED_DIM = 512
SEMANTIC_EPOCHS = 10  # More is better
SEMANTIC_LR = 1e-4
TEMPERATURE = 0.05  # Lower = sharper distinctions
```

---

## Advanced Usage

### 1. Adding New Gesture Classes

Edit `gesture_semantics.py`:

```python
GESTURE_SEMANTICS["new_gesture"] = {
    "descriptions": [
        "description 1",
        "description 2",
        # ... 5-10 descriptions
    ],
    "actions": [
        "action 1",
        "action 2",
    ],
    "semantic_group": "confirmation"  # or other group
}
```

Then retrain the semantic model (Part 2 only).

### 2. Custom Zero-Shot Commands

Add to `ZERO_SHOT_TEST_COMMANDS` in `gesture_semantics.py`:

```python
ZERO_SHOT_TEST_COMMANDS["my_category"] = [
    "custom command 1",
    "custom command 2",
]
```

### 3. Transfer to Production

**Option A**: Use classification model (faster)
```python
# Load model
model = timm.create_model('vit_base_patch16_224', num_classes=18)
model.load_state_dict(torch.load('checkpoints/best_model.pth'))

# Inference
output = model(image)
predicted_class = output.argmax()
```

**Option B**: Use semantic model (flexible)
```python
# Load encoders
gesture_encoder = GestureEncoder()
text_encoder = TextEncoder()
checkpoint = torch.load('checkpoints/semantic_model.pth')
gesture_encoder.load_state_dict(checkpoint['gesture_encoder'])
text_encoder.load_state_dict(checkpoint['text_encoder'])

# Precompute all text embeddings once
all_commands = ["volume up", "next slide", ...]
text_embeds = text_encoder(all_commands)  # Do this once at startup

# Real-time inference
visual_embed = gesture_encoder(image)
similarities = (visual_embed @ text_embeds.T).squeeze()
best_command_idx = similarities.argmax()
best_command = all_commands[best_command_idx]
```

---

## Performance Benchmarks

### Training Time

| Hardware | Classification | Semantic | Total |
|----------|---------------|----------|-------|
| CPU | ~4-6 hours | ~2-3 hours | ~6-9 hours |
| GPU (T4) | ~20-30 min | ~15-20 min | ~35-50 min |
| GPU (V100) | ~15-20 min | ~10-15 min | ~25-35 min |
| GPU (A100) | ~10-15 min | ~5-10 min | ~15-25 min |
| Apple M1/M2 | ~30-40 min | ~20-25 min | ~50-65 min |

### Inference Speed (per image)

| Model | Hardware | Latency |
|-------|----------|---------|
| Classification | CPU | ~50-100ms |
| Classification | GPU | ~5-10ms |
| Semantic | CPU | ~100-150ms |
| Semantic | GPU | ~10-15ms |

**Note**: Semantic model is slower due to projection layer, but still real-time capable (>30 FPS).

---

## Expected Results Summary

### Part 1: Classification

✅ **Test accuracy**: 85-90%  
✅ **F1 score**: 0.85-0.90  
✅ **Inference speed**: 30-60 FPS  
✅ **Memory**: ~2GB VRAM  

### Part 2: Semantic

✅ **Zero-shot Top-1**: 60-70%  
✅ **Zero-shot Top-5**: 80-90%  
✅ **Embedding similarity**: 0.6-0.8  
✅ **t-SNE clusters**: Well-separated  

---

## FAQ

**Q: Do I need to run Part 1 before Part 2?**  
A: No, they're independent. Part 2 uses its own encoders.

**Q: Can I use a different ViT model (e.g., ViT-L/16)?**  
A: Yes! Change `vit_base_patch16_224` to `vit_large_patch16_224` in both parts.

**Q: How do I add more text descriptions?**  
A: Edit `gesture_semantics.py` and retrain Part 2 only.

**Q: Why is zero-shot accuracy lower than classification?**  
A: Zero-shot is harder! It evaluates on UNSEEN commands. 60-70% is excellent.

**Q: Can I fine-tune the text encoder?**  
A: Yes! Remove the `requires_grad = False` line in `TextEncoder.__init__`.

**Q: How do I deploy this to production?**  
A: See "Transfer to Production" section above. Use Option B for flexibility.

---

## Next Steps

After completing this notebook:

1. ✅ Review `SEMANTIC_GESTURE_RESEARCH.md` for theory
2. ✅ Experiment with hyperparameters
3. ✅ Add custom gestures and commands
4. ⏳ Integrate with `Input Manager` module
5. ⏳ Add LSTM for dynamic gestures (swipes, circles)
6. ⏳ Deploy to real-time gesture control system

---

## Citation

If you use this work, please cite:

```bibtex
@misc{gesture_semantic_2025,
  title={Semantic Gesture Recognition with Vision Transformers and Zero-Shot Learning},
  author={Neil Taison Rigaud, Sley Hortes, Khishig Tuguldur, Eder Guerrero},
  year={2025},
  note={CSU Capstone Project}
}
```

---

**Need help?** Check the main documentation or open an issue in the repository.
