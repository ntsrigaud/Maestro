# Clean Baseline Refactoring - Complete Documentation

## Overview
This document explains the complete refactoring to establish a **pristine clean baseline** after discovering that test-time augmentation (TTA) caused catastrophic performance collapse.

## Critical Discovery: TTA-Induced Collapse

### The Problem
- **Standard inference**: 89-90% accuracy ✓
- **With TTA enabled**: Drops to 14% accuracy ✗
- **Confusion matrix**: Heavy bias toward 'palm' class
- **Root cause**: Augmentation corrupting spatial relationships

### Why This Happened
Hand gestures are **spatial tasks** that depend on:
- Precise finger positions
- Hand orientation (palm vs back of hand)
- Spatial topology (finger configurations)

Augmentation (flips, color jitter, etc.) corrupts these critical spatial features rather than helping.

## Refactoring: Version 3.0 - Clean Baseline

### Philosophy
1. **Remove ALL confounding variables**
2. **Establish ground truth baseline**
3. **Validate model capability**
4. **Add complexity ONLY after achieving >90% baseline**

### What Was Changed

#### 1. Data Preprocessing (Cell 8)
**BEFORE** (Minimal Augmentation):
```python
# Training transforms
train_transform = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.RandomHorizontalFlip(p=0.5),  # ✗ REMOVED
    transforms.ColorJitter(                   # ✗ REMOVED
        brightness=0.2,
        contrast=0.2,
        saturation=0.2,
        hue=0.05
    ),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

# Test-time augmentation (2 views)
tta_transforms = [
    transforms.Compose([...]),                # Original
    transforms.Compose([RandomHorizontalFlip(p=1.0), ...])  # Flipped
]
# Average predictions across views  # ✗ REMOVED
```

**AFTER** (Zero Augmentation):
```python
# IDENTICAL transform for train/val/test/demo
base_transform = transforms.Compose([
    transforms.Resize((224, 224)),           # Deterministic resize
    transforms.ToTensor(),                   # [0, 255] → [0, 1]
    transforms.Normalize(                    # ImageNet stats
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])

# All datasets use IDENTICAL preprocessing
train_transform = base_transform
val_transform = base_transform
test_transform = base_transform

# NO test-time augmentation
# Single forward pass only
```

**Changes**:
- ✗ Removed `RandomHorizontalFlip`
- ✗ Removed `ColorJitter`
- ✗ Removed all test-time augmentation (TTA)
- ✓ Made train/val/test transforms identical
- ✓ Deterministic preprocessing only

#### 2. Training Configuration (Cell 14)
**BEFORE** (Baseline with Weights/Smoothing):
```python
# CrossEntropyLoss with class weights + label smoothing
criterion = nn.CrossEntropyLoss(
    weight=class_weights_tensor,    # ✗ REMOVED
    label_smoothing=0.1             # ✗ REMOVED
)
```

**AFTER** (Pristine Baseline):
```python
# Plain CrossEntropyLoss (no tricks)
criterion = nn.CrossEntropyLoss()
```

**Changes**:
- ✗ Removed class weights (let model learn natural distribution)
- ✗ Removed label smoothing (clean one-hot targets)
- ✓ Simplest possible loss function

#### 3. Demo Script (scripts/demo_gesture_recognition.py)
**BEFORE**:
```python
self.transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])
```

**AFTER**:
```python
# MUST match training exactly
self.transform = transforms.Compose([
    transforms.Resize((224, 224)),      # Explicit resize
    transforms.ToTensor(),              # Convert to tensor
    transforms.Normalize(               # ImageNet normalization
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])
```

**Changes**:
- ✓ Added explicit `Resize` to match notebook
- ✓ Updated architecture detection (`head.head.` for EnhancedViT)
- ✓ Identical preprocessing to notebook

## Configuration Summary

### Model Architecture
- **Backbone**: ViT-B/16 (12 transformer blocks, 86M params)
- **Head**: Single linear layer (768 → 18 classes)
- **Pretrained**: ImageNet-21k → ImageNet-1k
- **No modifications**: Standard timm architecture

### Training Hyperparameters
```python
# Learning rates
LEARNING_RATE = 5e-5         # Backbone
HEAD_LR = 1e-4               # Head (2x backbone)

# Optimizer
Optimizer: AdamW
Weight Decay: 0.01

# Scheduler
Type: Cosine Annealing + Linear Warmup
Warmup: 2 epochs
Total: 15 epochs

# Loss
Type: CrossEntropyLoss()     # Plain, no arguments
```

### Data Configuration
```python
# Dataset split
Train: 70% (~21,000 images)
Val:   15% (~4,500 images)
Test:  15% (~4,500 images)

# Preprocessing (IDENTICAL for train/val/test)
Resize: 224×224 (bicubic)
ToTensor: [0, 255] → [0, 1]
Normalize: ImageNet mean/std

# Augmentation
Training: NONE (zero augmentation)
Testing: NONE (single forward pass)
```

### Hardware
- **Batch size**: 32
- **AMP**: Enabled (mixed precision)
- **Device**: CUDA/MPS/CPU (auto-detect)

## Expected Performance

### Success Criteria
- **Top-1 Accuracy**: 85-92%
- **Per-class Recall**: >70% for all 18 classes
- **Confusion Matrix**: Strong diagonal (>85% diagonality)
- **Training Stability**: Monotonic improvement, val loss tracks train loss

### Failure Indicators
- Accuracy <80%
- Confusion matrix collapse to single class
- Val loss diverges from train loss
- High off-diagonal confusion scores

## Validation Steps

After retraining with clean baseline:

1. **Check training curves**:
   - Train/val loss should decrease monotonically
   - Val loss should track train loss (no divergence)
   - Accuracy should improve steadily

2. **Evaluate confusion matrix**:
   ```python
   # Should see strong diagonal
   diagonality = cm.diagonal().sum() / cm.sum()
   print(f"Diagonality: {diagonality:.2%}")  # Target: >85%
   ```

3. **Per-class analysis**:
   - All classes should have >70% recall
   - Check for systematic confusion patterns
   - Identify challenging gesture pairs

4. **Demo script validation**:
   ```bash
   python scripts/demo_gesture_recognition.py
   ```
   - Should show stable real-time predictions
   - No jitter or flickering between classes
   - Consistent with notebook evaluation

## Future Enhancements (Only After >90% Baseline)

If clean baseline achieves >90% accuracy with strong diagonality:

1. **Horizontal Flip Only**:
   ```python
   # Left ↔ right hand equivalence (makes sense for gestures)
   train_transform = transforms.Compose([
       transforms.Resize((224, 224)),
       transforms.RandomHorizontalFlip(p=0.5),  # Only this
       transforms.ToTensor(),
       transforms.Normalize(...)
   ])
   ```

2. **Validate Improvement**:
   - Must improve accuracy >2%
   - Must maintain confusion matrix quality
   - Must not introduce class bias

3. **Add More Augmentation Gradually**:
   - Test each augmentation independently
   - Validate improvement before adding next
   - Stop if performance degrades

## Files Modified

### Notebooks
- `notebooks/finetune_vit_hagrid.ipynb`
  - Cell 1 (header): Updated with clean baseline documentation
  - Cell 8 (preprocessing): Removed ALL augmentation
  - Cell 14 (training config): Removed class weights & label smoothing

### Scripts
- `scripts/demo_gesture_recognition.py`
  - Updated preprocessing to match notebook exactly
  - Fixed architecture detection (head.head. vs head.)
  - Added explicit Resize step

### Documentation
- `CLEAN_BASELINE_REFACTORING.md` (this file)
- `TRAINING_DIAGNOSTIC_FIXES.md` (previous 29% failure analysis)
- `AUGMENTATION_STRATEGY_FIXES.md` (why augmentation harms gestures)
- `PERFORMANCE_RECOVERY_SUMMARY.md` (complete recovery guide)

## Troubleshooting

### Issue: Model still shows low accuracy after retraining
**Solution**: Verify preprocessing pipeline:
```python
# Check transforms are identical
print("Train:", train_transform)
print("Val:  ", val_transform)
print("Test: ", test_transform)
# All should print identical output
```

### Issue: Demo script predictions don't match notebook
**Solution**: Check preprocessing consistency:
```python
# In demo script, verify:
self.transform = transforms.Compose([
    transforms.Resize((224, 224)),  # Must match notebook
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])
```

### Issue: Architecture loading fails
**Solution**: Check state_dict keys:
```python
checkpoint = torch.load('best_model.pth')
print(list(checkpoint.keys())[:10])

# If keys start with 'head.head.': EnhancedViT
# If keys start with 'head.weight': Standard ViT
```

## Key Takeaways

1. **Augmentation can hurt**: For spatial tasks like hand gestures, augmentation may corrupt critical features
2. **Start simple**: Establish pristine baseline before adding complexity
3. **Consistency matters**: Train/val/test/demo must use IDENTICAL preprocessing
4. **TTA is not free**: Test-time augmentation can severely degrade performance
5. **Validate incrementally**: Add enhancements one at a time and validate improvement

## Next Steps

1. **Retrain model** with clean baseline configuration
2. **Evaluate performance**:
   - Target: 85-92% accuracy
   - Check confusion matrix diagonality >85%
   - Validate per-class recall >70%
3. **Test demo script**:
   - Real-time performance
   - Consistent predictions
   - No jitter or class bias
4. **Document results**:
   - Training curves
   - Confusion matrix
   - Per-class metrics
5. **Consider enhancements** (only if baseline >90%):
   - Horizontal flip augmentation
   - Validate improvement
   - Document changes

---

**Version**: 3.0 - Clean Baseline (Pristine Configuration)  
**Date**: 2025-01-XX  
**Status**: Ready for Training  
**Target**: 85-92% accuracy with strong confusion matrix diagonal  
