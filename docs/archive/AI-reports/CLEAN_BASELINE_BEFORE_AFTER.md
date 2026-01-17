# Clean Baseline: Before vs After Comparison

## Quick Reference: What Changed

### TL;DR
- **Removed**: ALL data augmentation (flip, color jitter, TTA)
- **Removed**: Class weights and label smoothing from loss
- **Changed**: Made train/val/test preprocessing IDENTICAL
- **Updated**: Demo script to match exact notebook preprocessing
- **Result**: Pristine baseline configuration for reliable evaluation

---

## 1. Data Preprocessing Transforms

### BEFORE (Minimal Augmentation)
```python
# Training transforms (DIFFERENT from val/test)
train_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.RandomHorizontalFlip(p=0.5),      # ❌ REMOVED
    transforms.ColorJitter(                      # ❌ REMOVED
        brightness=0.2,
        contrast=0.2,
        saturation=0.2,
        hue=0.05
    ),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])

# Validation/test transforms (simpler)
val_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])

# Test-time augmentation (2 views)
tta_transforms = [
    transforms.Compose([                         # ❌ REMOVED
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ]),
    transforms.Compose([                         # ❌ REMOVED
        transforms.RandomHorizontalFlip(p=1.0),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
]

# Evaluation with TTA
with torch.no_grad():
    for transform in tta_transforms:              # ❌ REMOVED
        img_transformed = transform(img)
        output = model(img_transformed)
        outputs_list.append(output)
    outputs_avg = torch.mean(torch.stack(outputs_list), dim=0)
```

### AFTER (Zero Augmentation)
```python
# Single transform for ALL datasets (train/val/test/demo)
base_transform = transforms.Compose([
    transforms.Resize((224, 224)),                # Deterministic resize
    transforms.ToTensor(),                        # [0,255] → [0,1]
    transforms.Normalize(                         # ImageNet stats
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
with torch.no_grad():
    img_tensor = test_transform(img).unsqueeze(0).to(device)
    output = model(img_tensor)                    # Single pass
    pred = output.argmax(dim=1).item()
```

**Key Changes**:
- ✅ `train_transform == val_transform == test_transform`
- ❌ No `RandomHorizontalFlip`
- ❌ No `ColorJitter`
- ❌ No test-time augmentation (TTA)
- ✅ Deterministic preprocessing only

---

## 2. Loss Function

### BEFORE (Weighted + Smoothing)
```python
# Class weights computed from dataset
class_weights = []
for cls_idx in range(NUM_CLASSES):
    count = sum(1 for _, label in dataset.samples if label == cls_idx)
    weight = 1.0 / (count + 1e-6)
    class_weights.append(weight)

class_weights_tensor = torch.tensor(class_weights, dtype=torch.float32).to(device)
class_weights_tensor = class_weights_tensor / class_weights_tensor.sum() * NUM_CLASSES

# CrossEntropyLoss with class weights AND label smoothing
criterion = nn.CrossEntropyLoss(
    weight=class_weights_tensor,                  # ❌ REMOVED
    label_smoothing=0.1                           # ❌ REMOVED
)
```

### AFTER (Plain)
```python
# Plain CrossEntropyLoss (no parameters)
criterion = nn.CrossEntropyLoss()
```

**Key Changes**:
- ❌ No class weights (let model learn natural distribution)
- ❌ No label smoothing (clean one-hot targets)
- ✅ Simplest possible loss function

---

## 3. Model Architecture

### BEFORE & AFTER (No Change)
```python
# Standard ViT-B/16 (remains unchanged)
model = timm.create_model(
    'vit_base_patch16_224',
    pretrained=True,
    num_classes=NUM_CLASSES  # 18 classes
)
model = model.to(device)
```

**Architecture Details**:
- Backbone: 12 transformer blocks, 768 hidden dim, 86M params
- Head: Single linear layer (768 → 18)
- Pretrained: ImageNet-21k → ImageNet-1k
- No custom head, no additional layers

---

## 4. Optimizer Configuration

### BEFORE & AFTER (Minor Clarification)
```python
# AdamW with 2x learning rate disparity (unchanged)
optimizer = optim.AdamW([
    {
        'params': [p for n, p in model.named_parameters() if 'head' not in n],
        'lr': 5e-5,        # Backbone: 5e-5
        'name': 'backbone'
    },
    {
        'params': [p for n, p in model.named_parameters() if 'head' in n],
        'lr': 1e-4,        # Head: 1e-4 (2x backbone)
        'name': 'head'
    }
], weight_decay=0.01)
```

**Key Points**:
- ✅ 2x LR disparity (not 5x - already fixed previously)
- ✅ AdamW optimizer (unchanged)
- ✅ Weight decay 0.01 (unchanged)

---

## 5. Demo Script Preprocessing

### BEFORE
```python
# Demo script (scripts/demo_gesture_recognition.py)
# Line ~289
self.transform = transforms.Compose([
    transforms.ToTensor(),                        # ❌ Missing Resize
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])

# Architecture detection (incorrect)
is_enhanced_vit = any(
    key.startswith('backbone.') or key.startswith('head.')  # ❌ Too broad
    for key in state_dict_keys
)
```

### AFTER
```python
# Demo script (scripts/demo_gesture_recognition.py)
# Line ~289
self.transform = transforms.Compose([
    transforms.Resize((224, 224)),                # ✅ Added explicit resize
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])

# Architecture detection (corrected)
is_enhanced_vit = any(
    key.startswith('head.head.')                  # ✅ Specific to EnhancedViT
    for key in state_dict_keys
)
```

**Key Changes**:
- ✅ Added explicit `Resize((224, 224))` to match notebook
- ✅ Fixed architecture detection logic (`head.head.` for EnhancedViT)
- ✅ Preprocessing now IDENTICAL to notebook

---

## 6. Training Loop

### BEFORE & AFTER (No Significant Changes)
```python
# Standard training loop (unchanged)
for epoch in range(EPOCHS):
    # Training phase
    model.train()
    for images, labels in train_loader:
        optimizer.zero_grad()
        outputs = model(images.to(device))
        loss = criterion(outputs, labels.to(device))
        loss.backward()
        optimizer.step()
    
    scheduler.step()
    
    # Validation phase
    model.eval()
    with torch.no_grad():
        for images, labels in val_loader:
            outputs = model(images.to(device))
            loss = criterion(outputs, labels.to(device))
```

**No changes** to training loop logic, only to:
- Data preprocessing (transforms)
- Loss function (criterion)
- Evaluation (no TTA)

---

## Side-by-Side Comparison Table

| Component | BEFORE | AFTER | Change |
|-----------|--------|-------|--------|
| **Training Augmentation** | RandomHorizontalFlip + ColorJitter | None | Removed ALL |
| **Test-Time Augmentation** | 2 views (original + flip) | Single pass | Removed TTA |
| **Transform Consistency** | Different for train/val/test | IDENTICAL | Unified |
| **Loss Function** | CrossEntropyLoss + weights + smoothing | Plain CrossEntropyLoss() | Removed parameters |
| **Class Weights** | Computed from dataset | None | Removed |
| **Label Smoothing** | 0.1 | 0.0 | Removed |
| **Model Architecture** | Standard ViT-B/16 | Standard ViT-B/16 | No change |
| **Learning Rates** | 5e-5 / 1e-4 (2x) | 5e-5 / 1e-4 (2x) | No change |
| **Optimizer** | AdamW | AdamW | No change |
| **Scheduler** | Cosine + Warmup | Cosine + Warmup | No change |
| **Demo Preprocessing** | Missing Resize step | Includes Resize | Fixed |
| **Architecture Detection** | Too broad | Specific to EnhancedViT | Fixed |

---

## Visual Comparison: Data Pipeline

### BEFORE (Complex Pipeline with Augmentation)
```
Raw Image (BGR, variable size)
    ↓
[Training Branch]
    ↓
Resize(224×224) ← Deterministic
    ↓
RandomHorizontalFlip(p=0.5) ← ❌ Random
    ↓
ColorJitter(...) ← ❌ Random
    ↓
ToTensor()
    ↓
Normalize(ImageNet)
    ↓
Model Training

[Evaluation Branch]
    ↓
Resize(224×224)
    ↓
ToTensor()
    ↓
Normalize(ImageNet)
    ↓
[TTA: Original + Flipped] ← ❌ 2 views
    ↓
Average predictions
    ↓
Final prediction
```

### AFTER (Simple Deterministic Pipeline)
```
Raw Image (BGR, variable size)
    ↓
Resize(224×224) ← Deterministic
    ↓
ToTensor() ← Deterministic
    ↓
Normalize(ImageNet) ← Deterministic
    ↓
[SAME for Train/Val/Test/Demo]
    ↓
Single forward pass
    ↓
Final prediction
```

---

## Performance Expectations

### BEFORE (With Augmentation + TTA)
- **Training**: ~85-90% accuracy (with augmentation)
- **Validation**: ~85-90% accuracy (without augmentation)
- **Test (single pass)**: 89-90% ✅ GOOD
- **Test (with TTA)**: 14% ❌ CATASTROPHIC COLLAPSE

**Issue**: TTA corrupted predictions, caused heavy palm class bias

### AFTER (Clean Baseline)
- **Training**: 85-92% accuracy (target)
- **Validation**: 85-92% accuracy (target)
- **Test**: 85-92% accuracy (target)
- **Demo**: 85-92% accuracy (target)

**Goal**: Consistent performance across all evaluation modes

---

## Code Locations

### Files Modified

1. **Notebook**: `notebooks/finetune_vit_hagrid.ipynb`
   - Cell 1 (lines 2-221): Updated header with clean baseline documentation
   - Cell 8 (lines 505-647): Removed all augmentation, made transforms identical
   - Cell 14 (lines 892-982): Removed class weights and label smoothing

2. **Demo Script**: `scripts/demo_gesture_recognition.py`
   - Lines ~266-295: Added Resize to preprocessing, fixed architecture detection

3. **Documentation**: Created new files:
   - `CLEAN_BASELINE_REFACTORING.md`: Complete refactoring explanation
   - `CLEAN_BASELINE_VALIDATION.md`: Validation checklist
   - `CLEAN_BASELINE_BEFORE_AFTER.md`: This file (side-by-side comparison)

---

## Testing the Changes

### 1. Verify Transform Consistency
```python
# In notebook, after running Cell 8
assert str(train_transform) == str(val_transform) == str(test_transform), \
    "Transforms must be IDENTICAL"
print("✓ Transform consistency verified")
```

### 2. Verify Loss Simplicity
```python
# In notebook, after running Cell 14
assert criterion.weight is None, "Loss should have no class weights"
assert criterion.label_smoothing == 0.0, "Loss should have no label smoothing"
print("✓ Loss simplicity verified")
```

### 3. Verify Demo Consistency
```bash
# Run demo script and check logs
python scripts/demo_gesture_recognition.py

# Should see:
# "Detected standard ViT architecture (baseline)"
# "✓ Model loaded successfully"
```

---

## Rollback Instructions

If you need to revert to previous configuration:

1. **Restore augmentation** (Cell 8):
   ```python
   train_transform = transforms.Compose([
       transforms.Resize((224, 224)),
       transforms.RandomHorizontalFlip(p=0.5),
       transforms.ColorJitter(0.2, 0.2, 0.2, 0.05),
       transforms.ToTensor(),
       transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
   ])
   ```

2. **Restore class weights** (Cell 14):
   ```python
   criterion = nn.CrossEntropyLoss(
       weight=class_weights_tensor,
       label_smoothing=0.1
   )
   ```

3. **Restore TTA** (evaluation code):
   ```python
   tta_transforms = [base_transform, flip_transform]
   outputs_list = [model(t(img)) for t in tta_transforms]
   outputs_avg = torch.mean(torch.stack(outputs_list), dim=0)
   ```

---

## Summary

### What Was Removed ❌
- RandomHorizontalFlip augmentation
- ColorJitter augmentation
- Test-time augmentation (TTA)
- Class weights in loss function
- Label smoothing in loss function

### What Was Added ✅
- Transform consistency across train/val/test
- Explicit Resize in demo script
- Improved architecture detection logic
- Comprehensive documentation

### What Stayed the Same ✓
- Model architecture (Standard ViT-B/16)
- Optimizer (AdamW)
- Learning rates (5e-5 / 1e-4)
- Scheduler (Cosine + Warmup)
- Training loop logic
- Evaluation metrics

### Expected Outcome
- **Performance**: 85-92% accuracy
- **Consistency**: Identical across train/val/test/demo
- **Reliability**: No TTA-induced collapse
- **Interpretability**: Know exactly what model sees

---

**Version**: 3.0 - Clean Baseline  
**Status**: Ready for Training  
**Last Updated**: 2025-01-XX  
