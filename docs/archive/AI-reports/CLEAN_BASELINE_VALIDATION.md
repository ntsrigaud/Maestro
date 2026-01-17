# Clean Baseline Validation Checklist

## Pre-Training Verification

### 1. Notebook Configuration
- [ ] Cell 1: Header updated with clean baseline rationale
- [ ] Cell 8: All augmentation removed
  - [ ] `RandomHorizontalFlip` removed
  - [ ] `ColorJitter` removed
  - [ ] Test-time augmentation (TTA) disabled
  - [ ] `train_transform == val_transform == test_transform`
- [ ] Cell 14: Training config simplified
  - [ ] `CrossEntropyLoss()` with NO arguments
  - [ ] No class weights
  - [ ] No label smoothing
  - [ ] Model: Standard ViT-B/16 (not EnhancedViT)

### 2. Transform Consistency Check
Run this code in notebook to verify:
```python
print("="*70)
print("TRANSFORM CONSISTENCY CHECK")
print("="*70)
print("\nTrain transform:")
print(train_transform)
print("\nVal transform:")
print(val_transform)
print("\nTest transform:")
print(test_transform)
print("\n✓ All transforms should be IDENTICAL")
print("="*70)
```

Expected output:
```
Compose(
    Resize(size=(224, 224), interpolation=bicubic, ...)
    ToTensor()
    Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
)
```

### 3. Loss Function Verification
Run this code to verify loss has no parameters:
```python
print("="*70)
print("LOSS FUNCTION CHECK")
print("="*70)
print(f"Loss type: {type(criterion).__name__}")
print(f"Loss: {criterion}")
print("\n✓ Should show: CrossEntropyLoss()")
print("✗ Should NOT show: weight=..., label_smoothing=...")
print("="*70)
```

### 4. Model Architecture Check
Run this to verify standard ViT (not enhanced):
```python
print("="*70)
print("MODEL ARCHITECTURE CHECK")
print("="*70)
print(f"Model type: {type(model).__name__}")
print(f"Number of parameters: {sum(p.numel() for p in model.parameters()):,}")

# Check head structure
if hasattr(model, 'head'):
    print(f"\nHead type: {type(model.head).__name__}")
    if isinstance(model.head, nn.Linear):
        print(f"✓ Standard head: Linear({model.head.in_features} → {model.head.out_features})")
    else:
        print(f"⚠️  Enhanced head detected: {model.head}")
else:
    print("⚠️  No 'head' attribute found")
print("="*70)
```

Expected output:
```
Model type: VisionTransformer
Number of parameters: 86,415,890
Head type: Linear
✓ Standard head: Linear(768 → 18)
```

## During Training

### 5. Training Stability Checks
Monitor these metrics during training:

```python
# After each epoch, check:
print(f"Epoch {epoch+1}/{EPOCHS}")
print(f"  Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.2%}")
print(f"  Val Loss:   {val_loss:.4f} | Val Acc:   {val_acc:.2%}")
print(f"  LR (backbone): {optimizer.param_groups[0]['lr']:.2e}")
print(f"  LR (head):     {optimizer.param_groups[1]['lr']:.2e}")
```

Healthy signs:
- [ ] Train loss decreases monotonically
- [ ] Val loss decreases (with minor fluctuations ok)
- [ ] Val loss tracks train loss (gap <0.5)
- [ ] Train accuracy increases steadily
- [ ] Val accuracy increases (target: 85-92%)
- [ ] Learning rates decay smoothly

Warning signs:
- [ ] Val loss diverges from train loss (>1.0 gap)
- [ ] Val accuracy stagnates or decreases
- [ ] Train loss oscillates wildly
- [ ] Accuracy <70% after epoch 10

## Post-Training Evaluation

### 6. Final Metrics Check
```python
print("="*70)
print("FINAL PERFORMANCE METRICS")
print("="*70)
print(f"Test Accuracy: {test_acc:.2%}")
print(f"Test F1 Score: {test_f1:.2%}")
print(f"Confusion Matrix Diagonality: {cm.diagonal().sum()/cm.sum():.2%}")
print("\n✓ Target Metrics:")
print("  • Test Accuracy: 85-92%")
print("  • Confusion Matrix Diagonality: >85%")
print("  • Per-class Recall: >70% for all classes")
print("="*70)
```

Success criteria:
- [ ] Test accuracy: 85-92%
- [ ] Confusion matrix diagonal: >85%
- [ ] Per-class recall: >70% for all 18 classes
- [ ] No single class dominates (<30% of total predictions)

### 7. Confusion Matrix Analysis
```python
# Check for problematic patterns
print("\nTop 5 Confusion Pairs:")
conf_pairs = []
for i in range(NUM_CLASSES):
    for j in range(NUM_CLASSES):
        if i != j:
            conf_rate = cm[i, j] / cm[i].sum() if cm[i].sum() > 0 else 0
            if conf_rate > 0.1:  # >10% confusion
                conf_pairs.append((class_names[i], class_names[j], conf_rate))

conf_pairs.sort(key=lambda x: x[2], reverse=True)
for true_cls, pred_cls, rate in conf_pairs[:5]:
    print(f"  {true_cls} → {pred_cls}: {rate:.1%}")
```

Healthy signs:
- [ ] Strong diagonal (dark squares)
- [ ] Off-diagonal mostly white/light
- [ ] Confusion makes semantic sense (e.g., 'peace' ↔ 'peace_inverted')

Warning signs:
- [ ] Single column/row dominates (class bias)
- [ ] Weak diagonal (poor discrimination)
- [ ] Random confusion pattern (model not learning)

### 8. Per-Class Performance
```python
print("\nPer-Class Metrics:")
for i, cls in enumerate(class_names):
    recall = cm[i, i] / cm[i].sum() if cm[i].sum() > 0 else 0
    precision = cm[i, i] / cm[:, i].sum() if cm[:, i].sum() > 0 else 0
    f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
    
    status = "✓" if recall > 0.70 else "⚠️"
    print(f"  {status} {cls:20s} Recall: {recall:.1%} | Precision: {precision:.1%} | F1: {f1:.2%}")
```

Check:
- [ ] All classes >70% recall
- [ ] No class <50% recall (failure mode)
- [ ] Precision and recall balanced (±10%)

## Demo Script Validation

### 9. Preprocessing Consistency
In `demo_gesture_recognition.py`, verify:

```python
# Check transform definition (around line 289)
self.transform = transforms.Compose([
    transforms.Resize((224, 224)),  # ✓ Must be present
    transforms.ToTensor(),          # ✓ Must be present
    transforms.Normalize(           # ✓ Must match notebook
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])
```

- [ ] `Resize((224, 224))` present
- [ ] `ToTensor()` present
- [ ] `Normalize` with correct ImageNet stats
- [ ] NO augmentation (no flips, no color jitter)

### 10. Architecture Detection
Check architecture detection logic (around line 275):

```python
# Correct detection logic
is_enhanced_vit = any(key.startswith('head.head.') for key in state_dict_keys)

if is_enhanced_vit:
    # Load EnhancedViT
    ...
else:
    # Load standard ViT (for clean baseline)
    ...
```

- [ ] Detection uses `'head.head.'` (not `'backbone.'` or `'head.'`)
- [ ] Standard ViT branch loads correctly
- [ ] Model type logged correctly

### 11. Real-Time Demo Test
Run the demo script:
```bash
python scripts/demo_gesture_recognition.py
```

Check:
- [ ] Demo starts without errors
- [ ] Model loads successfully (logs show "standard ViT architecture")
- [ ] Hand detection works (bounding boxes appear)
- [ ] Gesture predictions appear
- [ ] Predictions are stable (not flickering)
- [ ] Predictions match training results
- [ ] FPS is reasonable (>10 FPS)

## Checklist Summary

### Pre-Training (Must Complete Before Training)
- [ ] All augmentation removed from notebook
- [ ] Train/val/test transforms identical
- [ ] Loss function has no parameters
- [ ] Model architecture is standard ViT
- [ ] Demo script preprocessing matches notebook

### During Training (Monitor Throughout)
- [ ] Train/val loss decrease monotonically
- [ ] Val loss tracks train loss (gap <0.5)
- [ ] Accuracy improves to 85-92%
- [ ] Learning rates decay smoothly

### Post-Training (Must Pass Before Deployment)
- [ ] Test accuracy 85-92%
- [ ] Confusion matrix diagonality >85%
- [ ] All classes recall >70%
- [ ] No single class dominates
- [ ] Demo script runs successfully
- [ ] Real-time predictions are stable

## Troubleshooting

### If Training Fails (Accuracy <80%)
1. Check transforms are truly identical (print all three)
2. Verify loss has no parameters (print criterion)
3. Check dataset is loaded correctly (print sample counts)
4. Verify model is standard ViT (print model.head)
5. Check learning rates are not too high/low

### If Demo Script Fails
1. Check preprocessing matches notebook exactly
2. Verify architecture detection logic
3. Ensure model checkpoint is from clean baseline
4. Test with single image first (not real-time)
5. Check device compatibility (CUDA/MPS/CPU)

### If Performance Degrades Over Time
1. Check for overfitting (train vs val accuracy gap)
2. Verify no data leakage (train/val/test splits)
3. Check for label noise (inspect misclassified samples)
4. Consider early stopping (save best model)

---

**Status**: Ready for Validation  
**Version**: 3.0 - Clean Baseline  
**Last Updated**: 2025-01-XX  
