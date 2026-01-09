# Quick Start: Retrain Clean Baseline

## Prerequisites
- ✅ HaGRID Sample 30K dataset downloaded and organized
- ✅ Python environment configured with PyTorch, timm, torchvision
- ✅ CUDA/MPS available (recommended) or CPU fallback
- ✅ Clean baseline changes applied to notebook

## Step 1: Verify Configuration (2 minutes)

Open `notebooks/finetune_vit_hagrid.ipynb` and run these cells:

### Cell 1: Read Header
- Should show "ViT-B/16 Baseline Fine-tuning (No Augmentation)"
- Should document clean baseline rationale

### Cell 2: Import Libraries
```python
%run cell
```
Expected: No errors, all imports successful

### Cell 3: Configuration
```python
%run cell
```
Check output:
- Dataset path points to HaGRID Sample 30K
- Hyperparameters look correct (15 epochs, batch 32, LR 5e-5)

### Cell 8: Data Preprocessing
```python
%run cell
```
**CRITICAL CHECK**:
```python
# After running cell, verify transforms are identical
print("Train:", train_transform)
print("Val:  ", val_transform)
print("Test: ", test_transform)
```

Expected output (ALL THREE IDENTICAL):
```
Compose(
    Resize(size=(224, 224), interpolation=bicubic, ...)
    ToTensor()
    Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
)
```

✅ **PASS**: All three transforms identical  
❌ **FAIL**: Different transforms → Check Cell 8 code

### Cell 14: Training Configuration
```python
%run cell
```
**CRITICAL CHECK**:
```python
# Verify loss function
print(f"Loss: {criterion}")
print(f"Has class weights: {criterion.weight is not None}")
print(f"Label smoothing: {criterion.label_smoothing}")
```

Expected output:
```
Loss: CrossEntropyLoss()
Has class weights: False
Label smoothing: 0.0
```

✅ **PASS**: Plain CrossEntropyLoss with no parameters  
❌ **FAIL**: Has weights or smoothing → Check Cell 14 code

## Step 2: Start Training (2-4 hours)

### Run Training Cell
```python
# Cell 17 (or wherever training loop is)
%run cell
```

### Monitor Progress
Watch for these indicators:

**Epoch 1-2 (Warmup)**:
```
Epoch 1/15
  Train Loss: 2.4xxx | Train Acc: 25-35%
  Val Loss:   2.2xxx | Val Acc:   30-40%
  LR (backbone): 2.5e-05 | LR (head): 5.0e-05
```
- Loss should be ~2.0-2.5 initially
- Accuracy should start 25-40% (better than random 5.5%)
- Learning rates should be half of peak during warmup

**Epoch 5-7 (Mid-training)**:
```
Epoch 5/15
  Train Loss: 0.5xxx | Train Acc: 75-82%
  Val Loss:   0.6xxx | Val Acc:   72-80%
  LR (backbone): 4.5e-05 | LR (head): 9.0e-05
```
- Loss should drop to 0.4-0.8
- Accuracy should reach 70-85%
- Gap between train/val should be small (<5%)

**Epoch 12-15 (Final epochs)**:
```
Epoch 15/15
  Train Loss: 0.2xxx | Train Acc: 90-95%
  Val Loss:   0.3xxx | Val Acc:   85-92%
  LR (backbone): 1.0e-06 | LR (head): 2.0e-06
```
- Loss should plateau around 0.2-0.4
- Accuracy should reach target 85-92%
- Learning rates should decay near minimum

### Healthy Training Signs ✅
- Train loss decreases monotonically
- Val loss tracks train loss (gap <0.5)
- Accuracy improves steadily
- No wild oscillations

### Warning Signs ⚠️
- Val loss diverges from train loss (gap >1.0)
- Accuracy stuck <70% after epoch 10
- Wild oscillations in loss/accuracy
- NaN or inf loss values

## Step 3: Evaluate Results (5 minutes)

### Run Evaluation Cell
```python
# Cell after training
%run cell
```

### Check Key Metrics

**1. Overall Accuracy**:
```python
print(f"Test Accuracy: {test_acc:.2%}")
```
✅ **TARGET**: 85-92%  
⚠️ **ACCEPTABLE**: 80-85% (may need more epochs)  
❌ **FAILURE**: <80% (configuration issue)

**2. Confusion Matrix Diagonality**:
```python
cm_normalized = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
diagonality = cm.diagonal().sum() / cm.sum()
print(f"Diagonality: {diagonality:.2%}")
```
✅ **TARGET**: >85%  
⚠️ **ACCEPTABLE**: >75%  
❌ **FAILURE**: <70% (model not discriminating well)

**3. Per-Class Recall**:
```python
for i, cls in enumerate(class_names):
    recall = cm[i, i] / cm[i].sum() if cm[i].sum() > 0 else 0
    status = "✓" if recall > 0.70 else "⚠️" if recall > 0.50 else "✗"
    print(f"  {status} {cls:20s} Recall: {recall:.1%}")
```
✅ **TARGET**: All classes >70%  
⚠️ **ACCEPTABLE**: Most classes >70%, few >50%  
❌ **FAILURE**: Many classes <50%

**4. Confusion Matrix Visual**:
- Should see strong diagonal (dark squares)
- Off-diagonal mostly light/white
- No single row/column dominates

## Step 4: Save Model (1 minute)

Model should auto-save to:
```
models/gesture_recognition/vit_baseline/best_model.pth
```

Verify files exist:
```bash
ls -lh models/gesture_recognition/vit_baseline/
```

Should see:
- `best_model.pth` (~330 MB)
- `model_metadata.json` (~1 KB)

## Step 5: Test Demo Script (2 minutes)

### Run Demo
```bash
cd /home/rachel/.Neil/CSU-Capstone-Project
python scripts/demo_gesture_recognition.py
```

### Expected Output
```
Initializing gesture recognition demo...
Device: cuda (or mps/cpu)
Loading model from models/gesture_recognition/vit_baseline/best_model.pth...
Detected standard ViT architecture (baseline)
✓ Model loaded successfully (VisionTransformer)
✓ Initialization complete

Controls:
  'q' - Quit
  'c' - Toggle cropped hand view
  's' - Save screenshot
  'r' - Reset statistics
```

### Test Gestures
1. Show your hand to camera
2. Make different gestures (fist, palm, peace, ok, etc.)
3. Check predictions appear and are stable
4. Verify no flickering or random predictions

### Demo Success Criteria ✅
- Model loads without errors
- Hand detection works (bounding boxes appear)
- Gesture predictions appear with confidence scores
- Predictions are stable (not flickering every frame)
- FPS is reasonable (>10 FPS)
- Predictions make sense (match actual gestures)

## Troubleshooting

### Issue: Accuracy <80%

**Check transforms**:
```python
# All three should be IDENTICAL
print(train_transform)
print(val_transform)
print(test_transform)
```

**Check loss**:
```python
# Should have NO parameters
print(criterion)
assert criterion.weight is None
assert criterion.label_smoothing == 0.0
```

**Check dataset**:
```python
# Should have ~30k samples
print(f"Total samples: {len(dataset)}")
print(f"Train: {len(train_dataset)}")
print(f"Val:   {len(val_dataset)}")
print(f"Test:  {len(test_dataset)}")
```

### Issue: Val Loss Diverges

This indicates overfitting. Solutions:
1. Check if augmentation was accidentally enabled
2. Verify transforms are identical for train/val
3. Try training fewer epochs (10 instead of 15)
4. Check for data leakage in train/val split

### Issue: Demo Script Fails

**Architecture mismatch**:
```python
# Check what model was saved
checkpoint = torch.load('best_model.pth')
print(list(checkpoint.keys())[:10])

# Standard ViT: ['head.weight', 'head.bias', 'patch_embed.proj.weight', ...]
# EnhancedViT: ['backbone.patch_embed.proj.weight', 'head.head.0.weight', ...]
```

**Preprocessing mismatch**:
```python
# Demo transform MUST match notebook
# Should include Resize((224, 224))
print(demo.transform)
```

### Issue: Training Too Slow

**Use mixed precision** (if not already enabled):
```python
use_amp = True  # In configuration cell
```

**Reduce batch size** (if OOM):
```python
BATCH_SIZE = 16  # Or 8 if still OOM
```

**Use fewer workers**:
```python
train_loader = DataLoader(..., num_workers=2)  # Reduce from 4
```

## Success Checklist

### Before Training
- [ ] All augmentation removed (checked Cell 8)
- [ ] Transforms identical for train/val/test
- [ ] Loss has no class weights or smoothing
- [ ] Model is standard ViT (not EnhancedViT)
- [ ] Dataset loaded correctly (~30k samples)

### During Training
- [ ] Train loss decreases monotonically
- [ ] Val loss tracks train loss (gap <0.5)
- [ ] Accuracy reaches 85-92% by epoch 15
- [ ] No NaN or inf loss values
- [ ] Learning rates decay smoothly

### After Training
- [ ] Test accuracy: 85-92%
- [ ] Confusion matrix diagonality: >85%
- [ ] All classes recall: >70%
- [ ] Model saved to correct path
- [ ] Demo script runs successfully
- [ ] Real-time predictions are stable

## Next Steps

### If Baseline Succeeds (>90% accuracy)
Consider **horizontal flip ONLY** as first enhancement:
```python
train_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.RandomHorizontalFlip(p=0.5),  # Add this only
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])
```

Validate improvement:
- Must increase accuracy >2%
- Must maintain confusion matrix quality
- Must not introduce class bias

### If Baseline Fails (<80% accuracy)
1. Review preprocessing (most common issue)
2. Check dataset quality (corrupted images?)
3. Verify no data leakage
4. Try training longer (20 epochs)
5. Check hardware (CUDA/MPS working?)

## Timeline Summary

| Phase | Duration | Task |
|-------|----------|------|
| **Verification** | 2 min | Check configuration, transforms, loss |
| **Training** | 2-4 hours | Run training loop (15 epochs) |
| **Evaluation** | 5 min | Check metrics, confusion matrix |
| **Testing** | 2 min | Run demo script, test gestures |
| **Total** | ~2-4 hours | Complete retraining cycle |

## Support

If issues persist:
1. Check `CLEAN_BASELINE_VALIDATION.md` for detailed checklist
2. Review `CLEAN_BASELINE_BEFORE_AFTER.md` for configuration comparison
3. Check training logs for error messages
4. Verify dataset integrity

---

**Status**: Ready to Train  
**Configuration**: Clean Baseline (No Augmentation)  
**Expected Result**: 85-92% accuracy with stable performance  
**Last Updated**: 2025-01-XX  
