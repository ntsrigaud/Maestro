# Clean Baseline Reset

**Date:** 2026-01-09  
**Action:** Full pipeline reset after evaluation collapse  
**Status:** Ready for training

---

## Problem

ViT gesture recognition accuracy collapsed from 89% to 11-12% with single-class dominance. Root causes:
1. Preprocessing mismatch between training and inference
2. Test-time augmentation (TTA) corrupted predictions
3. Data augmentation destroyed spatial gesture features

---

## Solution

Complete reset to minimal, correct baseline:

**Removed:**
- All data augmentation
- Test-time augmentation
- Complex classification heads
- Experimental losses (Focal Loss)
- Label smoothing and class weighting

**Implemented:**
- Standard ViT-B/16 + single linear head (768 → 18)
- Single deterministic transform (Resize → ToTensor → Normalize)
- CrossEntropyLoss (standard)
- Identical preprocessing for train/val/test/demo

---

## Files

**Training:**
- `notebooks/finetune_vit_hagrid.ipynb` - Complete training pipeline

**Inference:**
- `scripts/demo_gesture_recognition.py` - Real-time demo with auto-download

**Upload:**
- `scripts/upload_model_to_huggingface.py` - Upload trained model to HF Hub

**Verification:**
- `scripts/verify_clean_baseline.py` - Pre-training sanity checks

---

## Quick Start

```bash
# Verify setup
python scripts/verify_clean_baseline.py

# Train (30-45 min on GPU)
jupyter notebook notebooks/finetune_vit_hagrid.ipynb

# Upload to Hugging Face
python scripts/upload_model_to_huggingface.py --repo-id username/repo-name

# Demo (auto-downloads from HF if needed)
python scripts/demo_gesture_recognition.py
# Or specify checkpoint
python scripts/demo_gesture_recognition.py --checkpoint checkpoints/best_model.pth
# Or download from HF
python scripts/demo_gesture_recognition.py --hf-repo username/repo-name
```

---

## Expected Results

- **Test accuracy:** 85-92%
- **Confusion matrix:** Clear diagonal, no single-class collapse
- **Demo:** Matches test accuracy, stable predictions

---

## Architecture

```
Camera → Hand Detection → ViT-B/16 → 18-class logits
```

**Preprocessing (all splits):**
```python
transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])
```

**Training:**
- Optimizer: AdamW (lr=1e-4, wd=0.01)
- Scheduler: CosineAnnealingLR
- Loss: CrossEntropyLoss
- Batch size: 32
- Epochs: 15

---

## Rules

1. **Do not modify baseline** except for correctness bugs
2. **All future work branches** from this clean state
3. **Add ONE change at a time**, compare to baseline
4. **No TTA or complex augmentation** until baseline >90%

---

## Dataset

[HaGRID Sample 30K (384p)](https://www.kaggle.com/datasets/innominate817/hagrid-sample-30k-384p)

Extract to: `./data/hagrid-sample-30k-384p/hagrid_30k/`

**Classes (18, alphabetically sorted):**
call, dislike, fist, four, like, mute, ok, one, palm, peace, peace_inverted, rock, stop, stop_inverted, three, three2, two_up, two_up_inverted

---

## Troubleshooting

**Low accuracy (<70%):**
- Run `verify_clean_baseline.py` to check setup
- Verify dataset has all 18 classes
- Check training curves for stability
- Inspect confusion matrix for patterns

**Demo issues:**
- Ensure checkpoint path is correct
- Verify preprocessing matches training
- Check camera access permissions

---

This baseline restores correctness and interpretability. All future enhancements must maintain preprocessing consistency and add changes incrementally.
