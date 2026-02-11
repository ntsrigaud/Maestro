# HaGRID Download Solutions - Size Comparison

## Problem

The original download script (`download_hagrid_subset.py`) downloads **entire 40GB ZIP files** per gesture class, then extracts only 800 samples. This is extremely inefficient!

## ✅ Recommended Solutions (Small Storage)

### Option 1: Kaggle Pre-made Subset (FASTEST) ⭐

**Size**: ~1.5 GB  
**Images**: 30,000 (all 18 classes)  
**Time**: 5-15 minutes

```bash
# Install Kaggle CLI
pip install kaggle

# Setup API key (one-time)
# 1. Go to https://www.kaggle.com/account
# 2. Click "Create New API Token"
# 3. Save kaggle.json to ~/.kaggle/

# Download
python scripts/download_hagrid_kaggle.py --save_path ./data/hagrid_subset

# Filter to your gestures (optional)
python scripts/download_hagrid_kaggle.py \
    --gestures fist palm peace like dislike \
    --samples_per_class 500
```

---

### Option 2: Even Smaller Kaggle Subset

**Size**: ~500 MB  
**Images**: ~10,000  
**Time**: 2-5 minutes

```bash
# Manual download:
# 1. Visit: https://www.kaggle.com/datasets/innominate817/hagrid-classification-512p
# 2. Click "Download" (requires Kaggle account)
# 3. Extract to ./data/hagrid_subset/
```

---

### Option 3: Ultra-Minimal DIY Dataset

**Size**: 100-300 MB  
**Images**: 1,000-3,000 (5-10 classes)  
**Time**: Manual collection

Create your own small dataset:

1. Use your webcam to capture 100-200 samples per gesture
2. Or find small gesture datasets on Roboflow/Kaggle
3. Organize as:
   ```
   data/hagrid_subset/
   ├── train/
   │   ├── fist/
   │   ├── palm/
   │   └── peace/
   └── annotations.csv
   ```

---

## Size Comparison Table

| Method              | Size        | Images      | Time          | Storage      |
| ------------------- | ----------- | ----------- | ------------- | ------------ |
| **Original script** | **~280 GB** | 8,000       | **4-8 hours** | ❌ Too large |
| Kaggle 30K subset   | 1.5 GB      | 30,000      | 5-15 min      | ✅ Good      |
| Kaggle 10K subset   | 500 MB      | 10,000      | 2-5 min       | ✅ Best      |
| DIY mini dataset    | 100-300 MB  | 1,000-3,000 | Varies        | ✅ Minimal   |

---

## Quick Start (Recommended)

### For Limited Storage (<2 GB available):

```bash
# Install dependencies
pip install kaggle pandas tqdm pillow

# Download Kaggle subset (1.5 GB)
python scripts/download_hagrid_kaggle.py \
    --save_path ./data/hagrid_subset \
    --gestures fist palm peace like dislike stop ok call \
    --samples_per_class 400

# Expected output: ~3,200 images, ~600 MB after filtering
```

### For Minimal Storage (<500 MB available):

```bash
# Download smaller subset manually:
# 1. Go to: https://www.kaggle.com/datasets/innominate817/hagrid-classification-512p
# 2. Download (requires Kaggle account)
# 3. Extract to ./data/hagrid_subset/

# Or use even fewer samples:
python scripts/download_hagrid_kaggle.py \
    --gestures fist palm peace like dislike \
    --samples_per_class 200

# Expected: ~1,000 images, ~150-200 MB
```

---

## Why the Original Script Was Inefficient

```
Original approach:
1. Download call.zip (37 GB) ❌
2. Download dislike.zip (40 GB) ❌
3. ... (repeat for 10 gestures) ❌
4. Extract only 800 samples from each ❌
5. Delete ZIPs ❌
Total: Downloads 280+ GB, keeps 2.5 GB

Better approach (Kaggle):
1. Download pre-filtered subset (1.5 GB) ✅
2. Filter to your gestures (600 MB) ✅
Total: Downloads 1.5 GB, keeps 600 MB
```

---

## Training Implications

| Dataset Size | Samples/Class | Expected Accuracy | Training Time |
| ------------ | ------------- | ----------------- | ------------- |
| 8,000 images | 800           | 88-92%            | 30-60 min     |
| 4,000 images | 400           | 85-89%            | 15-30 min     |
| 2,000 images | 200           | 80-85%            | 10-20 min     |
| 1,000 images | 100           | 75-80%            | 5-10 min      |

**Recommendation**: Start with 200-400 samples/class for prototyping. Expand later if needed.

---

## Updated Workflow

```bash
# 1. Setup environment
conda env create -f environment.yml
conda activate gesture

# 2. Download Kaggle subset (FAST METHOD)
pip install kaggle
python scripts/download_haggle_kaggle.py --save_path ./data/hagrid_subset

# 3. Update notebook
# In finetune_vit_hagrid.ipynb, cell 5:
DATASET_ROOT = './data/hagrid_subset/filtered'  # or './data/hagrid_subset'

# 4. Train
jupyter notebook notebooks/finetune_vit_hagrid.ipynb
```

---

## Troubleshooting

### "I don't have a Kaggle account"

- Sign up for free at kaggle.com (takes 2 minutes)
- Or use manual webcam collection method

### "Kaggle download is still too slow"

- Try the 500 MB subset instead of 1.5 GB
- Or reduce `samples_per_class` to 100-200

### "I need the full quality images"

- The Kaggle subsets use 384px or 512px (good enough for training)
- Full resolution is overkill for ViT-224×224 input size

---

## Next Steps

1. **Use Kaggle method** (`download_hagrid_kaggle.py`) for quick start
2. Train on the smaller dataset first
3. If accuracy is insufficient, expand with more samples
4. The model can be fine-tuned incrementally as you add more data
