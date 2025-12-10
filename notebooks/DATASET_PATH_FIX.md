# Dataset Path Fix

## Issue

When extracting the Kaggle HaGRID dataset, it creates a nested directory structure:

```
data/
└── hagrid-sample-30k-384p/
    └── hagrid-sample-30k-384p/
        ├── ann_train_val/
        └── hagrid_30k/
            ├── train_val_call/
            ├── train_val_dislike/
            ├── train_val_fist/
            └── ... (18 gesture classes total)
```

## Solution Applied

### 1. Updated `DATASET_ROOT` Path

Changed from:

```python
DATASET_ROOT = Path('./data/hagrid_sample_30k_384p')
```

To:

```python
DATASET_ROOT = Path('./data/hagrid-sample-30k-384p/hagrid-sample-30k-384p/hagrid_30k')
```

### 2. Created Custom Dataset Class

The folder names use `train_val_<gesture>` prefix, so I created a custom `HaGRIDDataset` class that:

- Removes the `train_val_` prefix from folder names to get clean class names
- Properly maps images to class labels
- Maintains compatibility with PyTorch's DataLoader

### 3. Fixed Class Name Handling

Original folder names like `train_val_call`, `train_val_peace`, etc. are now properly mapped to clean class names: `call`, `peace`, etc.

## Expected Directory Structure

After downloading from Kaggle and extracting, your structure should match:

```
notebooks/
└── data/
    └── hagrid-sample-30k-384p/
        └── hagrid-sample-30k-384p/
            └── hagrid_30k/
                ├── train_val_call/
                │   ├── 00000000.jpg
                │   └── ...
                ├── train_val_dislike/
                ├── train_val_fist/
                └── ... (more gesture folders)
```

## No Manual Path Changes Required

The notebook now automatically handles the nested structure and folder prefixes. Just extract the Kaggle dataset to `./data/` and run the cells.
