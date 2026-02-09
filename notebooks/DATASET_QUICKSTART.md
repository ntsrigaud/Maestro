# Quick Start: Organizing HaGRID Dataset

## Problem

You've downloaded the HaGRID Sample 30K dataset, but it's not in the right format for the semantic learning notebook.

## Solution (2 minutes)

### Step 1: Run the organization script

```bash
cd /home/neil/dev/repos/CSU-Capstone-Project
python scripts/organize_hagrid_dataset.py
```

### Step 2: Wait for completion (~2-3 minutes)

You'll see progress like this:
```
Processing 'call'...
  Total: 1763 images
  Train: 1234 | Val: 264 | Test: 265
  Training: 100%|████████████████████| 1234/1234
...
✓ Dataset successfully organized!
```

### Step 3: Use in semantic notebook

The notebook will automatically detect the organized dataset:
```python
✓ Found dataset at: data/hagrid_30k
  - Train: data/hagrid_30k/train
  - Val: data/hagrid_30k/val
  - Test: data/hagrid_30k/test
  - Classes found: 18
```

## That's it! 🎉

Your dataset is now ready for semantic gesture learning.

---

## Advanced Options

### Preview first (dry run)
```bash
python scripts/organize_hagrid_dataset.py --dry-run
```

### Save disk space (use symlinks)
```bash
python scripts/organize_hagrid_dataset.py --symlink
```
**Warning**: Don't delete the original dataset if using symlinks!

### Custom split ratios
```bash
python scripts/organize_hagrid_dataset.py --train-ratio 0.8 --val-ratio 0.1 --test-ratio 0.1
```

### Help
```bash
python scripts/organize_hagrid_dataset.py --help
```

---

## Troubleshooting

### "Input path does not exist"
Make sure you're in the project root directory:
```bash
cd /home/neil/dev/repos/CSU-Capstone-Project
```

### "No space left on device"
Use symlinks to save space:
```bash
python scripts/organize_hagrid_dataset.py --symlink
```

### More help
See `scripts/ORGANIZE_DATASET_README.md` for detailed documentation.
