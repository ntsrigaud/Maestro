# HaGRID Dataset Organization Script

## Overview

The `organize_hagrid_dataset.py` script reorganizes the HaGRID Sample 30K dataset from Kaggle's format into the standard train/val/test directory structure required by the semantic gesture learning notebook.

## Problem

The HaGRID Sample 30K dataset downloaded from Kaggle uses this structure:
```
hagrid-sample-30k-384p/
  hagrid-sample-30k-384p/
    ann_train_val/        # JSON annotation files
      call.json
      dislike.json
      ...
    hagrid_30k/
      train_val_call/     # All images mixed (train + val)
      train_val_dislike/
      ...
```

But the semantic learning notebook expects:
```
data/
  hagrid_30k/
    train/
      call/
      dislike/
      ...
    val/
      call/
      dislike/
      ...
    test/
      call/
      dislike/
      ...
```

## Usage

### Basic Usage (Default Settings)

```bash
# From project root directory
python scripts/organize_hagrid_dataset.py
```

This will:
- Read from `./notebooks/hagrid-sample-30k-384p/hagrid-sample-30k-384p`
- Create organized dataset at `./notebooks/data/hagrid_30k`
- Split data: 70% train, 15% val, 15% test
- Copy all image files

### Dry Run (Preview Without Changes)

```bash
python scripts/organize_hagrid_dataset.py --dry-run
```

See what would happen without actually copying files.

### Custom Paths

```bash
python scripts/organize_hagrid_dataset.py \
  --input /path/to/hagrid-sample-30k-384p \
  --output /path/to/output/data
```

### Custom Split Ratios

```bash
python scripts/organize_hagrid_dataset.py \
  --train-ratio 0.8 \
  --val-ratio 0.1 \
  --test-ratio 0.1
```

### Use Symbolic Links (Save Disk Space)

```bash
python scripts/organize_hagrid_dataset.py --symlink
```

Instead of copying files, create symbolic links. **Warning**: Don't delete the original dataset if using this option!

### All Options

```bash
python scripts/organize_hagrid_dataset.py --help
```

## Options Reference

| Option | Default | Description |
|--------|---------|-------------|
| `--input` | `./notebooks/hagrid-sample-30k-384p/hagrid-sample-30k-384p` | Path to downloaded dataset |
| `--output` | `./notebooks/data` | Output directory (will create `hagrid_30k` subdirectory) |
| `--train-ratio` | `0.7` | Proportion of data for training (0.0-1.0) |
| `--val-ratio` | `0.15` | Proportion of data for validation (0.0-1.0) |
| `--test-ratio` | `0.15` | Proportion of data for testing (0.0-1.0) |
| `--seed` | `42` | Random seed for reproducible splits |
| `--symlink` | `False` | Create symbolic links instead of copying |
| `--dry-run` | `False` | Preview changes without modifying filesystem |

## Expected Output

```
======================================================================
HAGRID DATASET ORGANIZATION
======================================================================
Input path:  notebooks/hagrid-sample-30k-384p/hagrid-sample-30k-384p
Output path: notebooks/data/hagrid_30k
Split ratios: Train=70%, Val=15%, Test=15%
Method: Copy files
Dry run: No
======================================================================

✓ Found 18 gesture classes:
  call, dislike, fist, four, like, mute, ok, one, palm, peace, 
  peace_inverted, rock, stop, stop_inverted, three, three2, 
  two_up, two_up_inverted

======================================================================
ORGANIZING FILES
======================================================================

Processing 'call'...
  Total: 1763 images
  Train: 1234 | Val: 264 | Test: 265
  Training: 100%|████████████████████| 1234/1234
  Validation: 100%|██████████████████| 264/264
  Testing: 100%|██████████████████████| 265/265

[... continues for all 18 classes ...]

======================================================================
SUMMARY
======================================================================
Total images organized:
  Train: 22,276 images (70%)
  Val:   4,766 images (15%)
  Test:  4,791 images (15%)
  Total: 31,833 images

✓ Dataset successfully organized!
✓ Output location: notebooks/data/hagrid_30k

You can now use this dataset with semantic_gesture_learning.ipynb
The notebook will automatically detect the dataset at: notebooks/data/hagrid_30k
======================================================================
```

## Workflow

1. **Download dataset** (if not already done):
   ```bash
   # In semantic_gesture_learning.ipynb, the auto-download will get the raw dataset
   ```

2. **Run organization script**:
   ```bash
   python scripts/organize_hagrid_dataset.py --dry-run  # Preview
   python scripts/organize_hagrid_dataset.py             # Actually organize
   ```

3. **Use in notebook**:
   - Open `semantic_gesture_learning.ipynb`
   - Run the dataset detection cell
   - It will automatically find the organized dataset at `./data/hagrid_30k`

## Disk Space

- **Copy mode** (default): Requires ~3GB additional space
  - Original: ~1.5GB
  - Organized copy: ~1.5GB
  - Total: ~3GB

- **Symlink mode** (`--symlink`): No additional space
  - Uses symbolic links pointing to original files
  - **Warning**: Don't delete original dataset!

## Troubleshooting

### Error: Input path does not exist
```
❌ Error: Input path does not exist: notebooks/hagrid-sample-30k-384p/hagrid-sample-30k-384p
```

**Solution**: Specify correct path with `--input` option, or download the dataset first.

### Error: hagrid_30k directory not found
```
❌ Error: hagrid_30k directory not found in [path]
```

**Solution**: Make sure you're pointing to the correct extracted dataset directory.

### Error: Split ratios must sum to 1.0
```
❌ Error: Split ratios must sum to 1.0 (got 0.95)
```

**Solution**: Ensure `--train-ratio`, `--val-ratio`, and `--test-ratio` add up to exactly 1.0.

### Permission denied errors

**Solution**: 
- Check write permissions for output directory
- Run with appropriate permissions
- On Linux/Mac: `chmod +x scripts/organize_hagrid_dataset.py`

## Notes

- The script uses a fixed random seed (42) by default for reproducible splits
- Files are split per gesture class to maintain balanced class distribution
- Original dataset is never modified - only read from
- The script creates all necessary directories automatically
- Progress bars show organization progress for each split

## See Also

- `semantic_gesture_learning.ipynb` - Main semantic learning notebook
- `SEMANTIC_GESTURE_RESEARCH.md` - Theory and methodology
- `gesture_semantics.py` - Text description mappings
