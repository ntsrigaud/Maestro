# HaGRID Subset Download Guide

## Overview

This guide explains how to download and prepare a manageable subset of the HaGRID dataset for gesture recognition training.

## Quick Start

### Step 1: Install Dependencies

```bash
conda create -n hagrid python=3.11 -y
conda activate hagrid
pip install pandas tqdm pillow
```

### Step 2: Download HaGRID Subset

Run the download script to fetch annotations and a subset of gesture images:

```bash
python scripts/download_hagrid_subset.py \
    --save_path ./data/hagrid_subset \
    --gestures call dislike fist like ok palm peace point rock no_gesture \
    --samples_per_class 800 \
    --download_annotations
```

**Parameters:**

- `--save_path`: Directory to save the subset dataset (default: `./data/hagrid_subset`)
- `--gestures`: List of gesture classes to include (default: 10 recommended gestures)
- `--samples_per_class`: Number of images per class distributed across train/val/test (default: 800)
- `--download_annotations`: Download annotations (required for first run)
- `--seed`: Random seed for reproducibility (default: 42)

### Step 3: Verify Dataset

After download completes, verify the structure:

```bash
tree data/hagrid_subset -L 3
```

Expected structure:

```
data/hagrid_subset/
├── annotations/
│   ├── train/
│   │   ├── call.json
│   │   ├── dislike.json
│   │   └── ...
│   ├── val/
│   └── test/
├── train/
│   ├── call/
│   │   ├── 00000001.jpg
│   │   └── ...
│   ├── dislike/
│   └── ...
├── val/
│   └── ...
├── test/
│   └── ...
├── annotations.csv
├── call.zip
├── dislike.zip
└── ...
```

The `annotations.csv` file contains:

- `image_path`: Absolute path to image
- `label`: Gesture class name
- `split`: train/val/test

## Available Gestures

From the HaGRID dataset, we recommend these 10 core gestures (from the screenshot):

1. **call** - Phone gesture (thumb and pinky extended)
2. **dislike** - Thumbs down
3. **fist** - Closed fist
4. **like** - Thumbs up
5. **ok** - OK sign (thumb and index finger touching)
6. **palm** - Open palm
7. **peace** - Peace/victory sign (index and middle finger up)
8. **point** - Pointing gesture
9. **rock** - Rock-on sign (index and pinky extended)
10. **no_gesture** - Background/no gesture class

### Additional Gestures (Optional)

You can add more gestures by including them in the `--gestures` argument:

- `four` - Four fingers up
- `mute` - Quiet/shh gesture
- `one` - One finger up
- `peace_inverted` - Peace sign palm facing inward
- `stop` - Stop gesture (palm forward)
- `stop_inverted` - Stop gesture palm backward
- `three` - Three fingers up
- `three2` - Alternative three fingers
- `two_up` - Two fingers up (peace)
- `two_up_inverted` - Two fingers inverted
- `grabbing` - Grabbing gesture
- `grip` - Grip gesture
- `middle_finger` - Middle finger gesture
- `three3` - Third variant of three fingers

## Storage Requirements

### Estimated Sizes

| Configuration              | Total Images | Estimated Size |
| -------------------------- | ------------ | -------------- |
| 10 gestures × 800 samples  | ~8,000       | 2.5 GB         |
| 10 gestures × 1000 samples | ~10,000      | 3.0 GB         |
| 15 gestures × 800 samples  | ~12,000      | 3.8 GB         |
| 24 gestures × 500 samples  | ~12,000      | 3.5 GB         |

**Note**: Full gesture archives (before subsetting) are 37-54 GB each, but the script extracts only the sampled images.

## Customization

### Adjust Sample Count

To reduce dataset size further:

```bash
python scripts/download_hagrid_subset.py \
    --samples_per_class 500 \
    --gestures call fist like ok palm peace point rock no_gesture
```

### Select Different Gestures

For a different set of gestures:

```bash
python scripts/download_hagrid_subset.py \
    --gestures one two_up three four palm stop \
    --samples_per_class 1000
```

### Split Ratios

The script maintains HaGRID's original split ratios:

- **Train**: 76% (~608 images if samples_per_class=800)
- **Val**: 9% (~72 images)
- **Test**: 15% (~120 images)

## Troubleshooting

### Download Failures

If a gesture download fails:

1. Check your internet connection
2. Retry the script (it will skip already-downloaded files)
3. Remove partial ZIPs: `rm data/hagrid_subset/*.zip`

### Disk Space Issues

If running low on disk space:

- Delete the ZIP files after extraction: `rm data/hagrid_subset/*.zip`
- Reduce `samples_per_class` to 400-500
- Select fewer gesture classes

### Slow Downloads

- Downloads are large (37-54 GB per gesture before extraction)
- Use a wired connection if possible
- Run overnight for large subsets
- Consider using the 512px version (future enhancement)

## Next Steps

After downloading the dataset:

1. **Verify the data**:

   ```bash
   python -c "import pandas as pd; df = pd.read_csv('data/hagrid_subset/annotations.csv'); print(df['label'].value_counts())"
   ```

2. **Open the training notebook**:

   ```bash
   jupyter notebook notebooks/finetune_vit_hagrid.ipynb
   ```

3. **Update `DATASET_ROOT`** in the notebook to point to your subset:

   ```python
   DATASET_ROOT = './data/hagrid_subset'
   ```

4. **Run training cells** to fine-tune ViT on your subset

## Dataset Statistics

The HaGRID v2-1M dataset:

- **Total**: 1,086,158 images (1.5TB)
- **Resolution**: FullHD (1920×1080)
- **Classes**: 33 gesture classes + no_gesture
- **Subjects**: 65,977 unique persons
- **Distance**: 0.5-4 meters from camera
- **Annotations**: COCO format bounding boxes, hand landmarks (MediaPipe), metadata

## License

HaGRID is licensed under **Creative Commons Attribution-ShareAlike 4.0 International (CC BY-SA 4.0)**.

See: https://github.com/hukenovs/hagrid/blob/master/license/en_us.pdf

## References

- **Repository**: https://github.com/hukenovs/hagrid
- **Paper**: https://arxiv.org/abs/2412.01508
- **Assessment Doc**: `docs/HaGRID_DATASET_ASSESSMENT.md`
