# HaGRID Dataset Assessment

## Overview

**HaGRID v2-1M** (HAnd Gesture Recognition Image Dataset) is a large-scale dataset for hand gesture recognition.

- **Total Size**: 1.5TB (full dataset)
- **Images**: 1,086,158 FullHD RGB images
- **Classes**: 33 gesture classes + 1 `no_gesture` class
- **Unique Persons**: 65,977 subjects (18+ years old)
- **Split**: Train 76% (821,458) / Val 9% (99,200) / Test 15% (165,500)
- **Distance**: 0.5-4 meters from camera
- **Lighting**: Indoor with natural and artificial light, including extreme conditions
- **Annotations**: COCO format bounding boxes, hand landmarks (MediaPipe), metadata (age, gender, race)
- **License**: Creative Commons Attribution-ShareAlike 4.0 International

## Dataset Structure

### Storage Format

The dataset is split into **34 separate archives by gesture class**, enabling selective download:

```
hagrid/
├── hagrid_dataset/
│   ├── call/
│   │   ├── 00000000.jpg
│   │   ├── 00000001.jpg
│   │   └── ...
│   ├── dislike/
│   ├── fist/
│   └── ...
└── hagrid_annotations/
    ├── train/
    │   ├── call.json
    │   └── ...
    ├── val/
    └── test/
```

### Per-Class Sizes (Relevant for Our Subset)

| Gesture             | Size    | Notes              |
| ------------------- | ------- | ------------------ |
| **call**            | 37.2 GB | ✓ In screenshot    |
| **dislike**         | 40.9 GB | ✓ In screenshot    |
| **fist**            | 42.3 GB | ✓ In screenshot    |
| **four**            | 43.1 GB | ✓ In screenshot    |
| **like**            | 42.2 GB | ✓ In screenshot    |
| **mute**            | 43.2 GB | ✓ In screenshot    |
| **ok**              | 42.5 GB | ✓ In screenshot    |
| **one**             | 42.7 GB | ✓ In screenshot    |
| **palm**            | 43.0 GB | ✓ In screenshot    |
| **peace**           | 41.4 GB | ✓ In screenshot    |
| **peace_inverted**  | 40.5 GB | ✓ In screenshot    |
| **rock**            | 41.7 GB | ✓ In screenshot    |
| **stop**            | 41.8 GB | ✓ In screenshot    |
| **stop_inverted**   | 41.4 GB | ✓ In screenshot    |
| **three**           | 42.2 GB | ✓ In screenshot    |
| **three2**          | 40.2 GB | ✓ In screenshot    |
| **two_up**          | 41.8 GB | ✓ In screenshot    |
| **two_up_inverted** | 40.9 GB | ✓ In screenshot    |
| **grabbing**        | 48.7 GB | ✓ In screenshot    |
| **grip**            | 48.6 GB | ✓ In screenshot    |
| **point**           | 50.4 GB | ✓ In screenshot    |
| **middle_finger**   | 50.5 GB | ✓ In screenshot    |
| **three3**          | 54.0 GB | ✓ In screenshot    |
| **no_gesture**      | 0.49 GB | ✓ Background class |

**Total for all 24 classes from screenshot: ~1,019 GB** ❌ (exceeds 1-3GB target)

## Solution: Subset Strategy

### Option 1: Download Lightweight 512px Version (RECOMMENDED)

HaGRID provides a **pre-resized 512px version** of the entire dataset:

- **Total size**: 119.4 GB for ALL 33 classes
- **Download URL**: https://rndml-team-cv.obs.ru-moscow-1.hc.sbercloud.ru/datasets/hagrid_v2/hagridv2_512.zip
- **Pro**: Much smaller, faster download, still FullHD quality downsampled to 512px min side
- **Con**: Still ~120GB for all classes, need to extract only target classes

### Option 2: Selective Download + Sample Subset (CHOSEN)

Since even the 512px version is too large, we'll:

1. **Download only annotations** (tiny, ~few MB)
2. **Use the official `download.py` script** to selectively download **specific gesture classes**
3. **Extract a random subset** of images from each class to stay under 1-3GB total

#### Recommended Subset Configuration

To stay under **3GB**, we can select:

- **8-10 core gestures** from the screenshot (most useful for real-time control)
- **~500-1000 images per class** (train+val+test combined)
- This gives us a **balanced, manageable dataset** for initial training

**Suggested core gestures (10 classes, ~2-3GB total):**

1. `call` (phone gesture)
2. `dislike` (thumbs down)
3. `fist` (closed fist)
4. `like` (thumbs up)
5. `ok` (OK sign)
6. `palm` (open palm / stop)
7. `peace` (peace sign)
8. `point` (pointing)
9. `rock` (rock-on sign)
10. `no_gesture` (background)

### Option 3: Use HuggingFace Datasets (ALTERNATIVE)

Check if HaGRID is available on HuggingFace Hub with streaming support to avoid full download.

## Download Strategy

### Step 1: Clone Repository and Setup

```bash
git clone https://github.com/hukenovs/hagrid.git
cd hagrid
conda create -n hagrid python=3.11 -y
conda activate hagrid
pip install -r requirements.txt
```

### Step 2: Download Annotations

```bash
python download.py --annotations --save_path ../hagrid_data
```

### Step 3: Selective Gesture Download (Full)

```bash
# Download only specific gestures (WARNING: still large!)
python download.py --dataset \
  --targets call dislike fist like ok palm peace point rock no_gesture \
  --save_path ../hagrid_data
```

### Step 4: Extract Subset (Custom Script)

We'll create a custom Python script to:

1. Parse annotations JSON files
2. Randomly sample N images per class (stratified by train/val/test)
3. Copy sampled images to a new directory
4. Generate a new simplified annotation file (CSV or JSON)

## Annotation Format

Each JSON file contains entries like:

```json
{
  "image_id": {
    "bboxes": [[x, y, w, h], ...],
    "labels": ["gesture_name", ...],
    "user_id": "hash",
    "hand_landmarks": [[[x,y], ...], ...],
    "meta": {
      "age": [24.41],
      "gender": ["female"],
      "race": ["White"]
    }
  }
}
```

Where:

- `bboxes`: Normalized [top_left_x, top_left_y, width, height] in range [0, 1]
- `labels`: Gesture class for each hand
- `user_id`: Subject identifier (for proper train/val/test split)
- `hand_landmarks`: 21 keypoints per hand (MediaPipe format)

## Next Steps

1. ✅ **Assessment complete**
2. ⏭️ Create a custom subset extraction script (`scripts/download_hagrid_subset.py`)
3. ⏭️ Download annotations only (lightweight)
4. ⏭️ Extract ~500-1000 images per selected gesture class
5. ⏭️ Generate a simplified dataset in folder-per-class structure for PyTorch
6. ⏭️ Update training notebook to point to the subset
7. ⏭️ Run smoke training on the subset

## Resources

- **Repository**: https://github.com/hukenovs/hagrid
- **Paper**: https://arxiv.org/abs/2412.01508
- **Pretrained Models**: YOLOv10, MobileNetV3, ViT-B/16, ResNet
- **Download Script**: `download.py` in repository
- **License**: CC BY-SA 4.0

## Estimated Timeline

- Download annotations: 5 minutes
- Create subset script: 30 minutes
- Download full classes + extract subset: 2-4 hours (depending on network)
- Dataset preparation: 30 minutes
- Total: ~4-5 hours

## Storage Requirements (Final)

- Annotations: ~50 MB
- Selected subset (10 classes × 800 images): **~2.5 GB**
- Preprocessed/augmented cache: +500 MB
- Model checkpoints: +2 GB
- **Total project storage: ~5 GB**
