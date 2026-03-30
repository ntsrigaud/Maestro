# IPN Hand Dataset Integration Guide

## Overview

This document describes how to use the IPN Hand dataset to train the LSTM
gesture classifier. The IPN dataset provides 200 videos with dense per-frame
annotations across 14 gesture classes, recorded at 30 FPS and 640×480
resolution. The integration pipeline converts those videos into the same
sliding-window NPZ format used by the existing training pipeline.

---

## 1. Dataset Analysis

### 1.1 Structure

The IPN Hand dataset is stored locally under `data/IPN/`:

```
data/IPN/
  annotations/
    Annot_List.txt          — 5 648 rows: all gesture segments across all videos
    Annot_TrainList.txt     — 4 038 rows: official train split
    Annot_TestList.txt      — 1 610 rows: official test split
    Video_TrainList.txt     — 147 video stems + total frame counts (train)
    Video_TestList.txt      —  51 video stems + total frame counts (test)
    classIdx.txt            — 14-class mapping (ID → label code)
    class_details.txt       — human-readable gesture descriptions
  videos/
    1CM1_1_R__217.avi       — 200 AVI files available locally
    …
```

Annotation file format (CSV with header):

```
video,label,id,t_start,t_end,frames
1CM1_4_R__229,G06,9,2351,2391,41
```

Both `t_start` and `t_end` are 1-indexed, inclusive frame numbers.  The
`frames` column equals `t_end − t_start + 1`.

### 1.2 Video Properties

| Property    | Value        |
|-------------|-------------|
| Format      | AVI (MJPEG) |
| Resolution  | 640 × 480   |
| Frame rate  | 30 FPS      |
| Videos      | 200         |
| Train split | 148 videos  |
| Test split  | 52 videos   |

### 1.3 Participant Diversity

| Participant | Videos | Hand         |
|-------------|--------|--------------|
| 1CM1        | 16     | Right (all)  |
| 1CM42       | 64     | Right (all)  |
| 1CV12       | 48     | Right + Left |
| 4CM11       | 72     | Right (most) |

All 4 participants contributed across train and test splits.

### 1.4 Gesture Classes

| ID  | Code | Gesture                      |
|-----|------|------------------------------|
| 1   | D0X  | Non-gesture                  |
| 2   | B0A  | Pointing with one finger     |
| 3   | B0B  | Pointing with two fingers    |
| 4   | G01  | Click with one finger        |
| 5   | G02  | Click with two fingers       |
| 6   | G03  | Throw up                     |
| 7   | G04  | Throw down                   |
| 8   | G05  | Throw left                   |
| 9   | G06  | Throw right                  |
| 10  | G07  | Open twice                   |
| 11  | G08  | Double click with one finger |
| 12  | G09  | Double click with two fingers|
| 13  | G10  | Zoom in                      |
| 14  | G11  | Zoom out                     |

**Class balance**: each gesture class (G01–G11) has exactly 200 instances
across all 200 videos.  D0X has 1 431 instances; B0A and B0B each have
~1 000 instances.

**Gesture duration statistics (all mapped classes, n = 4 218)**:

| Stat      | Frames | Seconds at 30 FPS |
|-----------|--------|-------------------|
| Min       | 9      | 0.3 s             |
| Max       | 650    | 21.7 s            |
| Mean      | 140    | 4.7 s             |
| Median    | 112    | 3.7 s             |

The LSTM window is 24 frames (0.8 s), so most gesture segments comfortably
contain multiple non-overlapping windows.

---

## 2. Gesture Mapping

### 2.1 IPN → Production Vocabulary

The original production vocabulary targets a presentation and meeting
controller.  The IPN dataset provides strong coverage for directional swipe
gestures and an open-palm hold but cannot supply data for fist, thumbs-up,
v-sign, or pinch poses without additional custom recording.

A revised vocabulary derived entirely from IPN data is defined in
`configs/gestures/vocabulary_ipn.yaml`:

| Vocabulary Class  | IPN Source | Semantic Command           | Match Quality |
|-------------------|-----------|----------------------------|---------------|
| `open_palm_hold`  | G07       | Activation / toggle UI     | Approximate   |
| `swipe_right`     | G06       | Next slide                 | Direct        |
| `swipe_left`      | G05       | Previous slide             | Direct        |
| `swipe_up`        | G03       | Raise hand / next section  | Direct        |
| `swipe_down`      | G04       | Start / stop presentation  | Direct        |
| `zoom_in`         | G10       | Zoom in                    | Direct        |
| `zoom_out`        | G11       | Zoom out                   | Direct        |
| `unknown`         | D0X + B0A + B0B | No-gesture / noise  | Direct        |

### 2.2 Omitted IPN Classes

| IPN Code | Gesture                       | Reason Omitted                          |
|----------|-------------------------------|------------------------------------------|
| G01      | Click with one finger         | No directional or hold semantic          |
| G02      | Click with two fingers        | No directional or hold semantic          |
| G08      | Double click with one finger  | No directional or hold semantic          |
| G09      | Double click with two fingers | No directional or hold semantic          |

### 2.3 Command Mapping

Update `configs/app.yaml` and `configs/thresholds/production_ipn.yaml` when
deploying a model trained on the IPN vocabulary.  The IPN threshold config
is already provided at `configs/thresholds/production_ipn.yaml`.

---

## 3. Processing Pipeline

### 3.1 Architecture

```
IPN Videos (AVI, 30 FPS)
       │
       ▼
┌─────────────────────────────────────────────────┐
│  Phase 1: Landmark Extraction  (parallel + cache)│
│                                                  │
│  • OpenCV VideoCapture (sequential read)         │
│  • MediaPipe Hands (static_image_mode=False)    │
│  • Per-frame: landmarks (21×3), valid flag,      │
│    detect_conf, track_conf, handedness           │
│  • Cache: data/IPN/cache/<stem>.npz             │
└─────────────────────────────────────────────────┘
       │  per-video NPZ caches
       ▼
┌─────────────────────────────────────────────────┐
│  Phase 2: Dataset Assembly                       │
│                                                  │
│  • Annotation-driven segment selection           │
│  • IPN label → vocabulary label mapping          │
│  • Unknown class: subsampled by segment count    │
│    and per-segment frame cap (default: 60 frames)│
│  • SequenceDatasetBuilder:                       │
│    – LandmarkFeatureTransformer (wrist-relative, │
│      scale-normalised positions + velocities)    │
│    – SlidingWindowSequenceBuffer (24 frames,     │
│      step 3, min_valid_ratio 0.8)                │
│  • recording_id = {video}:seg{i:04d}             │
│    (builder resets feature state at each segment)│
└─────────────────────────────────────────────────┘
       │
       ▼
  data/processed/ipn_dataset.npz
  (sequences: N×24×126, labels, label_ids,
   recording_ids, metadata_json)
```

### 3.2 Feature Dimensions

| Component       | Frames | Features per frame | Total dims |
|-----------------|--------|--------------------|------------|
| Position (21×3) | 24     | 63                 | 126        |
| Velocity (21×3) | 24     | 63                 | 126        |
| **Sequence**    | 24     | **126**            |            |

This matches `input_size: 126` in `configs/model/lstm_default.yaml` exactly.

### 3.3 Frame Sampling Strategy

- **No stride at extraction**: all frames are read sequentially to preserve
  MediaPipe's temporal tracking quality.
- **Annotation boundaries**: only frames within `[t_start, t_end]` for each
  gesture segment are fed to the dataset builder.  The rest are cached but
  ignored during assembly.
- **Sliding windows**: the builder emits one window every `window_step = 3`
  frames once the buffer reaches `window_length = 24` frames.
- **Unknown segment capping**: long background spans (D0X, ~300 frames mean)
  are truncated to `max_unknown_frames_per_segment = 60` at assembly time to
  prevent class imbalance from background over-representation.

### 3.4 Class Imbalance Handling

Two complementary mechanisms keep the `unknown` class from dominating:

1. `--max-unknown-segments-per-source N` (default 150): randomly selects at
   most N annotation rows per source label (D0X, B0A, B0B).
2. `--max-unknown-frames-per-segment F` (default 60): truncates each selected
   unknown segment to at most F frames before window generation.

Remaining imbalance is handled by `class_weighting: true` in the training
configuration, which applies inverse-frequency weights to the cross-entropy
loss.

---

## 4. Dataset Generation Commands

### 4.1 Generate Full Dataset (All 200 Videos)

```bash
python scripts/build_ipn_dataset.py \
    --ipn-root data/IPN \
    --output-npz data/processed/ipn_dataset.npz \
    --num-workers 4
```

On a modern laptop with 4 cores, extraction takes approximately 15–30 minutes
per run.  Subsequent runs reuse the per-video caches and complete in under
5 minutes.

### 4.2 Use Original IPN Train/Test Splits

Process training videos only (recommended if you want reproducible results
against the IPN benchmark):

```bash
# Training split
python scripts/build_ipn_dataset.py \
    --ipn-root data/IPN \
    --annotation-file Annot_TrainList.txt \
    --output-npz data/processed/ipn_train.npz \
    --num-workers 4

# Test split
python scripts/build_ipn_dataset.py \
    --ipn-root data/IPN \
    --annotation-file Annot_TestList.txt \
    --output-npz data/processed/ipn_test.npz \
    --num-workers 4
```

### 4.3 Force Re-Extraction (Ignore Caches)

```bash
python scripts/build_ipn_dataset.py \
    --ipn-root data/IPN \
    --output-npz data/processed/ipn_dataset.npz \
    --no-cache \
    --num-workers 4
```

### 4.4 Tune Unknown Class Balance

```bash
python scripts/build_ipn_dataset.py \
    --ipn-root data/IPN \
    --output-npz data/processed/ipn_dataset.npz \
    --max-unknown-segments-per-source 50 \
    --max-unknown-frames-per-segment 48 \
    --num-workers 4
```

### 4.5 Single-Core Debug Run

```bash
python scripts/build_ipn_dataset.py \
    --ipn-root data/IPN \
    --output-npz data/processed/ipn_debug.npz \
    --annotation-file Annot_TrainList.txt \
    --num-workers 1 \
    --log-level DEBUG
```

---

## 5. Training Commands

After generating the dataset, train the LSTM:

```bash
python scripts/train_model.py \
    --dataset data/processed/ipn_dataset.npz \
    --config configs/model/lstm_default.yaml
```

The vocabulary is already embedded in the dataset NPZ by `build_ipn_dataset.py`
(under the `vocabulary` key in `metadata_json`).  `train_model.py` reads it
from there — no separate `--vocab-config` flag is needed or accepted.

---

## 6. Configuration Files Added

| File                                          | Purpose                                |
|-----------------------------------------------|----------------------------------------|
| `configs/gestures/vocabulary_ipn.yaml`        | 8-class IPN-native gesture vocabulary  |
| `configs/thresholds/production_ipn.yaml`      | Inference thresholds for IPN classes   |
| `scripts/build_ipn_dataset.py`                | IPN video processing pipeline          |

---

## 7. Key Script Options Reference

| Flag                                  | Default               | Description                                          |
|---------------------------------------|-----------------------|------------------------------------------------------|
| `--ipn-root`                          | `data/IPN`            | Dataset root directory                               |
| `--output-npz`                        | *(required)*          | Output NPZ path                                      |
| `--annotation-file`                   | `Annot_List.txt`      | Which annotation file to process                     |
| `--vocab-config`                      | vocabulary_ipn.yaml   | Vocabulary definition                                |
| `--num-workers`                       | `1`                   | Parallel extraction workers (0 = all cores)          |
| `--no-cache`                          | off                   | Force re-extraction of all videos                    |
| `--max-unknown-segments-per-source`   | `150`                 | Subsample unknown segments per source class          |
| `--max-unknown-frames-per-segment`    | `60`                  | Cap unknown segment frame count                      |
| `--window-length`                     | `24`                  | Sliding window length (frames)                       |
| `--window-step`                       | `3`                   | Window step size (frames)                            |
| `--min-valid-ratio`                   | `0.8`                 | Min fraction of valid frames to emit a window        |
| `--label-strategy`                    | `majority`            | Window label assignment (`majority` or `last`)       |
| `--min-detection-confidence`          | `0.7`                 | MediaPipe detection confidence threshold             |
| `--min-tracking-confidence`           | `0.5`                 | MediaPipe tracking confidence threshold              |
| `--enable-orientation-normalization`  | off                   | Enable palm orientation normalisation                |
| `--seed`                              | `42`                  | Random seed for unknown segment sampling             |

---

## 8. Compatibility Notes

The output NPZ from `build_ipn_dataset.py` is fully compatible with
`GestureDataModule` and the existing training pipeline.  No changes to
`datamodule.py`, `train_lstm.py`, or the feature transformer are required.

The only pipeline difference from custom-collected data is the vocabulary:
- Use `configs/gestures/vocabulary_ipn.yaml` instead of `vocabulary_v1.yaml`
- Update inference to load `configs/thresholds/production_ipn.yaml`
- Pass the IPN vocab to `run_realtime.py` via its `--vocab-config` flag

For real-time inference with an IPN-trained model, the command mappings in
`configs/app.yaml` should be updated to reflect the new class names
(e.g., `swipe_up` → raise hand, `swipe_down` → stop presentation).
