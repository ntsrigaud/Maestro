> **LEGACY DOCUMENT** — This guide describes the original IPN-only training pipeline.
> The production system now uses the hybrid Jester+IPN vocabulary (`vocabulary_hybrid.yaml`).
> See `configs/gestures/vocabulary_hybrid.yaml` and `configs/thresholds/production_hybrid.yaml`
> for the current production configuration.

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

Both `t_start` and `t_end` are 1-indexed, inclusive frame numbers. The
`frames` column equals `t_end − t_start + 1`.

### 1.2 Video Properties

| Property    | Value       |
| ----------- | ----------- |
| Format      | AVI (MJPEG) |
| Resolution  | 640 × 480   |
| Frame rate  | 30 FPS      |
| Videos      | 200         |
| Train split | 148 videos  |
| Test split  | 52 videos   |

### 1.3 Participant Diversity

| Participant | Videos | Hand         |
| ----------- | ------ | ------------ |
| 1CM1        | 16     | Right (all)  |
| 1CM42       | 64     | Right (all)  |
| 1CV12       | 48     | Right + Left |
| 4CM11       | 72     | Right (most) |

All 4 participants contributed across train and test splits.

### 1.4 Gesture Classes

| ID  | Code | Gesture                       |
| --- | ---- | ----------------------------- |
| 1   | D0X  | Non-gesture                   |
| 2   | B0A  | Pointing with one finger      |
| 3   | B0B  | Pointing with two fingers     |
| 4   | G01  | Click with one finger         |
| 5   | G02  | Click with two fingers        |
| 6   | G03  | Throw up                      |
| 7   | G04  | Throw down                    |
| 8   | G05  | Throw left                    |
| 9   | G06  | Throw right                   |
| 10  | G07  | Open twice                    |
| 11  | G08  | Double click with one finger  |
| 12  | G09  | Double click with two fingers |
| 13  | G10  | Zoom in                       |
| 14  | G11  | Zoom out                      |

**Class balance**: each gesture class (G01–G11) has exactly 200 instances
across all 200 videos. D0X has 1 431 instances; B0A and B0B each have
~1 000 instances.

**Gesture duration statistics (all mapped classes, n = 4 218)**:

| Stat   | Frames | Seconds at 30 FPS |
| ------ | ------ | ----------------- |
| Min    | 9      | 0.3 s             |
| Max    | 650    | 21.7 s            |
| Mean   | 140    | 4.7 s             |
| Median | 112    | 3.7 s             |

The LSTM window is 32 frames (≈1.07 s), so most gesture segments comfortably
contain multiple non-overlapping windows.

---

## 2. Gesture Mapping

### 2.1 IPN → Runtime Vocabulary

The default vocabulary used by `build_ipn_dataset.py` is
`configs/gestures/vocabulary_ipn.yaml` (10 classes):

| Vocabulary Class | IPN Source | Runtime Role                      |
| ---------------- | ---------- | --------------------------------- |
| `unknown`        | D0X        | Rejection / background            |
| `point_one`      | B0A        | Continuous laser pointer gesture  |
| `point_two`      | B0B        | Continuous annotation pen gesture |
| `open_palm_hold` | G07        | Discrete command gesture          |
| `swipe_right`    | G06        | Discrete command gesture          |
| `swipe_left`     | G05        | Discrete command gesture          |
| `swipe_up`       | G03        | Discrete command gesture          |
| `swipe_down`     | G04        | Discrete command gesture          |
| `zoom_in`        | G10        | Discrete command gesture          |
| `zoom_out`       | G11        | Discrete command gesture          |

> **`open_palm_hold` alias rationale** — IPN G07 is annotated as "Open twice"
> (the performer opens their hand from a partial to fully open pose, twice in
> quick succession). No IPN class captures a _static_ open-palm hold, so G07
> is the nearest approximation: the double-open motion naturally involves the
> palm facing the camera in an open state. During dataset assembly
> (`scripts/build_ipn_dataset.py`) G07 frames are labeled `open_palm_hold`.
> **Retraining is not required** — vocabulary, model, and runtime are fully
> consistent. The "Approximate" match quality above reflects the semantic
> distance between G07 ("Open twice") and the intended gesture ("Open palm
> hold"), not a data or label error.

### 2.2 Extended Labels

`build_ipn_dataset.py` contains mappings for G01/G02/G08/G09
(`click_one`, `click_two`, `double_click_one`, `double_click_two`).
Those labels are included only when the selected vocabulary YAML contains them.

### 2.3 Runtime Command Mapping

Runtime command execution is configured by
`configs/gestures/command_map_presentation.yaml`:

- Discrete path (`gesture_commands`): `swipe_*`, `zoom_*`, `open_palm_hold`, `unknown`
- Continuous path (`gesture_modes`): `point_one`, `point_two`

Continuous gestures bypass `CommandDispatcher` and are handled per-frame by
runtime trackers.

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
│  • MediaPipe Holistic (legacy solutions API)     │
│  • Per-frame cache: keypoints (67x3), valid mask │
│  • Cache: data/IPN/cache/<stem>_holistic.npz     │
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
│  • Output format selectable:                      │
│    – graph (default): graph windows (N,T,67,3)    │
│    – flat: feature-schema-v5 windows (N,T,147)    │
│  • Sliding window defaults:                        │
│    window_length=32, step=3, min_valid_ratio=0.8  │
│  • recording_id = {video}:seg{i:04d}             │
│    (builder resets feature state at each segment)│
└─────────────────────────────────────────────────┘
       │
       ▼
  data/processed/ipn_dataset.npz
  (sequences, labels, label_ids,
   recording_ids, metadata_json)
```

### 3.2 Feature Dimensions

For `--format flat`, each sequence has shape `(T, 147)` using `feature-schema-v5`.
With default CLI values this is `(32, 147)`.

For `--format graph` (default), each sequence has shape `(T, 67, 3)`.

### 3.3 Frame Sampling Strategy

- **No stride at extraction**: all frames are read sequentially to preserve
  MediaPipe's temporal tracking quality.
- **Annotation boundaries**: only frames within `[t_start, t_end]` for each
  gesture segment are fed to the dataset builder. The rest are cached but
  ignored during assembly.
- **Sliding windows**: the builder emits one window every `window_step`
  frames once the buffer reaches `window_length` frames (defaults: 3 and 32).
- **Unknown segment capping**: long background spans (D0X, ~300 frames mean)
  are truncated to `max_unknown_frames_per_segment = 60` at assembly time to
  prevent class imbalance from background over-representation.

### 3.4 Class Imbalance Handling

Two complementary mechanisms keep the `unknown` class from dominating:

1. `--max-unknown-segments-per-source N` (default 150): randomly selects at
   most N annotation rows per source label (D0X, B0A, B0B).
2. `--max-unknown-frames-per-segment F` (default 60): truncates each selected
   unknown segment to at most F frames before window generation.

Remaining imbalance can be further controlled by training-time class capping
(`max_samples_per_class`) and optional class weighting in the model config.

---

## 4. Dataset Generation Commands

### 4.1 Generate Full Dataset (All 200 Videos)

```bash
python scripts/build_ipn_dataset.py \
    --ipn-root data/IPN \
    --format flat \
    --output-npz data/processed/ipn_dataset.npz \
    --num-workers 4
```

On a modern laptop with 4 cores, extraction takes approximately 15–30 minutes
per run. Subsequent runs reuse the per-video caches and complete in under
5 minutes.

### 4.2 Use Original IPN Train/Test Splits

Process training videos only (recommended if you want reproducible results
against the IPN benchmark):

```bash
# Training split
python scripts/build_ipn_dataset.py \
    --ipn-root data/IPN \
    --annotation-file Annot_TrainList.txt \
    --format flat \
    --output-npz data/processed/ipn_train.npz \
    --num-workers 4

# Test split
python scripts/build_ipn_dataset.py \
    --ipn-root data/IPN \
    --annotation-file Annot_TestList.txt \
    --format flat \
    --output-npz data/processed/ipn_test.npz \
    --num-workers 4
```

### 4.3 Force Re-Extraction (Ignore Caches)

```bash
python scripts/build_ipn_dataset.py \
    --ipn-root data/IPN \
    --format flat \
    --output-npz data/processed/ipn_dataset.npz \
    --no-cache \
    --num-workers 4
```

### 4.4 Tune Unknown Class Balance

```bash
python scripts/build_ipn_dataset.py \
    --ipn-root data/IPN \
    --format flat \
    --output-npz data/processed/ipn_dataset.npz \
    --max-unknown-segments-per-source 50 \
    --max-unknown-frames-per-segment 48 \
    --num-workers 4
```

### 4.5 Single-Core Debug Run

```bash
python scripts/build_ipn_dataset.py \
    --ipn-root data/IPN \
    --format flat \
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
    --dataset-path data/processed/ipn_dataset.npz \
    --config configs/model/two_stream_attn.yaml
```

The vocabulary is embedded in the dataset NPZ (`metadata_json.vocabulary`).
`train_model.py` reads labels from the dataset metadata.

---

## 6. Configuration Files Added

| File                                     | Purpose                                 |
| ---------------------------------------- | --------------------------------------- |
| `configs/gestures/vocabulary_ipn.yaml`   | Default 10-class IPN-derived vocabulary |
| `configs/thresholds/production_ipn.yaml` | Inference thresholds for IPN classes    |
| `scripts/build_ipn_dataset.py`           | IPN video processing pipeline           |

---

## 7. Key Script Options Reference

| Flag                                | Default             | Description                                        |
| ----------------------------------- | ------------------- | -------------------------------------------------- |
| `--ipn-root`                        | `data/IPN`          | Dataset root directory                             |
| `--output-npz`                      | _(required)_        | Output NPZ path                                    |
| `--annotation-file`                 | `Annot_List.txt`    | Which annotation file to process                   |
| `--vocab-config`                    | vocabulary_ipn.yaml | Vocabulary definition                              |
| `--num-workers`                     | `1`                 | Parallel extraction workers (0 = all cores)        |
| `--no-cache`                        | off                 | Force re-extraction of all videos                  |
| `--max-unknown-segments-per-source` | `150`               | Subsample unknown segments per source class        |
| `--max-unknown-frames-per-segment`  | `60`                | Cap unknown segment frame count                    |
| `--unknown-subsample-ratio`         | `1.0`               | Fraction of unknown sequences kept after filtering |
| `--window-length`                   | `32`                | Sliding window length (frames)                     |
| `--window-step`                     | `3`                 | Window step size (frames)                          |
| `--min-valid-ratio`                 | `0.8`               | Min fraction of valid frames to emit a window      |
| `--label-strategy`                  | `majority`          | Window label assignment (`majority` or `last`)     |
| `--min-detection-confidence`        | `0.7`               | MediaPipe detection confidence threshold           |
| `--min-tracking-confidence`         | `0.5`               | MediaPipe tracking confidence threshold            |
| `--format`                          | `graph`             | Output format (`graph` or `flat`)                  |
| `--metadata-json`                   | auto                | Optional metadata summary JSON output path         |
| `--seed`                            | `42`                | Random seed for unknown segment sampling           |

---

## 8. Compatibility Notes

The output NPZ is compatible with the existing training pipeline:

- `--format flat` datasets are consumed by `GestureDataModule` and two-stream models.
- `--format graph` datasets are consumed by `GraphGestureDataModule` and graph models.

Runtime config resolution is directory-based (via `--config` path / config root).
`build_container()` loads:

- command map from `configs/app.yaml` (`app.command_map`),
- thresholds from `configs/thresholds/production_ipn.yaml`,
- model source from `configs/model_source.yaml`.
