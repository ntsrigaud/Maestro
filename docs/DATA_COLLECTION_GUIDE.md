# Data Collection and Training Guide

**Branch:** `fix/training-pipeline-improvement`  
**Date:** 2026-03-30  
**Scope:** How to record a properly segmented dataset, build the NPZ artifact, and train a new LSTM model.

---

## Background and Motivation

The original `lstm_v1_20260328T100850Z` model achieved 100 % validation accuracy — an
artefact of **data leakage** rather than genuine generalisation. Root causes:

| #   | Issue                                                          | Impact                                         |
| --- | -------------------------------------------------------------- | ---------------------------------------------- |
| 1   | Continuous session recording (rest frames mixed with gestures) | Model learns ambiguous signals                 |
| 2   | One recording per gesture class                                | No diversity; model memorises one trajectory   |
| 3   | Window-level random split with 87.5 % window overlap           | Train/val/test windows nearly identical        |
| 4   | No data augmentation                                           | Model cannot handle speed or spatial variation |

The infrastructure described in this guide fixes all four issues.

---

## 1. Prerequisites

```bash
conda activate gesture-control
# Webcam connected and accessible at camera_index 0 (see configs/app.yaml)
```

---

## 2. Recording Workflow Overview

```
Interactive collect_data.py                build_dataset.py
┌─────────────────────┐                  ┌──────────────────────┐
│ one SPACE press =   │  JSONL segments  │ reads recording_id   │
│ one isolated        │ ───────────────► │ boundaries; resets   │
│ gesture instance    │                  │ LSTM window buffer   │
└─────────────────────┘                  └──────────────────────┘
        │                                          │
        │  unique recording_id per gesture         │  NPZ dataset
        │  e.g. rec-swipe-right-001               ▼
        └──────────────────────────►  GestureDataModule (group split)
                                        no leakage across splits
```

Each **SPACE-delimited recording** captures exactly **one isolated gesture
instance**. There are no rest-period frames mixed in.

---

## 3. Collecting Recordings — Interactive Mode

### 3.1 Command

```bash
python scripts/collect_data.py \
  --label swipe_right \
  --recording-id rec-swipe-right \
  --output-jsonl data/raw/labeled_landmarks.jsonl \
  --interactive
```

| Argument         | Meaning                                                                                                |
| ---------------- | ------------------------------------------------------------------------------------------------------ |
| `--label`        | Gesture class name (must be in `configs/gestures/vocabulary_v1.yaml`)                                  |
| `--recording-id` | **Prefix** used for this batch of recordings. The script appends `-001`, `-002`, … automatically.      |
| `--output-jsonl` | JSONL file to append records to (multiple sessions and labels can share one file).                     |
| `--interactive`  | Enable the segmented recording mode.                                                                   |
| `--max-frames N` | Optional safety cap: auto-stop a recording after `N` frames (e.g. `--max-frames 120` ≈ 4 s at 30 fps). |

### 3.2 What the Preview Window Shows

| Visual cue                                             | State                                                   |
| ------------------------------------------------------ | ------------------------------------------------------- |
| **Green border** + "IDLE Next: rec-swipe-right-001"    | Waiting; camera is live but not recording               |
| **Red border** + "REC rec-swipe-right-001 [12 frames]" | Recording in progress                                   |
| Bottom bar                                             | Label, total recordings completed, total frames written |

### 3.3 Keyboard Controls

| Key         | Action                                      |
| ----------- | ------------------------------------------- |
| `SPACE`     | **Start** a new recording (IDLE → REC)      |
| `SPACE`     | **Stop** the current recording (REC → IDLE) |
| `Q` / `ESC` | Quit the session                            |

### 3.4 Recording a Single Gesture

1. **Position your hand** in the camera frame in the rest/start position.
2. **Press SPACE** — the border turns red.
3. **Perform the gesture once**, cleanly and completely.
4. **Press SPACE** — the border turns green; the segment is saved.
5. Return to rest position. Repeat from step 2 for the next instance.

> **Important:** Each recording should contain **one clean repetition** of the
> gesture. Do not include the hand returning to start position inside the same
> recording segment.

### 3.5 Recommended Recording Protocol

Collect **≥ 15 recordings per gesture class** across varied conditions:

| Variation     | Description                                                     |
| ------------- | --------------------------------------------------------------- |
| Speed         | Perform the gesture at slow, normal, and fast pace              |
| Distance      | Vary how far your hand is from the camera (40 cm, 60 cm, 80 cm) |
| Angle         | Slight left/right wrist rotation                                |
| Lighting      | Record in at least two lighting conditions                      |
| Hand position | Centre, left, and right of the camera frame                     |

For dynamic classes (`swipe_left`, `swipe_right`, `pinch_hold`) aim for
**≥ 20 recordings** because they have higher within-class variance.

---

## 4. Collecting Each Gesture Class

Repeat section 3 for every class in the vocabulary. Use a distinct prefix per
class so recording IDs are globally unique across the JSONL file:

```bash
# Static holds
python scripts/collect_data.py --label open_palm_hold  --recording-id rec-open-palm  --output-jsonl data/raw/labeled_landmarks.jsonl --interactive
python scripts/collect_data.py --label fist_hold       --recording-id rec-fist        --output-jsonl data/raw/labeled_landmarks.jsonl --interactive
python scripts/collect_data.py --label thumbs_up       --recording-id rec-thumbs-up   --output-jsonl data/raw/labeled_landmarks.jsonl --interactive
python scripts/collect_data.py --label v_sign          --recording-id rec-v-sign      --output-jsonl data/raw/labeled_landmarks.jsonl --interactive

# Dynamic gestures (collect more instances)
python scripts/collect_data.py --label swipe_right     --recording-id rec-swipe-right --output-jsonl data/raw/labeled_landmarks.jsonl --interactive
python scripts/collect_data.py --label swipe_left      --recording-id rec-swipe-left  --output-jsonl data/raw/labeled_landmarks.jsonl --interactive
python scripts/collect_data.py --label pinch_hold      --recording-id rec-pinch       --output-jsonl data/raw/labeled_landmarks.jsonl --interactive

# Negative / unknown class (essential for low false-trigger rate)
python scripts/collect_data.py --label unknown         --recording-id rec-unknown     --output-jsonl data/raw/labeled_landmarks.jsonl --interactive
```

Non-interactive (batch) mode is still available if needed:

```bash
python scripts/collect_data.py \
  --label fist_hold \
  --recording-id rec-fist-001 \
  --output-jsonl data/raw/labeled_landmarks.jsonl \
  --max-seconds 5
```

---

## 5. Inspecting Raw Data

Quick sanity check before building the dataset:

```bash
python - <<'EOF'
import json, collections
recs = collections.defaultdict(set)
with open("data/raw/labeled_landmarks.jsonl") as f:
    for line in f:
        r = json.loads(line)
        recs[r["label"]].add(r["recording_id"])
for lbl, ids in sorted(recs.items()):
    print(f"  {lbl:<20s}  {len(ids):>3d} recordings")
EOF
```

Expected output (minimum viable dataset):

```
  fist_hold             15 recordings
  open_palm_hold        15 recordings
  pinch_hold            20 recordings
  swipe_left            20 recordings
  swipe_right           20 recordings
  thumbs_up             15 recordings
  unknown               15 recordings
  v_sign                15 recordings
```

---

## 6. Building the Dataset

Convert raw JSONL landmarks into a feature-engineered `.npz` artifact:

```bash
python scripts/build_dataset.py \
  --input-jsonl  data/raw/labeled_landmarks.jsonl \
  --output-npz   data/processed/gesture_dataset_v2.npz \
  --config-dir   configs
```

The builder:

1. Reads all frames and groups them by `recording_id`.
2. **Resets the sliding-window buffer** at each recording boundary, so windows
   never span frames from two different recordings.
3. Applies `LandmarkFeatureTransformer` (wrist-relative normalisation +
   velocity features) — the same transform used at inference time.
4. Writes per-sample `recording_ids` to the NPZ for use by the group splitter.

---

## 7. Training the Model

```bash
python scripts/train_model.py \
  --dataset    data/processed/gesture_dataset_v2.npz \
  --config     configs/model/lstm_default.yaml \
  --config-dir configs \
  --output-dir models/checkpoints \
  --export-dir models/exported \
  --run-dir    experiments/runs
```

### 7.1 What Has Changed in the Training Pipeline

| Component     | Old behaviour                                      | New behaviour                                                                                    |
| ------------- | -------------------------------------------------- | ------------------------------------------------------------------------------------------------ |
| Dataset split | Random window-index split (87.5 % overlap leakage) | **Recording-level `GroupShuffleSplit`** — no recording appears in more than one partition        |
| Augmentation  | None                                               | **Temporal speed scaling** ×0.8–1.2 + **spatial Gaussian jitter** σ = 0.005 on position features |
| Fallback      | —                                                  | If < 3 unique recording IDs, emits `UserWarning` and falls back to random split                  |

### 7.2 Expected Training Metrics with a Proper Dataset

| Metric         | Old (leaked)                | New (proper)           |
| -------------- | --------------------------- | ---------------------- |
| Val accuracy   | ~100 %                      | 80–90 %                |
| Val macro-F1   | ~1.00                       | 0.80–0.90              |
| Interpretation | Memorised single trajectory | Genuine generalisation |

A validation accuracy of **80–90 % is a success** — it means the model actually
generalised to held-out recordings it has never seen. A drop below 75 % suggests
more recording diversity is needed.

---

## 8. Verifying No Recording Leakage

After `train_model.py` completes, check the split info from Python:

```python
from gesture_control.training.datamodule import DataSplitConfig, GestureDataModule

dm = GestureDataModule(
    dataset_path="data/processed/gesture_dataset_v2.npz",
    batch_size=64,
    split=DataSplitConfig(train_ratio=0.7, val_ratio=0.15, test_ratio=0.15),
    seed=42,
)
dm.setup()
info = dm.split_info()

print("Train recordings:", len(info["train"]))
print("Val recordings  :", len(info["val"]))
print("Test recordings :", len(info["test"]))

# Verify no recording appears in two or more splits
assert info["train"].isdisjoint(info["val"]),  "LEAK: train/val share a recording"
assert info["train"].isdisjoint(info["test"]), "LEAK: train/test share a recording"
assert info["val"].isdisjoint(info["test"]),   "LEAK: val/test share a recording"
print("No leakage detected.")
```

---

## 9. Augmentation Configuration

Augmentation is controlled in `configs/model/lstm_default.yaml`:

```yaml
augmentation:
  enabled: true
  temporal_scale_range: [0.8, 1.2] # gesture speed ×0.8 … ×1.2
  spatial_jitter_sigma: 0.005 # σ on normalised position features
  apply_prob: 0.5 # each aug fires independently with prob 0.5
```

Augmentation applies **only to the training split**. Validation and test data
are always evaluated on the original unperturbed sequences.

To disable augmentation (not recommended for a real dataset):

```yaml
augmentation:
  enabled: false
```

---

## 10. Non-Interactive Fallback

If a GUI is unavailable (e.g. headless server), use the original non-interactive
mode. **Ensure each `--recording-id` is unique per gesture instance** to
preserve recording-level segmentation:

```bash
# Record 15 separate instances, one per invocation
for i in $(seq -f "%03g" 1 15); do
  python scripts/collect_data.py \
    --label swipe_right \
    --recording-id "rec-swipe-right-${i}" \
    --output-jsonl data/raw/labeled_landmarks.jsonl \
    --max-seconds 3
  echo "Recorded instance $i — press Enter for next"
  read
done
```

---

## 11. Troubleshooting

| Symptom                                 | Likely cause                        | Fix                                             |
| --------------------------------------- | ----------------------------------- | ----------------------------------------------- |
| Warning: "falling back to random split" | < 3 unique recording IDs in dataset | Collect more recordings per class               |
| Val accuracy > 95 % on new dataset      | Not enough recording diversity      | Add more subjects, lighting conditions          |
| Val accuracy < 70 %                     | Too little data or noisy recordings | Collect more instances; check gesture execution |
| Recording warning "only N frames"       | Recording stopped too early         | Aim for ≥ 40 frames per recording (≈ 1.5 s)     |
| `cv2` window doesn't open               | `--interactive` requires a display  | Use non-interactive mode on headless systems    |
