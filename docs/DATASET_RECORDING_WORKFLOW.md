# Dataset Recording and Construction Workflow (Phase 4 Ready)

Date: 2026-03-23  
Role: Senior Machine Learning Systems Engineer  
Scope: End-to-end, reproducible process to generate `labeled_landmarks.jsonl` and build LSTM-ready `.npz` dataset.

---

## 1. Repository Infrastructure Audit

### 1.1 Dataset-related components and responsibilities

- [scripts/collect_data.py](scripts/collect_data.py)
  - Produces labeled landmark JSONL records from live webcam stream.
  - Uses existing app container infrastructure for camera + MediaPipe extraction.
- [scripts/build_dataset.py](scripts/build_dataset.py)
  - Consumes labeled JSONL and builds sequence dataset (`.npz`) via `SequenceDatasetBuilder`.
- [src/gesture_control/infrastructure/camera/opencv_source.py](src/gesture_control/infrastructure/camera/opencv_source.py)
  - Webcam frame capture (`FramePacket`).
- [src/gesture_control/infrastructure/mediapipe/hands_extractor.py](src/gesture_control/infrastructure/mediapipe/hands_extractor.py)
  - MediaPipe Hands extraction (`LandmarkPacket`).
- [src/gesture_control/pipeline/feature_engineering.py](src/gesture_control/pipeline/feature_engineering.py)
  - Landmark to 126-dim feature transform (position + velocity).
- [src/gesture_control/pipeline/buffering.py](src/gesture_control/pipeline/buffering.py)
  - Sliding window buffering (`window_length=24`, `window_step=3`, `min_valid_ratio=0.8`).
- [src/gesture_control/training/dataset_builder.py](src/gesture_control/training/dataset_builder.py)
  - Labeled frame to sequence sample conversion.
  - NPZ serialization and loading utility.

### 1.2 Which component produces labeled landmark sequences?

- Producer: [scripts/collect_data.py](scripts/collect_data.py)
- It writes line-delimited JSON (`.jsonl`) records containing:
  - recording metadata
  - frame timestamps
  - per-frame hand landmarks
  - confidence and validity flags
  - assigned gesture label

### 1.3 How `labeled_landmarks.jsonl` is generated

- Run one collection command per gesture class (or per recording session).
- Use append mode into the same file.
- Each command assigns one label to all captured frames in that run.
- Repeat for all gestures in the vocabulary.

### 1.4 Data flow between components

1. Webcam frame capture: `OpenCVFrameSource`
2. Hand landmark extraction: `MediaPipeHandsExtractor`
3. JSONL recording: `collect_data.py`
4. JSONL parsing and validation: `build_dataset.py`
5. Feature transform and temporal windowing: `SequenceDatasetBuilder` internals
6. Sequence dataset serialization: `.npz` output

---

## 2. JSONL Dataset Format Verification

### 2.1 Exact schema expected by `build_dataset.py`

`build_dataset.py` requires these fields per JSON line:

- `recording_id` (required, string)
- `timestamp_ms` (required, integer)
- `label` (required, string)
- `landmarks` (required, list of 21 points for valid frames; each point is `[x, y, z]`)

Optional fields supported:

- `valid` (bool, default `true`)
- `handedness` (string or null)
- `detect_conf` (float, default `1.0`)
- `track_conf` (float, default `1.0`)

Additional fields are allowed and ignored by `build_dataset.py` parser.  
Example: `frame_index` is written by collection CLI and is safe.

### 2.2 Notes on requested fields

- `timestamp`: mapped to `timestamp_ms` (required).
- `gesture_label`: mapped to `label` (required).
- `frame_index`: optional, included by collection script for traceability.
- `hand_landmarks`: mapped to `landmarks` (required).
- `orientation data`: not required in JSONL; orientation normalization is optional and applied during dataset build via `--enable-orientation-normalization`.
- `feature vectors`: not expected in JSONL; features are computed inside `SequenceDatasetBuilder`.

### 2.3 Valid JSONL example (single line)

```json
{
  "recording_id": "rec-swipe-right-001",
  "frame_index": 42,
  "timestamp_ms": 1386,
  "label": "swipe_right",
  "landmarks": [
    [0.5123, 0.6281, -0.0432],
    [0.5351, 0.5988, -0.0511],
    [0.5567, 0.5702, -0.0593],
    [0.5789, 0.541, -0.0674],
    [0.6014, 0.5127, -0.0748],
    [0.5441, 0.5332, -0.0487],
    [0.5639, 0.5094, -0.0558],
    [0.5824, 0.4867, -0.0622],
    [0.6021, 0.4649, -0.0681],
    [0.559, 0.5468, -0.0466],
    [0.5798, 0.523, -0.0531],
    [0.6006, 0.4997, -0.0598],
    [0.621, 0.4779, -0.0659],
    [0.5732, 0.5621, -0.0449],
    [0.595, 0.5396, -0.0506],
    [0.6168, 0.5174, -0.0563],
    [0.638, 0.4958, -0.062],
    [0.5869, 0.5796, -0.0438],
    [0.6104, 0.5597, -0.0491],
    [0.6341, 0.5404, -0.0548],
    [0.6572, 0.5218, -0.0601]
  ],
  "valid": true,
  "handedness": "Right",
  "detect_conf": 0.97,
  "track_conf": 1.0
}
```

---

## 3. Reconstructed End-to-End Recording Pipeline

1. Gesture recording (live webcam)
2. MediaPipe landmark extraction (21x3)
3. Labeled landmark JSONL append
4. Feature engineering (wrist-relative, scale-normalized, velocity)
5. Temporal sequence buffering (24-frame windows, step=3)
6. Sequence labeling (`last` or `majority`)
7. Dataset serialization (`.npz`)
8. Dataset loading for LSTM training pipeline

---

## 4. Recording Procedure Guide (Executable Commands)

### 4.1 Environment setup

```bash
python -m pip install --upgrade pip
pip install -r requirements/dev.txt
pip install -e .
```

Optional sanity checks:

```bash
ruff check .
pytest -q
```

### 4.2 Configuration check

- Camera/runtime defaults: [configs/app.yaml](configs/app.yaml)
- Windowing defaults: [configs/inference.yaml](configs/inference.yaml)
- Gesture vocabulary: [configs/gestures/vocabulary_v1.yaml](configs/gestures/vocabulary_v1.yaml)

Current vocabulary:

- open_palm_hold
- swipe_right
- swipe_left
- fist_hold
- thumbs_up
- v_sign
- pinch_hold
- unknown

### 4.3 Start a recording session per gesture label

Create output folder first:

```bash
mkdir -p data/raw
```

Record one gesture class (example: `swipe_right`):

```bash
python scripts/collect_data.py \
  --config configs \
  --label swipe_right \
  --recording-id rec-swipe-right-001 \
  --output-jsonl data/raw/labeled_landmarks.jsonl \
  --max-seconds 45
```

Record another gesture class (append to same JSONL):

```bash
python scripts/collect_data.py \
  --config configs \
  --label swipe_left \
  --recording-id rec-swipe-left-001 \
  --output-jsonl data/raw/labeled_landmarks.jsonl \
  --max-seconds 45
```

Record `unknown` class (non-command hand activity):

```bash
python scripts/collect_data.py \
  --config configs \
  --label unknown \
  --recording-id rec-unknown-001 \
  --output-jsonl data/raw/labeled_landmarks.jsonl \
  --max-seconds 60 \
  --include-invalid
```

Repeat for all labels and optional extra sessions (`...-002`, `...-003`) to improve class balance.

### 4.4 Expected recording outputs

- Primary output: `data/raw/labeled_landmarks.jsonl`
- Script summary markers in terminal:
  - `DATA_COLLECTION_START`
  - `DATA_COLLECTION_END`
  - `frames_seen=...`
  - `records_written=...`

---

## 5. Dataset Construction Guide (`.jsonl` to `.npz`)

### 5.1 Build command (recommended)

```bash
mkdir -p data/processed
python scripts/build_dataset.py \
  --input-jsonl data/raw/labeled_landmarks.jsonl \
  --output-npz data/processed/gesture_dataset_v1.npz \
  --config-dir configs \
  --label-strategy majority \
  --enable-orientation-normalization
```

### 5.2 Parameter reference

- `--input-jsonl`: input landmark JSONL file.
- `--output-npz`: target serialized sequence dataset.
- `--config-dir`: directory containing inference + gestures configs.
- `--label-strategy`:
  - `last`: sequence label = last frame label.
  - `majority`: sequence label = majority label in window.
- `--enable-orientation-normalization`: enables orientation canonicalization in feature transform.

### 5.3 Expected build outputs

- Primary output file: `data/processed/gesture_dataset_v1.npz`
- Terminal markers:
  - `DATASET_BUILD_START`
  - `input_frames=...`
  - `output_samples=...`
  - `window_length=24`
  - `window_step=3`
  - `DATASET_BUILD_END`

---

## 6. Validation Checks

### 6.1 Quick file-level checks

```bash
wc -l data/raw/labeled_landmarks.jsonl
ls -lh data/processed/gesture_dataset_v1.npz
```

### 6.2 JSONL schema spot-check

```bash
python - <<'PY'
import json
from pathlib import Path
p = Path('data/raw/labeled_landmarks.jsonl')
with p.open('r', encoding='utf-8') as f:
    first = json.loads(next(line for line in f if line.strip()))
required = {'recording_id', 'timestamp_ms', 'label', 'landmarks'}
missing = required - set(first)
print('missing_required_fields=', missing)
print('has_frame_index=', 'frame_index' in first)
print('landmark_points=', len(first.get('landmarks', [])))
PY
```

### 6.3 NPZ load and shape verification

```bash
python - <<'PY'
from gesture_control.training.dataset_builder import SequenceDatasetBuilder
p = SequenceDatasetBuilder.load_npz('data/processed/gesture_dataset_v1.npz')
seq = p['sequences']
labels = p['labels']
meta = p['metadata']
print('sequences_shape=', seq.shape)
print('labels_shape=', labels.shape)
print('feature_dim=', meta['feature_dim'])
print('window_length=', meta['window_length'])
print('window_step=', meta['window_step'])
assert seq.ndim == 3
assert seq.shape[1] == meta['window_length']
assert seq.shape[2] == meta['feature_dim']
assert labels.shape[0] == seq.shape[0]
print('dataset_validation=OK')
PY
```

### 6.4 Label distribution inspection

```bash
python - <<'PY'
from collections import Counter
from gesture_control.training.dataset_builder import SequenceDatasetBuilder
p = SequenceDatasetBuilder.load_npz('data/processed/gesture_dataset_v1.npz')
c = Counter(p['labels'].tolist())
for k, v in sorted(c.items()):
    print(f'{k}: {v}')
PY
```

---

## 7. Documentation Improvements (Recommended)

The workflow was previously unclear because recording was not documented and `collect_data.py` was a placeholder.  
Recommended improvements:

1. Add a "Dataset Recording" section in [README.md](README.md) linking this guide.
2. Keep this guide as the canonical recording/build SOP.
3. Add examples to CLI help text in [scripts/collect_data.py](scripts/collect_data.py) and [scripts/build_dataset.py](scripts/build_dataset.py).
4. Add a short JSONL schema snippet to [docs/DATASET_GENERATION_STRATEGY.md](docs/DATASET_GENERATION_STRATEGY.md).

---

## 8. Final Answer to the Original Error

Why `build_dataset.py` failed:

- The required input file `labeled_landmarks.jsonl` did not exist yet.

How to generate it now:

- Use [scripts/collect_data.py](scripts/collect_data.py) with `--label`, `--recording-id`, and `--output-jsonl`.
- Run once per gesture label (append mode), then build NPZ with [scripts/build_dataset.py](scripts/build_dataset.py).

This fully prepares the project dataset for the upcoming Phase 4 LSTM training pipeline.
