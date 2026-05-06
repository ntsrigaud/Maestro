# STGCN-LSTM Migration Audit & Refactoring Plan

**System:** Gesture Control System (Maestro)  
**Prepared by:** Senior ML Systems Engineer Audit  
**Date:** 2026-05-01  
**Scope:** Two-Stream LSTM → STGCN-LSTM migration

---

## 1. Executive Summary

The current system achieves ~79.8% accuracy using a 147-dim handcrafted feature vector fed into a two-stream LSTM. The core limitations are: (a) loss of spatial relational structure through feature engineering that collapses graph topology into a flat vector; (b) single-hand input that misses bilateral coordination signals; and (c) temporal context limited to attention over a fixed 24-frame window without multi-scale temporal reasoning.

The STGCN-LSTM migration addresses all three by treating hand/body keypoints as nodes in a graph, applying learned spatial aggregation via graph convolution, and then passing the resulting feature sequence into an LSTM. The theoretical accuracy ceiling rises substantially — realistic targets are 86–91% macro-F1 based on comparable skeleton-based action recognition benchmarks on similarly-scaled datasets.

**Critical finding:** The reference implementation (`fokhruli/STGCN-rehab`) is a TensorFlow regression model that must be treated as a conceptual reference only. A full PyTorch reimplementation is required. None of its code can be ported directly.

**Critical path:** The dataset pipeline is the single riskiest and most laborious component of this migration. The model swap is straightforward once clean `(T, 67, 3)` data exists. Invest disproportionately in the data pipeline.

---

## 2. Reference Model Analysis

### 2.1 Architectural Breakdown

**Source:** `fokhruli/STGCN-rehab @ 82097e4e` — `GCN/sgcn_lstm.py`

**Framework:** TensorFlow 2.x / Keras (not PyTorch).

**Input:** `(B, T, N=25, C=3)` — 25 COCO-pose body joints, 3 coordinates, passed as a TensorFlow Tensor to `Conv2D` layers with NHWC layout where H=T, W=N.

**SGCN block** (method `sgcn(Input)`):

| Step | Operation                                                                                                                                                                 | Output shape         |
| ---- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------- | -------------------- |
| 1    | `Conv2D(64, kernel=(9,1), padding='same')` — temporal convolution                                                                                                         | `(B, T, N, 64)`      |
| 2    | Skip concat with input: `[Input, k1]`                                                                                                                                     | `(B, T, N, C+64)`    |
| 3a   | 1-hop: `Conv2D(64, 1×1)` → `expand_dims(axis=3)` → `ConvLSTM2D(25, 1×1, return_sequences=True)` → slice `[:,:,:,0,:]` → softmax + `bias_mat_1` → einsum `ntvw,ntwc->ntvc` | `(B, T, N, 64)`      |
| 3b   | 2-hop: same structure with `AD2`, `bias_mat_2`                                                                                                                            | `(B, T, N, 64)`      |
| 4    | Concat 1-hop + 2-hop                                                                                                                                                      | `(B, T, N, 128)`     |
| 5    | Multi-scale temporal: `Conv2D(16, 9×1)` → `Conv2D(16, 15×1)` → `Conv2D(16, 20×1)` + `Dropout(0.25)` each                                                                  | `(B, T, N, 16)` each |
| 6    | Concat all three temporal scales                                                                                                                                          | `(B, T, N, 48)`      |

**SGCN stack:** `x → sgcn(x) → y; y = y + x → z = sgcn(y); z = z + y` — two residual SGCN blocks.

**LSTM block** (method `Lstm(x)`):

| Step | Operation                                            | Output         |
| ---- | ---------------------------------------------------- | -------------- |
| 1    | `Reshape(T, N*C)`                                    | `(B, T, N×48)` |
| 2    | `LSTM(80, return_sequences=True)` + `Dropout(0.25)`  | `(B, T, 80)`   |
| 3    | `LSTM(40, return_sequences=True)` + `Dropout(0.25)`  | `(B, T, 40)`   |
| 4    | `LSTM(40, return_sequences=True)` + `Dropout(0.25)`  | `(B, T, 40)`   |
| 5    | `LSTM(80, return_sequences=False)` + `Dropout(0.25)` | `(B, 80)`      |
| 6    | `Dense(1, activation='linear')`                      | `(B, 1)`       |

**Loss:** Huber loss (delta=0.1). This is a **regression** model for rehabilitation movement scoring.

**Adjacency:** `AD`, `AD2`, `bias_mat_1`, `bias_mat_2` are externally pre-computed NumPy arrays, injected at construction. The model does not learn adjacency.

### 2.2 Critical Limitations of Reference

| Limitation                                      | Impact on Migration                                                                 |
| ----------------------------------------------- | ----------------------------------------------------------------------------------- |
| **TensorFlow, not PyTorch**                     | No code is reusable; requires full port                                             |
| **Regression output `Dense(1, linear)`**        | Must replace with `Linear(hidden, num_classes=8)` + CrossEntropy                    |
| **25 nodes hardcoded** (`self.num_joints = 25`) | Must redesign for 67 nodes                                                          |
| **ConvLSTM2D used for attention coefficients**  | Unusual design; standard `einsum`-based graph conv is simpler and equivalent        |
| **4-layer LSTM** (80→40→40→80)                  | ~400K+ LSTM params; too heavy for CPU real-time at N=67×48                          |
| **No batch normalisation in SGCN**              | Stabilisation risk during training; add `BatchNorm1d` per block                     |
| **Static pre-computed adjacency**               | Hardcoded graph structure; must define hand skeleton adjacency for 67 nodes         |
| **Multi-scale temporal kernels (9, 15, 20)**    | Good design; preserve this in the port                                              |
| **No normalisation of input coordinates**       | Raw x,y,z values from MediaPipe need wrist-relative normalisation before graph conv |

### 2.3 Conceptual Design Worth Preserving

- **Two-hop graph convolution:** first-order and second-order neighbourhood aggregation captures both direct joint connections and indirect dependencies (e.g., palm centre to fingertip via MCP→PIP→DIP chain).
- **Multi-scale temporal convolution** before LSTM: captures short-term velocity/acceleration patterns at multiple timescales, reducing the burden on the LSTM.
- **Residual connections** between SGCN blocks: training stability and gradient flow.
- **Learned attention coefficients** (the `coefs = softmax(leaky_relu(logits) + bias_mat)` pattern): allows the model to adaptively weight neighbour contributions beyond the fixed adjacency structure.

---

## 3. Gap Analysis

### 3.1 Input Representation

| Dimension                    | Current System                              | Target System                                          | Status            |
| ---------------------------- | ------------------------------------------- | ------------------------------------------------------ | ----------------- |
| Keypoint source              | MediaPipe Hands (21 kpts, 1 hand)           | MediaPipe Holistic / Hands+Pose+Face (67 kpts)         | **MUST change**   |
| Feature type                 | 147-dim engineered vector                   | Raw `(x, y, z)` per keypoint                           | **MUST change**   |
| Tensor shape entering model  | `(B, T=24, F=147)`                          | `(B, T, N=67, C=3)`                                    | **MUST change**   |
| Wrist-relative normalisation | Applied inside `LandmarkFeatureTransformer` | Must reapply per-frame to raw coords before graph conv | **MUST redesign** |
| Handedness                   | Single hand selected by `HandSelector`      | Both hands + body as unified graph                     | **MUST change**   |

### 3.2 Dataset Format

| Property                  | Current NPZ         | Required NPZ                                                                 |
| ------------------------- | ------------------- | ---------------------------------------------------------------------------- |
| `sequences` shape         | `(N, 24, 147)`      | `(N, T, 67, 3)`                                                              |
| `feature_dim` metadata    | `147`               | `67` nodes × `3` coords                                                      |
| `feature_schema_version`  | `feature-schema-v5` | New identifier, e.g. `graph-schema-v1`                                       |
| Adjacency matrix          | Not stored          | **MUST store** in NPZ or companion file                                      |
| Missing keypoint encoding | Zero vector         | Must define explicit missing-keypoint convention (zero fill with valid mask) |

### 3.3 Pipeline Component Gaps

| Component                        | Current Role                                       | Required Change                                           | Priority |
| -------------------------------- | -------------------------------------------------- | --------------------------------------------------------- | -------- |
| `LandmarkFeatureTransformer`     | 21 kpts → 147-dim vector                           | **Retire or replace** with raw-coordinate extractor       | MUST     |
| `SlidingWindowSequenceBuffer`    | Buffers `FeatureFrame` (147-dim vectors)           | Must buffer `(67, 3)` graph frames                        | MUST     |
| `FeatureFrame` entity            | `vector: list[float]`, `valid: bool`               | Needs `nodes: np.ndarray (N, C)` or keep flat and reshape | MUST     |
| `SequenceWindow` entity          | `features: list[list[float]]`                      | Must change to `(T, N, C)` array                          | MUST     |
| `SequenceDatasetBuilder`         | Uses `LandmarkFeatureTransformer` internally       | Must remove transformer, accept raw holistic kpts         | MUST     |
| `HandSelector`                   | Selects one hand from `MultiHandPacket`            | **Retire** — graph includes both hands + body             | MUST     |
| `MediaPipeHandsExtractor`        | Hands only                                         | **Retire or supplement** with Holistic extractor          | MUST     |
| `GestureDataModule`              | Validates `sequences.ndim == 3`                    | Must accept `ndim == 4` `(N, T, N_nodes, C)`              | MUST     |
| `SequenceTensorDataset`          | Validates `ndim == 3`                              | Must accept `ndim == 4`                                   | MUST     |
| `AugmentedSequenceTensorDataset` | Augments `(T, F)` arrays                           | Must augment `(T, N, C)` arrays                           | MUST     |
| `GestureAugmenter`               | Feature-index-aware (hard-coded dim indices)       | **Retire** current; redesign for node-axis operations     | MUST     |
| `train_lstm.py → _build_model()` | Hardcoded to `TwoStreamGestureLSTM`                | Must support `STGCN_LSTM`                                 | MUST     |
| `checkpoint_loader.py`           | Validates `artifact_type.startswith("two_stream")` | Must accept new artifact type                             | MUST     |
| `export_checkpoint.py`           | `artifact_type = "two_stream_lstm_inference"`      | Must emit new type + adjacency metadata                   | MUST     |
| `InferenceArtifact` dataclass    | Has `feature_dim`, `window_length`, etc.           | Add `num_nodes`, `num_coords`, `adjacency_schema`         | SHOULD   |

### 3.4 Categorised Change Register

**MUST change:**

- `src/maestro/infrastructure/mediapipe/hands_extractor.py` → replace/supplement with holistic extractor
- `src/maestro/pipeline/feature_engineering.py` → retire `LandmarkFeatureTransformer` from inference path
- `src/maestro/pipeline/hand_selection.py` → retire `HandSelector`
- `src/maestro/pipeline/buffering.py` → `SlidingWindowSequenceBuffer` must buffer graph frames `(N, C)` not flat vectors
- `src/maestro/domain/entities.py` → `FeatureFrame`, `SequenceWindow` shape contracts
- `src/maestro/training/dataset_builder.py` → complete rewrite of extraction + buffering logic
- `src/maestro/training/datamodule.py` → 4-D tensor support throughout
- `src/maestro/training/augmentation.py` → node-axis augmentation
- `src/maestro/training/train_lstm.py` → model factory, input_size validation
- `src/maestro/infrastructure/model/checkpoint_loader.py` → artifact type validation
- `src/maestro/training/export_checkpoint.py` → artifact schema
- `scripts/build_ipn_dataset.py` → holistic extraction, graph tensor output
- `notebooks/01_dataset_preparation.ipynb` → full pipeline replacement
- `notebooks/02_model_training.ipynb` → model config, loader, training cells

**SHOULD change:**

- `src/maestro/infrastructure/model/two_stream_lstm.py` → keep for compatibility, add new `stgcn_lstm.py` alongside
- `src/maestro/domain/entities.py` → `InferenceArtifact` fields
- `configs/model/two_stream_lstm.yaml` → add `configs/model/stgcn_lstm.yaml`
- `src/maestro/utils/schema_validation.py` → add `STGCNModelConfig` Pydantic schema
- `src/maestro/utils/feature_validation.py` → update schema compatibility checks

**OPTIONAL:**

- `src/maestro/app/bootstrap.py` → `HandSelector` removal (flows from MUST changes)
- `src/maestro/app/runtime.py` → runtime loop simplification
- `src/maestro/training/hf_publish.py` → update model card template for STGCN

---

## 4. Dataset Notebook Refactor Plan — `01_dataset_preparation.ipynb`

This is the highest-risk component. The notebook currently assumes a pre-built NPZ with shape `(N, 24, 147)`. The entire pipeline from video frame → feature vector must be replaced.

### 4.1 Proposed Notebook Structure

The refactored notebook must be restructured into 8 sections:

---

**Section 0 — Prerequisites & Environment** _(replaces current Cell 1–2)_

Purpose: Import libraries, define repo root, verify IPN dataset presence, verify MediaPipe Holistic is installed.

Required libraries added vs current: `mediapipe` (Holistic model, not just Hands), `cv2` for video decoding.

Remove from imports: any import of `LandmarkFeatureTransformer`, `FeatureFrame`, `SequenceWindow`, feature-schema constants.

Add explicit check: confirm `mediapipe >= 0.10` is installed, print holistic model availability.

---

**Section 1 — Vocabulary & Graph Schema Configuration** _(replaces current Section 1 "Configuration")_

Purpose: Load vocabulary YAML (unchanged). Define adjacency matrix. Print the 67-node index map.

**Critical cell — adjacency matrix definition:**

This cell must define, in pure Python/NumPy, the adjacency list for the 67-node skeleton. The node index scheme must be documented exhaustively in the cell, because it is the contract between dataset builder, model, and inference pipeline.

Required node index map (67 nodes):

| Index range | Source                                                                                                                                                                                      | Count |
| ----------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- | ----- |
| 0–20        | MediaPipe Hands (right hand, landmark indices 0–20)                                                                                                                                         | 21    |
| 21–41       | MediaPipe Hands (left hand, landmark indices 0–20)                                                                                                                                          | 21    |
| 42–52       | MediaPipe Pose (upper body: nose=0, left/right eye inner/outer/ear=1-6, left/right shoulder=11,12, left/right elbow=13,14, left/right wrist=15,16, left/right pinky/index/thumb tips=17-22) | 11    |
| 53–66       | MediaPipe Face (14 face contour keypoints: selected subset — not full 468-landmark mesh)                                                                                                    | 14    |

> **Important:** The exact subset of pose and face keypoints must be pinned at this cell and never changed without incrementing the schema version. The cell must print the full index→name mapping as a validation output.

**Adjacency edges to define:**

- Right hand: 20 edges following standard MediaPipe hand skeleton (wrist→CMC→MCP→PIP→DIP→TIP for all 5 fingers)
- Left hand: same 20 edges, offset by 21
- Cross-hand wrist bridge: right wrist (node 0) ↔ pose right wrist (node 57); left wrist (node 21) ↔ pose left wrist (node 58)
- Pose upper body: shoulder–elbow–wrist, shoulder–shoulder, nose–shoulder connections
- Face–pose: nose (pose) ↔ face centre keypoints

Store adjacency as a `numpy.ndarray` of shape `(67, 67)` (symmetric, binary or normalised). Compute the **degree-normalised adjacency** $\hat{A} = D^{-1/2} A D^{-1/2}$ for use in graph convolution.

The adjacency array must be exported alongside the NPZ dataset as `adjacency.npy` for use during training without re-computation.

---

**Section 2 — IPN Annotation Ingestion** _(largely unchanged from `build_ipn_dataset.py` logic)_

Purpose: Load `Annot_List.txt`, parse gesture segments, apply IPN→vocabulary label mapping.

No changes required to the label mapping logic. Remove any reference to `LandmarkFeatureTransformer`.

Add: count of segments per gesture class at this stage (pre-extraction diagnostic).

---

**Section 3 — Holistic Landmark Extraction** _(replaces MediaPipe Hands extraction)_

Purpose: For each IPN video segment, extract 67 keypoints per frame using MediaPipe Holistic.

**Current state:** `MediaPipeHandsExtractor` extracts 21 keypoints from one hand. The notebook calls `build_ipn_dataset.py` which uses `HandSelector` + `LandmarkFeatureTransformer`.

**Required changes:**

1. **Replace MediaPipe Hands with MediaPipe Holistic pipeline.** `mp.solutions.holistic.Holistic(static_image_mode=False, min_detection_confidence=0.5, min_tracking_confidence=0.5)` must be used. This yields: `results.right_hand_landmarks` (21), `results.left_hand_landmarks` (21), `results.pose_landmarks` (33), `results.face_landmarks` (468).

2. **Keypoint extraction function:** Define a function `extract_67_keypoints(results, image_size) → np.ndarray (67, 3)`. This function must:
   - Extract right hand: if `results.right_hand_landmarks` is not None, take landmarks[0–20] as x,y,z; else fill with zeros and mark invalid.
   - Extract left hand: if `results.left_hand_landmarks` is not None, take landmarks[0–20]; else zeros.
   - Extract pose subset (11 nodes): select the pinned indices from `results.pose_landmarks.landmark` list.
   - Extract face subset (14 nodes): select pinned face contour indices from `results.face_landmarks.landmark`.
   - Stack into `(67, 3)` array.

3. **Missing keypoint handling:** When a hand is not detected, fill its 21 nodes with zeros. Do **not** skip the frame. The model must be robust to partially missing nodes. Store a per-frame `valid_mask: (67,)` boolean array alongside coordinates. This mask will be used to zero-attention missing nodes during graph convolution.

4. **Per-video caching:** Save extracted raw `(F, 67, 3)` arrays to `data/IPN/pose_cache/<video_stem>_holistic.npz`. Do not cache feature vectors (they are not computed). The cache key changes from current hands-only format.

5. **Coordinate convention:** All x,y,z values are MediaPipe's normalised image coordinates (x,y ∈ [0,1], z ∈ ≈[-1,1] relative depth). Do NOT denormalise to pixel space. Wrist-relative normalisation is applied in Section 4.

---

**Section 4 — Graph Frame Construction & Normalisation** _(new section, does not exist in current notebook)_

Purpose: Convert raw `(T, 67, 3)` per-segment arrays into normalised graph tensors.

**Normalisation procedure (per frame, must be identical at training and inference time):**

1. **Wrist-anchor normalisation:** Subtract the mean of right-wrist (node 0) and left-wrist (node 21) from all node coordinates. This makes the representation translation-invariant with respect to hand position.
2. **Scale normalisation:** Divide by the Euclidean distance between node 0 and node 21 (inter-wrist distance). If this distance is below a threshold (e.g., 0.01, one hand not visible), fall back to dividing by the mean hand size computed from the detected hand.
3. **Result:** Each frame is a `(67, 3)` tensor in a canonical coordinate space.

**This normalisation must be documented as the training–inference contract.** It replaces the wrist-relative normalisation currently performed inside `LandmarkFeatureTransformer`.

Add a visualisation cell: plot a single frame's graph (nodes + edges) using matplotlib to visually verify the skeleton structure and normalisation.

---

**Section 5 — Temporal Sequencing** _(replaces current sliding-window logic)_

Purpose: Apply sliding window over the per-segment `(F, 67, 3)` arrays to produce `(T, 67, 3)` sequence samples.

**Changes from current:**

- The sliding window now operates on `(N, C)` graph frames, not `(F,)` feature vectors.
- Window output shape: `(T=24, 67, 3)` per sample.
- The `SlidingWindowSequenceBuffer` must be adapted or its logic replicated inline in this notebook section (see Section 5.2 below).
- **Label assignment logic:** identical to current (last-frame label or majority vote, with trailing-purity guard). Do not change.
- **Valid-frame logic:** a frame is "valid" if at least one hand is detected (i.e., not all zeros in nodes 0–41). The existing `min_valid_ratio=0.8` threshold applies. Store per-frame valid mask alongside.

Add: histogram of per-segment valid_ratio distribution before and after windowing.

---

**Section 6 — Dataset Export** _(replaces current save cell)_

Purpose: Save the graph dataset in the new NPZ schema.

**Required NPZ schema:**

```
sequences:        float32 (N, T, 67, 3)    — normalised graph frames
labels:           object  (N,)             — string class names
label_ids:        int64   (N,)             — integer class indices
recording_ids:    object  (N,)             — IPN video stem
sample_ids:       object  (N,)             — unique sample identifier
valid_masks:      bool    (N, T, 67)       — per-node, per-frame detection validity
metadata_json:    object  scalar           — JSON string (see below)
adjacency:        float32 (67, 67)         — degree-normalised adjacency matrix
```

**Required `metadata_json` fields:**

```json
{
  "graph_schema_version": "graph-schema-v1",
  "num_nodes": 67,
  "num_coords": 3,
  "window_length": 24,
  "window_step": 3,
  "node_index_map": {
    "right_hand": [0, 20],
    "left_hand": [21, 41],
    "pose": [42, 52],
    "face": [53, 66]
  },
  "normalisation": "wrist_anchor_scale",
  "vocabulary": [
    "open_palm_hold",
    "swipe_right",
    "swipe_left",
    "swipe_up",
    "swipe_down",
    "zoom_in",
    "zoom_out",
    "unknown"
  ],
  "min_valid_ratio": 0.8
}
```

**Filename convention:** `data/processed/ipn_graph_v1.npz`

Do NOT save a flat `(N, T, 147)` NPZ. The old schema is fully retired.

---

**Section 7 — Dataset Validation** _(new section)_

Purpose: Automated shape and sanity checks.

Required checks:

1. `sequences.shape == (N, T, 67, 3)` — assert exactly.
2. `label_ids` covers all 8 classes — assert `set(label_ids) == set(range(8))`.
3. Per-class sequence counts — flag any class with < 50 samples.
4. `valid_masks` mean ≥ 0.8 per sample across the whole dataset.
5. Coordinate range: `sequences` values in `[-5, 5]` after normalisation (spot-check on a random subsample).
6. Adjacency matrix: assert symmetric, assert diagonal zero, assert connected (one component) via `scipy.sparse.csgraph.connected_components`.
7. Recording-level split preview: show that a group-aware train/val/test split would not leak recordings.

---

**Section 8 — Dataset Analysis & Visualisation** _(adapted from current Sections 3–8)_

The existing analysis cells (class distribution, recording coverage, valid_ratio distribution) are largely portable. Required adaptations:

- Replace `F = sequences.shape[2]` with `_, T, N, C = sequences.shape`.
- Replace feature correlation heatmaps (meaningless for raw coordinates) with: per-node mean position maps, skeleton overlay on sample frames, per-class PCA of flattened `(T×N×C,)` vectors.
- Keep: class imbalance bar chart, recording coverage table, train/val/test split verification.

---

## 5. Dataset Builder Refactor Plan — `dataset_builder.py`

### 5.1 `LabeledLandmarkFrame` — replace

**Current:** `landmarks: list[list[float]]` — 21×3 from a single hand.

**Required:** `keypoints: np.ndarray (67, 3)` — all 67 holistic keypoints. `valid_mask: np.ndarray (67,)` — boolean per-node validity. `recording_id: str`, `timestamp_ms: int`, `label: str` — unchanged.

Rename to `LabeledGraphFrame` to make the schema break explicit and prevent accidental reuse of the old type.

### 5.2 `SequenceSample` — update shape contract

**Current:** `sequence: list[list[float]]` — `(T, F)`.

**Required:** `sequence: np.ndarray (T, N, C)` — graph tensor. `valid_mask: np.ndarray (T, N)` — detection validity.

### 5.3 `SequenceDatasetBuilder` — replace core logic

**Remove entirely:**

- `self._transformer = LandmarkFeatureTransformer(...)` — no feature engineering.
- The `LandmarkPacket` construction and `.transform()` call in `add_frame()`.

**Replace `add_frame()` with:**

1. Accept `LabeledGraphFrame` (not `LabeledLandmarkFrame`).
2. Normalise the `(67, 3)` keypoint array using the wrist-anchor-scale procedure (identical to Section 4 of the notebook).
3. Push the normalised `(67, 3)` array directly into the buffer.
4. Buffer now stores graph frames, not `FeatureFrame` objects.

**`save_npz()` — update to emit 4-D sequences:**

`sequences = np.stack([s.sequence for s in self._samples], axis=0)` → shape `(N, T, 67, 3)`.

Add `adjacency` array to the NPZ payload (pass it in at construction time).

Update `metadata` dict: replace `feature_dim` / `feature_schema_version` with `num_nodes`, `num_coords`, `graph_schema_version`.

**`load_npz()` — update:**

Load `adjacency` from NPZ. Accept `sequences.ndim == 4`. Update `metadata` parsing.

### 5.4 `SlidingWindowSequenceBuffer` — adapt

The buffer currently stores `FeatureFrame` (a dataclass with `vector: list[float]`).

**Option A (minimal invasive):** Wrap the `(67, 3)` graph frame as a fake `FeatureFrame` by flattening to `(201,)` and reshaping on exit. **Strongly discouraged** — it hides a shape contract violation.

**Option B (correct):** Introduce a `GraphFrame` domain entity (dataclass with `nodes: np.ndarray (N, C)`, `valid_mask: np.ndarray (N,)`, `timestamp_ms: int`, `valid: bool`). Modify `SlidingWindowSequenceBuffer` to accept a generic frame type, or create a parallel `GraphSequenceBuffer`. `window()` returns a `(T, N, C)` numpy array.

**Recommendation:** Option B. The inference runtime will also need this buffer, so the shape contract must be clean.

### 5.5 Build Script — `scripts/build_ipn_dataset.py`

**Remove:** `from maestro.infrastructure.mediapipe.hands_extractor import MediaPipeHandsExtractor`, `from maestro.pipeline.hand_selection import HandSelector`, `from maestro.pipeline.feature_engineering import LandmarkFeatureTransformer`.

**Add:** New `HolisticExtractor` infrastructure class (see Section 8 of this plan). The build script becomes a thin orchestrator that calls the holistic extractor per video frame and feeds `LabeledGraphFrame` objects to the `SequenceDatasetBuilder`.

**Cache format change:** Current cache is `<stem>.npz` with hands-only landmarks. New cache must be `<stem>_holistic.npz` with `(F, 67, 3)` raw arrays. Old cache files are incompatible — clear the `data/IPN/cache/` directory.

---

## 6. Training Pipeline Refactor Plan

### 6.1 `datamodule.py`

**`SequenceTensorDataset.__init__()`:**

The guard `if sequences.ndim != 3: raise ValueError(...)` must change to accept `ndim == 4`. The shape validation must become:

```
assert sequences.ndim in (3, 4), "sequences must be (N, T, F) or (N, T, N_nodes, C)"
```

However, since the two schemas are incompatible with one another, the cleanest approach is to create a `GraphSequenceTensorDataset` class that validates `ndim == 4` and returns `(B, T, N, C)` tensors. Keep `SequenceTensorDataset` for backward-compat with the two-stream LSTM baseline.

**`AugmentedSequenceTensorDataset.__init__()`:**

Same — replace with `AugmentedGraphSequenceTensorDataset` accepting `(T, N, C)` arrays and using the new augmenter (see 6.3).

**`GestureDataModule.setup()`:**

Replace `sequences.ndim != 3` guard. Update `feature_dim` property: for graph datasets, return `num_nodes * num_coords`. Load `adjacency` from NPZ and expose it as a `self.adjacency` property for consumption by `_build_model()`.

**`GestureDataModule.metadata` properties:**

Add `num_nodes`, `num_coords`, `adjacency_schema` alongside existing properties. The `feature_dim` property must correctly compute `67 * 3 = 201` for compatibility with any code that reads it as a flat dimension (though model config should use `num_nodes`/`num_coords` directly).

### 6.2 `train_lstm.py`

**`ModelConfig` dataclass:**

Add fields: `num_nodes: int`, `num_coords: int`, `num_stgcn_layers: int`, `stgcn_channels: list[int]`.

**`_build_model()`:**

Remove the `if not config.model_name.startswith("two_stream")` guard. Replace with a model registry dispatch:

```python
if config.model_name.startswith("stgcn"):
    # instantiate STGCN_LSTM
elif config.model_name.startswith("two_stream"):
    # instantiate TwoStreamGestureLSTM
else:
    raise ValueError(...)
```

Pass `adjacency` tensor (from datamodule) into `STGCN_LSTM` constructor.

**`train_lstm()` function:**

The existing `datamodule.feature_dim != config.model.input_size` validation must be updated. For graph models, validation becomes: `datamodule.num_nodes != config.model.num_nodes or datamodule.num_coords != config.model.num_coords`.

**`_train_one_epoch()`:** No changes required. `model(batch_x)` is agnostic to shape — the `STGCN_LSTM` forward pass accepts `(B, T, N, C)` and returns `(B, num_classes)`.

### 6.3 `augmentation.py`

**Remove entirely:** `_X_COORD_DIMS`, `_POS_START`, `_POS_END`, etc. — all feature-schema-v5 specific constants.

**Replace `GestureAugmenter.augment(sequence: np.ndarray (T, F))` with augmentation on `(T, N, C)` arrays:**

- **Temporal scaling:** unchanged logic — resample along axis 0 (time). Works identically for any shape with T as dim 0.
- **Spatial jitter:** instead of indexing specific feature dims, add Gaussian noise to all entries in the `C` coordinate axis: `sequence += rng.normal(0, sigma, size=sequence.shape)`. This is simpler and semantically correct for raw coordinates.
- **Mirror flip:** negate `sequence[:, :, 0]` (the x-coordinate of all nodes) without requiring a handcrafted index list. Swap right-hand node block (0–20) and left-hand node block (21–41) along the node axis to reflect bilateral symmetry. This also requires swapping the corresponding `swipe_left` ↔ `swipe_right` labels in the dataset (same as current behaviour).

**`AugmentationConfig` dataclass:** Remove `spatial_jitter_sigma` (or reinterpret it as uniform across all `C` dims). Remove `mirror_flip_prob` (or retain with updated axis semantics). No structural change needed.

### 6.4 `export_checkpoint.py`

**`artifact_type`:** Change from `"two_stream_lstm_inference"` to `"stgcn_lstm_inference"`.

**Payload additions:**

```python
"graph_schema": {
    "graph_schema_version": "graph-schema-v1",
    "num_nodes": 67,
    "num_coords": 3,
    "adjacency": adjacency.tolist(),  # serialise as list for portability
}
```

Remove `feature_dim` from `feature_schema` (or retain as `num_nodes * num_coords = 201` for backward-compat with any loading code that reads it).

### 6.5 `checkpoint_loader.py`

**`_build_model_from_artifact()`:**

Remove `if not (artifact_type.startswith("two_stream") or ...)` guard.

Add dispatch:

```python
if artifact_type == "stgcn_lstm_inference":
    # load adjacency from payload["graph_schema"]["adjacency"]
    # instantiate STGCN_LSTM
elif artifact_type.startswith("two_stream"):
    # existing TwoStreamGestureLSTM path (preserved for old artifacts)
```

**`InferenceArtifact` dataclass:** Add fields `num_nodes: int`, `num_coords: int`, `adjacency: np.ndarray`.

---

## 7. Training Notebook Refactor Plan — `02_model_training.ipynb`

Cell-by-cell analysis against current notebook structure:

### Cell 1 (lines 28–52) — Imports & Environment

**Change:** Add `from maestro.infrastructure.model.stgcn_lstm import STGCN_LSTM`. Remove `gesture_control` import alias (it uses the wrong package path). Verify `torch.__version__ >= 2.0`.

### Cell 2 (lines 61–103) — Configuration

**Change:** Replace `MODEL_TYPE = "two_stream"` with `MODEL_TYPE = "stgcn"`. Replace `_CONFIG_MAP` to point to `configs/model/stgcn_lstm.yaml`. Replace dataset candidates: prefer `ipn_graph_v1.npz` (new schema). Add `N_NODES = 67; N_COORDS = 3` constants.

### Cell 3 (lines 109–138) — Load Training Configuration

**Change:** After loading config, add: load `adjacency` from NPZ dataset via `np.load(DATASET_PATH)["adjacency"]`. Convert to `torch.FloatTensor`. Print adjacency shape for verification.

### Cell 4 (lines 144–160) — Load Dataset

**Change:** Replace `SequenceTensorDataset` with `GraphSequenceTensorDataset`. Assert `sequences.shape[2] == 67` and `sequences.shape[3] == 3`. Print `(N, T, N_nodes, C)` shape.

### Cell 5 (lines 163–207) — Data Splits Visualisation

**Change:** Replace `F = sequences.shape[2]` with `_, T, N_nodes, C = sequences.shape`. Feature correlation heatmap: replace with node-position mean visualisation. Class distribution bar chart: no change.

### Cell 6 (lines 213–246) — Build Model

**Replace entirely:**

```python
model = STGCN_LSTM(
    num_nodes=67,
    in_channels=3,
    adjacency=adjacency_tensor,
    stgcn_channels=[64, 128],  # per STGCN layer
    lstm_hidden=128,
    lstm_layers=2,
    num_classes=8,
    dropout=0.3,
)
```

Print parameter count. Target: < 500K parameters for CPU real-time viability.

### Cell 7 (lines 252–275) — Training Loop

**No structural change required.** `train_lstm()` function receives graph datamodule; model receives `(B, T, 67, 3)` batches. Verify `grad_clip_norm` is active (unchanged).

### Cell 8 (lines 281–345) — Learning Curves

**No change required.** Loss/accuracy vs epoch plots are model-agnostic.

### Cell 9 (lines 351–378) — Validation Evaluation

**No change required.** `evaluate_model()` operates on logits.

### Cell 10 (lines 384–415) — Confusion Matrix

**No change required.**

### Cell 11 (lines 421–473) — Per-Class Metrics

**No change required.** Flag any class with recall < 70%.

### Cell 12 (lines 483–560) — Export

**Change:** Pass adjacency into export payload. `artifact_type` becomes `"stgcn_lstm_inference"`. Filename: `<version>_stgcn_inference.pt`.

### Cell 13 (lines 566–606) — Hugging Face Publish

**Change:** Repo target: `ntsrigaud/maestro-stgcn` (not the existing LSTM repo). Update model card template to describe STGCN-LSTM, graph schema version, 67 nodes, adjacency structure. `hf_publish.py` must be updated to pass adjacency in the model card metadata.

### New Required Section — Graph Dataset Inspection (insert before Cell 4)

Add a dedicated cell to visualise a random sample's graph tensor: plot skeleton overlaid on a blank frame, show the 67-node positions using matplotlib, overlay edges from the adjacency matrix. This confirms extraction correctness before training.

---

## 8. Codebase Integration Plan

### 8.1 New Files to Create

| File                                                         | Purpose                                                                          |
| ------------------------------------------------------------ | -------------------------------------------------------------------------------- |
| `src/maestro/infrastructure/mediapipe/holistic_extractor.py` | `HolisticLandmarkExtractor` — wraps MediaPipe Holistic, returns `(67, 3)` arrays |
| `src/maestro/infrastructure/model/stgcn_lstm.py`             | `STGCN_LSTM` PyTorch module (new implementation)                                 |
| `configs/model/stgcn_lstm.yaml`                              | STGCN-LSTM architecture and training hyperparameters                             |
| `src/maestro/domain/entities.py` → add `GraphFrame`          | New domain entity for graph-structured frame data                                |
| `src/maestro/pipeline/graph_buffering.py`                    | `GraphSequenceBuffer` — buffers `GraphFrame` objects                             |
| `src/maestro/pipeline/graph_schema.py`                       | Single source of truth for node index map, adjacency construction, normalisation |

### 8.2 `configs/model/stgcn_lstm.yaml` — Required Structure

```yaml
model:
  model_name: stgcn_lstm_v1
  num_nodes: 67
  num_coords: 3
  stgcn_channels: [64, 128] # output channels per STGCN layer
  num_stgcn_layers: 2
  lstm_hidden: 128
  lstm_layers: 2
  dropout: 0.3
  num_classes: 8

training:
  epochs: 60 # longer convergence expected
  batch_size: 32 # smaller due to 4-D tensors
  learning_rate: 0.0005 # lower initial LR for graph models
  weight_decay: 0.0001
  grad_clip_norm: 1.0
  early_stopping_patience: 10
  early_stopping_min_delta: 0.0001
  seed: 42
  num_workers: 0
  class_weighting: true

splits:
  train_ratio: 0.7
  val_ratio: 0.15
  test_ratio: 0.15
  stratified: true

scheduler:
  enabled: true
  factor: 0.5
  patience: 5
  min_lr: 0.000001

export:
  checkpoint_dir: models/checkpoints
  exported_dir: models/exported
  graph_schema_version: graph-schema-v1

augmentation:
  enabled: true
  temporal_scale_range: [0.8, 1.2]
  spatial_jitter_sigma: 0.003 # lower: raw coords more sensitive than normalised features
  apply_prob: 0.5
  mirror_flip_prob: 0.5
```

### 8.3 Training–Inference Parity Requirements

The following components must be **identical** between training and runtime inference:

| Component                    | Training path                                   | Inference path                                       |
| ---------------------------- | ----------------------------------------------- | ---------------------------------------------------- |
| Holistic keypoint extraction | `HolisticLandmarkExtractor`                     | Same class                                           |
| 67-node index mapping        | Defined once in `graph_schema.py`               | Imported by extractor                                |
| Wrist-anchor normalisation   | Applied in `SequenceDatasetBuilder.add_frame()` | Applied in `GraphSequenceBuffer` or runtime pipeline |
| Adjacency matrix             | Loaded from NPZ at training time                | Loaded from `InferenceArtifact.adjacency` at runtime |
| Window length (T=24)         | `SlidingWindowSequenceBuffer`                   | `GraphSequenceBuffer`                                |
| Window step                  | Configured in `stgcn_lstm.yaml`                 | `InferenceArtifact.window_step`                      |

**Recommended:** Create `src/maestro/pipeline/graph_schema.py` as a single source of truth for the node index map and adjacency construction. Import this module in both the training dataset builder and the runtime pipeline.

### 8.4 Runtime Pipeline Changes (Inference Path)

The inference runtime (`src/maestro/app/runtime.py`, `bootstrap.py`) currently sequences:

```
FrameSource → HandsExtractor → HandSelector → FeatureTransformer → SequenceBuffer → GestureModel → Stabilizer → ActivationController → CommandDispatcher
```

The new sequence must be:

```
FrameSource → HolisticExtractor → GraphNormaliser → GraphSequenceBuffer → GestureModel → Stabilizer → ActivationController → CommandDispatcher
```

**Removed components:** `HandSelector`, `LandmarkFeatureTransformer`.  
**New components:** `HolisticExtractor`, `GraphNormaliser` (applies wrist-anchor normalisation inline).

The `IFeatureTransformer` interface must be replaced with a `IGraphNormaliser` interface, or the existing interface must be generalised to accept arbitrary frame types.

---

## 9. Performance & Risk Analysis

### 9.1 Expected Gains

| Metric                 | Current (Two-Stream LSTM) | Expected (STGCN-LSTM)                  | Basis                                                                                                                                              |
| ---------------------- | ------------------------- | -------------------------------------- | -------------------------------------------------------------------------------------------------------------------------------------------------- |
| Overall accuracy       | ~79.8%                    | 85–90%                                 | Graph models on skeletal data routinely exceed feature-engineered baselines by 6–12% on similar-scale datasets                                     |
| Macro-F1               | ~79.3%                    | 84–89%                                 | Spatial relational modelling eliminates the main failure mode: gestures where hand shape matters as much as trajectory (e.g., zoom_in vs zoom_out) |
| Worst-class recall     | <70%                      | >75%                                   | Two-hop graph conv captures finger configurations not representable in flat velocity vectors                                                       |
| Translation invariance | Wrist-relative only       | Structural (graph topology preserving) | Graph conv is inherently permutation-invariant and translation-equivariant                                                                         |

### 9.2 Risks

**Risk 1: CPU inference latency (CRITICAL)**

The primary real-time risk. The current two-stream LSTM has ~182K parameters and operates on a 147-dim flat vector. The STGCN-LSTM will operate on `(T=24, 67, 3)` tensors with graph conv at each step.

- MediaPipe Holistic is significantly slower than MediaPipe Hands. On CPU: Hands ≈ 20–30ms/frame; Holistic ≈ 60–90ms/frame at 720p. This alone may break 30 FPS targets.
- Graph conv (matrix multiply against 67×67 adjacency): at `(B=1, T=24, N=67, C=64)`, a single GCN layer is `67×67 × 64 ≈ 288K` multiply-accumulates. For 2 layers: ~576K MACs. This is fast but the batched tensor operations add overhead.

**Mitigation strategies:**

1. **MediaPipe Holistic model complexity reduction:** Use `model_complexity=0` (lightweight model) for real-time inference. Use `model_complexity=1` only for dataset building offline.
2. **Downscale input video** to 480p before Holistic extraction in the runtime pipeline (not during dataset building, where accuracy matters more).
3. **Reduce STGCN layer count** to 1 if latency testing shows >40ms inference time on the target CPU.
4. **Pre-compute adjacency multiplication** as a fixed registered buffer in the model, not recomputed per batch.
5. **LSTM layer count:** Use 2 layers maximum (not the reference's 4).
6. **Benchmark before committing:** Run `scripts/benchmark_system.py` with the new pipeline and set a hard latency gate of < 50ms end-to-end before shipping.

**Risk 2: IPN dataset scale vs model complexity (HIGH)**

The IPN dataset provides ~7,000–10,000 gesture sequences (exact count depends on the annotation-to-sequence pipeline). The STGCN-LSTM with 67 nodes and 2 STGCN layers will have more parameters than the current 182K-param model — potentially 400K–800K depending on channel widths.

Overfitting is a serious risk at this scale.

**Mitigation strategies:**

1. **DropEdge regularisation:** randomly zero out edges in the adjacency matrix during training (analogous to Dropout for graphs). Drop probability 0.1–0.2.
2. **Aggressive data augmentation:** Spatial jitter, temporal scaling, mirror flip (left/right swap) — all defined in Section 6.3. Additionally: random node dropout (zero out random non-wrist nodes during training to simulate partial occlusion).
3. **Keep STGCN channels small:** `[32, 64]` instead of `[64, 128]`. Prioritise training stability over parameter capacity.
4. **Weight decay:** 1e-4 (current value) is appropriate; increase to 5e-4 if validation loss diverges after epoch 20.
5. **Early stopping patience = 10:** longer than current (7) to account for slower convergence on graph tasks.

**Risk 3: Training–inference parity break (HIGH)**

The current parity guarantee comes from sharing `LandmarkFeatureTransformer` and `SlidingWindowSequenceBuffer` between training and inference. These components are being retired.

The new parity depends on the normalisation procedure being implemented identically in `SequenceDatasetBuilder` and the runtime `GraphNormaliser`. Any divergence (e.g., different wrist-anchor formula, different keypoint ordering) silently degrades inference accuracy without raising an error.

**Mitigation strategies:**

1. Create `src/maestro/pipeline/graph_schema.py` as the single source of truth for all graph-related constants (node indices, normalisation formula, adjacency construction). Both paths must import from this module — never inline constants.
2. Add a `src/maestro/utils/feature_validation.py` check that verifies `graph_schema_version` in the artifact matches the current runtime schema version. Fail loudly (not silently) on mismatch.
3. Write an integration test that extracts a known video frame, normalises it via both paths, and asserts coordinate arrays are bitwise identical.

**Risk 4: MediaPipe Holistic deprecation**

MediaPipe Holistic is deprecated in MediaPipe ≥ 0.10.14 in favour of the task-based API (`HandLandmarker`, `PoseLandmarker`, `FaceLandmarker`). The task API requires three separate model files and separate inference passes.

**Mitigation:** If `mediapipe >= 0.10.14`, use the task-based API with three separate landmarkers. Define a `HolisticAdapter` that wraps all three and presents a unified interface. This is an engineering overhead but necessary for forward compatibility.

**Risk 5: Annotation synchronisation for holistic keypoints**

IPN annotations give frame-level gesture labels for the original video frames. The current pipeline extracts hand landmarks frame-by-frame in video order (preserving MediaPipe's temporal tracker). Holistic extraction must follow the same sequential order — do not process frames in parallel within a video, as MediaPipe's tracker depends on temporal continuity.

**Mitigation:** The existing `build_ipn_dataset.py` already processes frames sequentially within each video (`Phase 1`). Preserve this sequential order when porting to holistic extraction. Parallelism is only acceptable across videos, not within a video.

---

## 10. Final Recommendation

### 10.1 Recommended Migration Path

Execute in strict phase order. Do not begin Phase 2 before Phase 1 is verified with shape assertions.

**Phase 1 — Data Infrastructure (2–3 weeks, highest risk)**

1. Implement `graph_schema.py` — pin node index map, adjacency construction, normalisation function.
2. Implement `HolisticLandmarkExtractor` in `infrastructure/mediapipe/`.
3. Refactor `dataset_builder.py` — `LabeledGraphFrame`, `SequenceDatasetBuilder` overhaul.
4. Refactor `01_dataset_preparation.ipynb` per Section 4 of this plan.
5. Rebuild IPN dataset → `ipn_graph_v1.npz`. Verify shape `(N, T=24, 67, 3)`, all 8 classes present, class counts reasonable.
6. Implement Section 7 validation cell. All checks must pass before proceeding.

**Phase 2 — Model Implementation (1 week)**

1. Implement `STGCN_LSTM` in `infrastructure/model/stgcn_lstm.py` using PyTorch, following the conceptual design from the reference but adapted for 67 nodes and classification.
2. Unit test: forward pass with `(2, 24, 67, 3)` input → `(2, 8)` logits. Test gradient flow.
3. Add `stgcn_lstm.yaml` config.

**Phase 3 — Training Pipeline Integration (1 week)**

1. Refactor `datamodule.py` — 4-D tensor support, `GraphSequenceTensorDataset`.
2. Refactor `augmentation.py` — node-axis augmentation.
3. Refactor `train_lstm.py` — model factory dispatch.
4. Refactor `02_model_training.ipynb` per Section 7 of this plan.
5. Run first training run. Expect > 10 epochs to convergence on graph data.

**Phase 4 — Export & Inference Integration (1 week)**

1. Refactor `export_checkpoint.py`, `checkpoint_loader.py`.
2. Replace `HandSelector` + `LandmarkFeatureTransformer` in runtime pipeline with `GraphNormaliser` + `GraphSequenceBuffer`.
3. Latency benchmark on target CPU. If > 50ms end-to-end, reduce model complexity.
4. Publish to `ntsrigaud/maestro-stgcn` on Hugging Face.

**Phase 5 — Validation & Regression (0.5 weeks)**

1. Run full evaluation: confusion matrix, per-class recall. All classes must exceed 70% recall.
2. Run `scripts/benchmark_system.py` — P95 latency < 50ms.
3. Parity integration test (normalisation path equivalence).

### 10.2 Do Not Do

- Do not attempt to reuse the reference TensorFlow code directly.
- Do not store engineered features alongside raw coordinates in the NPZ — keep the new schema clean.
- Do not merge the old `(N, T, 147)` and new `(N, T, 67, 3)` datasets by concatenation or flattening — they are incompatible at the semantic level.
- Do not use MediaPipe Holistic's full 468-point face mesh — select a minimal face subset (14 keypoints) and pin those indices in `graph_schema.py`.
- Do not process video frames out of sequential order within a single video clip.
- Do not hardcode `num_nodes=67` anywhere except `graph_schema.py`. All other modules must read this value from the schema or the config.

### 10.3 Success Criteria

| Criterion             | Gate                                                                                          |
| --------------------- | --------------------------------------------------------------------------------------------- |
| Dataset shape         | `ipn_graph_v1.npz` sequences: `(N, 24, 67, 3)`, all 8 classes                                 |
| Training convergence  | Val macro-F1 improves over 20 epochs without divergence                                       |
| Accuracy target       | Test macro-F1 ≥ 84%                                                                           |
| Worst-class recall    | All classes ≥ 72% recall                                                                      |
| CPU inference latency | P95 < 50ms end-to-end (Holistic + buffer + model + stabiliser)                                |
| Parity test           | Training and runtime normalisation produce bitwise-identical coordinates on shared test frame |
| Artifact integrity    | `checkpoint_loader.py` loads new artifact without errors; model runs in eval mode             |
