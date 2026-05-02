# STGCN-LSTM Migration — Phase-by-Phase Implementation Plan

**System:** Gesture Control System (Maestro)  
**Branch:** `test/dual-stream-stgcn-lstm`  
**Prepared:** 2026-05-01  
**Supersedes:** `stgcn-lstm-migration-audit.md` (audit source)  
**Scope:** Full system migration from `TwoStreamGestureLSTM` (147-dim flat vector) to `STGCN_LSTM` (67-node graph, `(B, T, N, C)` tensors)

---

## Prerequisites

Before starting any phase, ensure the following are in place:

- `conda activate maestro` is active with all existing dependencies installed.
- `mediapipe >= 0.10` is available (`pip show mediapipe` to verify).
- The IPN Hand video dataset is present at the path configured in `scripts/build_ipn_dataset.py`.
- The existing `data/processed/gesture_dataset_v2.npz` is **archived** but not deleted (baseline comparison).
- The unit test suite passes on the current main: `pytest tests/unit/ -q`.

---

## Phase Overview

| Phase | Title                                 | Risk     | Duration  |
| ----- | ------------------------------------- | -------- | --------- |
| **1** | Graph Schema & Data Infrastructure    | Critical | 2–3 weeks |
| **2** | STGCN-LSTM Model Implementation       | High     | 1 week    |
| **3** | Training Pipeline Integration         | High     | 1 week    |
| **4** | Export, Loading & Inference Runtime   | High     | 1 week    |
| **5** | Validation, Benchmarking & Regression | Medium   | 3–4 days  |

**Rule:** Do not begin a phase until all acceptance criteria of the previous phase are verified. Shape assertions and unit tests are the gates between phases.

---

## Phase 1 — Graph Schema & Data Infrastructure

This is the highest-risk phase. The model swap in Phase 2 is straightforward; the data pipeline is where semantic errors silently degrade accuracy. Invest disproportionate effort here.

### 1.1 Create `src/maestro/pipeline/graph_schema.py` _(new file)_

This module is the **single source of truth** for all graph-related constants. Every other module that references node indices, adjacency, or normalisation must import from here — never inline constants elsewhere.

**Contents to implement:**

```
NODE_INDEX_MAP = {
    "right_hand":  (0,  20),   # MediaPipe right hand landmarks 0–20
    "left_hand":   (21, 41),   # MediaPipe left hand landmarks 0–20 (offset +21)
    "pose":        (42, 52),   # 11 selected MediaPipe Pose landmarks
    "face":        (53, 66),   # 14 selected MediaPipe Face contour landmarks
}
NUM_NODES: int = 67
NUM_COORDS: int = 3
GRAPH_SCHEMA_VERSION: str = "graph-schema-v1"
WINDOW_LENGTH: int = 24
WINDOW_STEP: int = 3
MIN_VALID_RATIO: float = 0.8
```

**Pose landmark subset (11 nodes, indices 42–52):**

| Graph index | MediaPipe Pose index | Name               |
| ----------- | -------------------- | ------------------ |
| 42          | 0                    | nose               |
| 43          | 2                    | left eye inner     |
| 44          | 5                    | right eye inner    |
| 45          | 7                    | left ear           |
| 46          | 8                    | right ear          |
| 47          | 11                   | left shoulder      |
| 48          | 12                   | right shoulder     |
| 49          | 13                   | left elbow         |
| 50          | 14                   | right elbow        |
| 51          | 15                   | left wrist (pose)  |
| 52          | 16                   | right wrist (pose) |

**Face landmark subset (14 nodes, indices 53–66):** Select 14 face contour points from MediaPipe's 468-point mesh. Pin these indices permanently in `graph_schema.py`. Recommended: jawline midpoint, chin, left/right cheek, forehead centre, and 8 perioral landmarks.

**Functions to implement:**

```python
def build_adjacency() -> np.ndarray:
    """Returns a binary symmetric adjacency matrix of shape (67, 67)."""
    # Right hand: 20 edges per MediaPipe hand skeleton
    # Left hand: same 20 edges, offset by 21
    # Cross-wrist bridges: (0, 52), (21, 51)  [hand wrists to pose wrists]
    # Pose: shoulder–elbow–wrist chains, shoulder–shoulder, nose–shoulders
    # Face–pose: nose (42) to nearest face keypoints
    ...

def normalise_adjacency(A: np.ndarray) -> np.ndarray:
    """Degree-normalised adjacency: D^{-1/2} A D^{-1/2}."""
    ...

def normalise_graph_frame(frame: np.ndarray) -> np.ndarray:
    """
    Wrist-anchor normalisation applied identically at training and inference.

    Args:
        frame: (67, 3) raw MediaPipe normalised coordinates.

    Returns:
        (67, 3) array in canonical coordinate space.

    Procedure:
        1. anchor = mean(frame[0], frame[21])  [mean of both wrists]
        2. frame -= anchor
        3. scale = ||frame[0] - frame[21]||  [inter-wrist distance]
        4. if scale < 0.01: scale = fallback (mean hand span from detected nodes)
        5. frame /= scale
    """
    ...
```

**Acceptance criterion:** `build_adjacency()` returns a `(67, 67)` symmetric matrix with zero diagonal. `scipy.sparse.csgraph.connected_components` confirms the graph is fully connected (1 component).

---

### 1.2 Create `src/maestro/infrastructure/mediapipe/holistic_extractor.py` _(new file)_

Replaces the hands-only `MediaPipeHandsExtractor`. Do not modify the existing `hands_extractor.py` — it will be retired but kept until Phase 4.

**Class:** `HolisticLandmarkExtractor`

**Interface to implement:**

```python
class HolisticLandmarkExtractor:
    def __init__(self, *, model_complexity: int = 1,
                 min_detection_confidence: float = 0.5,
                 min_tracking_confidence: float = 0.5) -> None: ...

    def extract(self, frame: FramePacket) -> HolisticPacket: ...
    def close(self) -> None: ...
```

**`HolisticPacket` dataclass** (add to `src/maestro/domain/entities.py`):

```python
@dataclass(slots=True)
class HolisticPacket:
    keypoints: np.ndarray   # shape (67, 3) — normalised image coordinates
    valid_mask: np.ndarray  # shape (67,)  — bool: True = landmark detected
    timestamp_ms: int
```

**Implementation details:**

- Use `mp.solutions.holistic.Holistic` for `mediapipe < 0.10.14`. For `mediapipe >= 0.10.14`, implement a `HolisticAdapter` that wraps three separate task-based landmarkers (`HandLandmarker`, `PoseLandmarker`, `FaceLandmarker`) and presents the same `HolisticPacket` interface.
- The `extract()` method must call `graph_schema.normalise_graph_frame()` internally before returning. The returned `HolisticPacket.keypoints` are **always normalised**.
- Missing landmarks (hand not detected) must be zero-filled in their node slots. `valid_mask` is `False` for those nodes.
- Use `model_complexity=0` for real-time inference. Use `model_complexity=1` for offline dataset building.
- Process frames **sequentially** (never in parallel within a single video) to preserve MediaPipe's temporal tracker continuity.

**Acceptance criterion:** Unit test in `tests/unit/test_holistic_extractor.py` that feeds a synthetic `FramePacket` (blank image) and asserts `packet.keypoints.shape == (67, 3)` and `packet.valid_mask.shape == (67,)`.

---

### 1.3 Add `GraphFrame` to `src/maestro/domain/entities.py` _(modify existing file)_

Add the following dataclass, replacing the role of `FeatureFrame` for graph-based pipelines:

```python
@dataclass(slots=True)
class GraphFrame:
    nodes: np.ndarray       # shape (N, C) — normalised keypoint coordinates
    valid_mask: np.ndarray  # shape (N,)   — bool per-node detection validity
    timestamp_ms: int
    valid: bool             # True if min_valid_ratio is satisfied for this frame
```

Do **not** remove `FeatureFrame` or `SequenceWindow` in this phase. They remain in use until Phase 4.

---

### 1.4 Create `src/maestro/pipeline/graph_buffering.py` _(new file)_

A dedicated sliding-window buffer for `GraphFrame` objects. Mirrors the logic of `SlidingWindowSequenceBuffer` but outputs `(T, N, C)` arrays instead of `SequenceWindow`.

**Class:** `GraphSequenceBuffer`

```python
class GraphSequenceBuffer:
    def __init__(self, *, window_length: int = 24, window_step: int = 3,
                 min_valid_ratio: float = 0.8) -> None: ...

    def push(self, frame: GraphFrame) -> None: ...
    def ready(self) -> bool: ...

    def window(self) -> tuple[np.ndarray, np.ndarray]:
        """Returns (sequences, valid_masks) where:
           sequences:   np.ndarray (T, N, C) float32
           valid_masks: np.ndarray (T, N)    bool
        """
        ...

    def current_valid_ratio(self) -> float: ...
```

The `valid` flag on each `GraphFrame` is set by checking whether at least one hand node block (nodes 0–20 or 21–41) is non-zero, consistent with `MIN_VALID_RATIO` from `graph_schema.py`.

**Acceptance criterion:** Unit test in `tests/unit/test_graph_buffering.py` that pushes 24 synthetic `GraphFrame` objects and asserts `window()` returns arrays of shape `(24, 67, 3)` and `(24, 67)`.

---

### 1.5 Refactor `src/maestro/training/dataset_builder.py` _(modify existing file)_

This is the most extensive change in Phase 1.

**Remove:**

- Import of `LandmarkFeatureTransformer` from `feature_engineering.py`.
- Import of `FeatureFrame`, `SequenceWindow` from `entities.py`.
- `self._transformer = LandmarkFeatureTransformer(...)` construction.
- The `LandmarkPacket` construction and `.transform()` call in `add_frame()`.

**Rename:**

- `LabeledLandmarkFrame` → `LabeledGraphFrame` with the following updated fields:

```python
@dataclass(slots=True)
class LabeledGraphFrame:
    keypoints: np.ndarray   # (67, 3) — wrist-normalised coordinates
    valid_mask: np.ndarray  # (67,)   — bool
    label: str
    recording_id: str
    timestamp_ms: int
```

**Update `SequenceDatasetBuilder`:**

- `add_frame()` must accept a `LabeledGraphFrame`, normalise its keypoints via `graph_schema.normalise_graph_frame()`, wrap as a `GraphFrame`, and push to a `GraphSequenceBuffer` instance.
- `build()` / `save_npz()` must output the following NPZ schema:

```
sequences:        float32 (N, T, 67, 3)   — normalised graph frames
labels:           object  (N,)            — string class names
label_ids:        int64   (N,)            — integer class indices
recording_ids:    object  (N,)            — IPN video stem
sample_ids:       object  (N,)            — unique sample identifier
valid_masks:      bool    (N, T, 67)      — per-node, per-frame validity
adjacency:        float32 (67, 67)        — degree-normalised adjacency from graph_schema
metadata_json:    object  scalar          — JSON string (see below)
```

**`metadata_json` required fields:**

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

- `load_npz()` must be updated to accept `sequences.ndim == 4` and to load `adjacency` as a `float32` array.
- Output filename: `data/processed/ipn_graph_v1.npz`.
- The old `(N, 24, 147)` NPZ schema is **fully retired** — do not write it.

**Acceptance criterion:** `pytest tests/unit/test_dataset_builder.py` passes. The builder must produce a correctly shaped NPZ with all 8 gesture classes present.

---

### 1.6 Refactor `scripts/build_ipn_dataset.py` _(modify existing file)_

**Remove imports:**

```python
from maestro.infrastructure.mediapipe.hands_extractor import MediaPipeHandsExtractor
from maestro.pipeline.hand_selection import HandSelector
from maestro.pipeline.feature_engineering import LandmarkFeatureTransformer
```

**Add imports:**

```python
from maestro.infrastructure.mediapipe.holistic_extractor import HolisticLandmarkExtractor
from maestro.pipeline.graph_schema import build_adjacency, normalise_adjacency
from maestro.training.dataset_builder import LabeledGraphFrame, SequenceDatasetBuilder
```

**Changes:**

- Instantiate `HolisticLandmarkExtractor(model_complexity=1)` for offline extraction.
- Replace the per-frame extraction loop: call `extractor.extract(frame_packet)` to get a `HolisticPacket`, then construct a `LabeledGraphFrame` from it.
- Remove `HandSelector` instantiation and its per-frame call.
- Add `data/IPN/pose_cache/<video_stem>_holistic.npz` per-video caching for raw `(F, 67, 3)` arrays (cache key includes `_holistic` suffix to avoid collision with existing hands-only cache files).
- Frames must be processed **sequentially within each video** to preserve MediaPipe tracker state.

**Cache invalidation:** Clear `data/IPN/cache/` (old hands-only format) before the first run. Document this in a comment at the top of the script.

**Acceptance criterion:** Script completes without error and produces `data/processed/ipn_graph_v1.npz` with shape `(N, 24, 67, 3)`.

---

### 1.7 Refactor `notebooks/01_dataset_preparation.ipynb` _(modify existing notebook)_

Restructure the notebook into 9 sections as described below. All original cells are either replaced or updated; none are preserved verbatim.

| Section | Title                                    | Action                                                                                                               |
| ------- | ---------------------------------------- | -------------------------------------------------------------------------------------------------------------------- |
| 0       | Prerequisites & Environment              | Replace cells 1–2: import holistic libs, check mediapipe version, verify IPN path                                    |
| 1       | Vocabulary & Graph Schema Configuration  | Replace config section: load YAML, call `build_adjacency()`, print 67-node index map                                 |
| 2       | IPN Annotation Ingestion                 | Largely unchanged: load `Annot_List.txt`, label mapping. Remove `LandmarkFeatureTransformer` reference               |
| 3       | Holistic Landmark Extraction             | Replace hands extraction: use `HolisticLandmarkExtractor`, define `extract_67_keypoints()`, cache to `_holistic.npz` |
| 4       | Graph Frame Construction & Normalisation | New section: apply `normalise_graph_frame()`, visualise skeleton on sample frame using matplotlib                    |
| 5       | Temporal Sequencing                      | Replace sliding-window: operate on `(F, 67, 3)` arrays, emit `(24, 67, 3)` windows, compute valid-frame ratio        |
| 6       | Dataset Export                           | Replace save cell: write `ipn_graph_v1.npz` with full schema including `adjacency` and `valid_masks`                 |
| 7       | Dataset Validation                       | New section: all 7 shape/class/range/adjacency checks must pass (see below)                                          |
| 8       | Dataset Analysis & Visualisation         | Adapt existing: replace feature heatmaps with node-position maps, keep class distribution chart                      |

**Section 7 — Required Validation Checks (all must pass before proceeding to Phase 2):**

1. `sequences.shape == (N, T, 67, 3)` — exact shape assertion.
2. `set(label_ids) == set(range(8))` — all 8 classes present.
3. Per-class count ≥ 50 sequences — print per-class counts; flag any below threshold.
4. `valid_masks.mean() >= 0.8` over the entire dataset.
5. `sequences` values in range `[-5, 5]` after normalisation (random 100-sample check).
6. Adjacency matrix: symmetric, zero diagonal, single connected component (`scipy.sparse.csgraph.connected_components`).
7. Recording-level split preview: group-aware train/val/test split shows no recording leakage.

---

## Phase 2 — STGCN-LSTM Model Implementation

### 2.1 Create `src/maestro/infrastructure/model/stgcn_lstm.py` _(new file)_

A full PyTorch reimplementation. The reference TensorFlow code (`fokhruli/STGCN-rehab`) is a conceptual reference only — none of its code is portable. The following design is derived from the reference's conceptual architecture, adapted for 67 nodes, PyTorch, and classification.

**Key design decisions vs reference:**

| Design point | Reference (TF)                  | This implementation (PyTorch)                        |
| ------------ | ------------------------------- | ---------------------------------------------------- |
| Framework    | TensorFlow 2.x / Keras          | PyTorch `nn.Module`                                  |
| Output       | `Dense(1, linear)` — regression | `Linear(hidden, 8)` + `CrossEntropyLoss`             |
| Node count   | 25 (COCO body)                  | 67 (hands + pose + face)                             |
| Adjacency    | External NumPy array            | Registered `torch.Buffer` (precomputed, not learned) |
| LSTM depth   | 4 layers (80→40→40→80)          | 2 layers (max) — CPU latency constraint              |
| Batch norm   | Absent                          | `BatchNorm1d` after each STGCN block                 |
| Attention    | `ConvLSTM2D` for coefficients   | Simplified `einsum`-based graph convolution          |

**Module structure:**

```python
class STGCNBlock(nn.Module):
    """
    One spatio-temporal graph convolution block.

    Operations (in order):
    1. Temporal convolution:  Conv1d across the T axis, kernel sizes [9, 15, 20]
                              (multi-scale; concatenated along channel dim)
    2. Skip-concat with input
    3. 1-hop graph conv:      Conv2d(1×1) → learned coef via LeakyReLU + softmax
                              → einsum 'btvw,btwc->btvc' against A_hat (1-hop)
    4. 2-hop graph conv:      same with A_hat @ A_hat (2-hop neighbourhood)
    5. Concat 1-hop + 2-hop outputs
    6. BatchNorm1d + Dropout
    7. Residual add (input projected to match channel count if needed)
    """

class STGCN_LSTM(nn.Module):
    def __init__(
        self,
        *,
        num_nodes: int,          # 67
        in_channels: int,        # 3
        adjacency: torch.Tensor, # (67, 67) degree-normalised
        stgcn_channels: list[int],  # e.g. [64, 128]
        lstm_hidden: int,        # 128
        lstm_layers: int,        # 2
        num_classes: int,        # 8
        dropout: float,          # 0.3
    ) -> None: ...

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (B, T, N, C) float32

        Returns:
            logits: (B, num_classes) float32
        """
        ...
```

**Forward pass walkthrough:**

1. `x`: `(B, T, N=67, C=3)` input.
2. **STGCN Block 1:** output `(B, T, N, stgcn_channels[0])` + residual.
3. **STGCN Block 2:** output `(B, T, N, stgcn_channels[1])` + residual.
4. **Reshape:** `(B, T, N × stgcn_channels[-1])` — flatten node and channel dims.
5. **LSTM (2 layers, hidden=128):** `(B, T, lstm_hidden)` full sequence; take final hidden state `(B, lstm_hidden)`.
6. **Classification head:** `Dropout → Linear(lstm_hidden, num_classes)`.
7. Return raw logits. `CrossEntropyLoss` is applied externally in the training loop.

**Adjacency registration:**

```python
# Register as a buffer so it is serialised with the model state dict
# but is NOT a learned parameter.
self.register_buffer("A_hat", adjacency.float())
```

**Parameter budget target:** ≤ 500K parameters for CPU real-time viability. Verify with `sum(p.numel() for p in model.parameters())` before proceeding.

**Acceptance criteria:**

1. `pytest tests/unit/test_stgcn_lstm.py` passes — tests must include:
   - Forward pass with `(2, 24, 67, 3)` input returns `(2, 8)` logits.
   - Gradient flows to all parameters (no `None` grads after backward).
   - `model.state_dict()` contains `A_hat` (buffer, not parameter).
   - Total parameter count ≤ 500K.

---

### 2.2 Create `configs/model/stgcn_lstm.yaml` _(new file)_

```yaml
model:
  model_name: stgcn_lstm_v1
  num_nodes: 67
  num_coords: 3
  stgcn_channels: [64, 128]
  num_stgcn_layers: 2
  lstm_hidden: 128
  lstm_layers: 2
  dropout: 0.3
  num_classes: 8

training:
  epochs: 60
  batch_size: 32
  learning_rate: 0.0005
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
  spatial_jitter_sigma: 0.003
  apply_prob: 0.5
  mirror_flip_prob: 0.5
```

**Note on hyperparameter choices vs `two_stream_lstm.yaml`:**

| Parameter                 | Two-Stream LSTM | STGCN-LSTM | Reason                                 |
| ------------------------- | --------------- | ---------- | -------------------------------------- |
| `epochs`                  | 30              | 60         | Graph models converge more slowly      |
| `batch_size`              | 64              | 32         | 4-D tensors are heavier in memory      |
| `learning_rate`           | 0.001           | 0.0005     | Lower initial LR stabilises graph conv |
| `early_stopping_patience` | 7               | 10         | Longer patience for slower convergence |
| `spatial_jitter_sigma`    | 0.005           | 0.003      | Raw coordinates are more sensitive     |

---

### 2.3 Add `STGCNModelConfig` to `src/maestro/utils/schema_validation.py` _(modify existing file)_

Add a Pydantic model alongside the existing `TwoStreamModelConfig` (or equivalent):

```python
class STGCNModelConfig(BaseModel):
    model_name: str
    num_nodes: int
    num_coords: int
    stgcn_channels: list[int]
    num_stgcn_layers: int
    lstm_hidden: int
    lstm_layers: int
    dropout: float
    num_classes: int
```

Register `STGCNModelConfig` in the model config discriminator so `load_runtime_config()` can parse `stgcn_lstm.yaml` without error.

---

## Phase 3 — Training Pipeline Integration

### 3.1 Refactor `src/maestro/training/datamodule.py` _(modify existing file)_

**Add `GraphSequenceTensorDataset`** (new class, separate from `SequenceTensorDataset`):

```python
class GraphSequenceTensorDataset(Dataset):
    """Dataset for 4-D graph tensors: (N, T, N_nodes, C)."""

    def __init__(self, sequences: np.ndarray, labels: np.ndarray) -> None:
        assert sequences.ndim == 4, (
            f"Expected 4-D sequences (N, T, N_nodes, C), got shape {sequences.shape}"
        )
        self.X = torch.from_numpy(sequences).float()
        self.y = torch.from_numpy(labels).long()

    def __len__(self) -> int: ...
    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]: ...
```

**Add `AugmentedGraphSequenceTensorDataset`** using the new graph augmenter (see 3.2).

**Update `GestureDataModule`:**

- `setup()`: detect schema from NPZ metadata (`graph_schema_version` key). If present, use `GraphSequenceTensorDataset`; otherwise fall back to `SequenceTensorDataset` for backward compatibility with the two-stream baseline.
- Load `adjacency` from `npz["adjacency"]` and expose as `self.adjacency: torch.Tensor`.
- Add properties: `num_nodes: int`, `num_coords: int`, `adjacency_schema: str`.
- `feature_dim` property: for graph datasets, compute `num_nodes * num_coords` (= 201) for any code that reads it as a flat dimension. Model configuration must use `num_nodes`/`num_coords` directly.

**Validation guard change:**

```python
# Old (remove):
if sequences.ndim != 3:
    raise ValueError(...)

# New:
if sequences.ndim not in (3, 4):
    raise ValueError(
        f"sequences must be 3-D (N, T, F) or 4-D (N, T, N_nodes, C), "
        f"got shape {sequences.shape}"
    )
```

**Acceptance criterion:** `pytest tests/unit/test_dataset_builder.py tests/integration/test_phase3_feature_pipeline.py -q` passes.

---

### 3.2 Refactor `src/maestro/training/augmentation.py` _(modify existing file)_

**Remove entirely:**

- `_X_COORD_DIMS`, `_POS_START`, `_POS_END` — all feature-schema-v5-specific index constants.
- Any augmentation logic that uses hardcoded slice indices from the 147-dim feature vector.

**Rewrite `GestureAugmenter.augment()` to accept `(T, N, C)` arrays:**

| Augmentation         | Old (T, F) logic                  | New (T, N, C) logic                                                                            |
| -------------------- | --------------------------------- | ---------------------------------------------------------------------------------------------- |
| Temporal scaling     | Resample on axis 0                | Unchanged — works on any shape with T on axis 0                                                |
| Spatial jitter       | `seq[:, _X_COORD_DIMS] += noise`  | `seq += rng.normal(0, sigma, size=seq.shape)`                                                  |
| Mirror flip (x-axis) | Negate specific dim indices       | Negate `seq[:, :, 0]` (all x-coords); swap node blocks: `seq[:, 0:21, :]` ↔ `seq[:, 21:42, :]` |
| Label-aware flip     | `swipe_left` ↔ `swipe_right` swap | Unchanged logic                                                                                |

**`AugmentationConfig` dataclass:** No structural changes required. `spatial_jitter_sigma` and `mirror_flip_prob` retain their meaning with the updated axis semantics.

**Acceptance criterion:** `pytest tests/unit/test_augmentation.py -q` passes with the new graph-based augmenter.

---

### 3.3 Refactor `src/maestro/training/train_lstm.py` _(modify existing file)_

**`ModelConfig` dataclass — add fields:**

```python
@dataclass
class ModelConfig:
    # Existing fields retained:
    model_name: str
    num_classes: int
    dropout: float
    # New fields for STGCN:
    num_nodes: int = 67
    num_coords: int = 3
    stgcn_channels: list[int] = field(default_factory=lambda: [64, 128])
    num_stgcn_layers: int = 2
    lstm_hidden: int = 128
    lstm_layers: int = 2
    # Two-stream fields retained for backward compat:
    input_size: int = 147
    hidden_size: int = 64
    num_layers: int = 2
```

**`_build_model()` — replace hardcoded two-stream construction with model registry dispatch:**

```python
def _build_model(config: ModelConfig, datamodule: GestureDataModule) -> nn.Module:
    if config.model_name.startswith("stgcn"):
        from maestro.infrastructure.model.stgcn_lstm import STGCN_LSTM
        return STGCN_LSTM(
            num_nodes=config.num_nodes,
            in_channels=config.num_coords,
            adjacency=datamodule.adjacency,
            stgcn_channels=config.stgcn_channels,
            lstm_hidden=config.lstm_hidden,
            lstm_layers=config.lstm_layers,
            num_classes=config.num_classes,
            dropout=config.dropout,
        )
    elif config.model_name.startswith("two_stream"):
        from maestro.infrastructure.model.two_stream_lstm import TwoStreamGestureLSTM
        return TwoStreamGestureLSTM(
            input_size=config.input_size,
            hidden_size=config.hidden_size,
            num_layers=config.num_layers,
            dropout=config.dropout,
            num_classes=config.num_classes,
        )
    else:
        raise ValueError(f"Unknown model_name: {config.model_name!r}")
```

**Input validation — replace flat-dim check with graph-dim check:**

```python
# Old (remove):
if datamodule.feature_dim != config.model.input_size:
    raise ValueError(...)

# New:
if config.model_name.startswith("stgcn"):
    if datamodule.num_nodes != config.model.num_nodes:
        raise ValueError(
            f"Dataset num_nodes={datamodule.num_nodes} does not match "
            f"model config num_nodes={config.model.num_nodes}"
        )
    if datamodule.num_coords != config.model.num_coords:
        raise ValueError(...)
```

**`_train_one_epoch()` and `_validate_one_epoch()`:** No structural changes required. `model(batch_x)` receives `(B, T, 67, 3)` and returns `(B, 8)` logits — the call site is shape-agnostic.

**Acceptance criterion:** A smoke-test training run of 2 epochs completes without error: `pytest tests/integration/test_training_pipeline_smoke.py -q`.

---

### 3.4 Refactor `notebooks/02_model_training.ipynb` _(modify existing notebook)_

Apply the following targeted changes (cell line references are approximate; verify against current notebook):

| Cell                                                     | Change                                                                                                                                                               |
| -------------------------------------------------------- | -------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| Imports (cells 1–2)                                      | Add `from maestro.infrastructure.model.stgcn_lstm import STGCN_LSTM`. Remove `gesture_control` alias.                                                                |
| Configuration (cell 2)                                   | Set `MODEL_TYPE = "stgcn"`. Update `_CONFIG_MAP` to include `"stgcn": "configs/model/stgcn_lstm.yaml"`. Prefer `ipn_graph_v1.npz`. Add `N_NODES = 67; N_COORDS = 3`. |
| Load training config (cell 3)                            | After loading config, load `adjacency = np.load(DATASET_PATH)["adjacency"]` and convert to `torch.FloatTensor`. Print `adjacency.shape`.                             |
| Load dataset (cell 4)                                    | Replace `SequenceTensorDataset` with `GraphSequenceTensorDataset`. Assert `sequences.shape[2] == 67` and `sequences.shape[3] == 3`.                                  |
| **New cell — Graph Inspection** _(insert before cell 4)_ | Visualise a random sample's skeleton: plot 67-node positions + adjacency edges on a blank frame using `matplotlib`. Confirm extraction correctness visually.         |
| Data splits visualisation (cell 5)                       | Replace `F = sequences.shape[2]` with `_, T, N_nodes, C = sequences.shape`. Replace feature correlation heatmap with node-position mean map.                         |
| Build model (cell 6)                                     | Replace entirely with `STGCN_LSTM(...)` constructor (see Section 2.1). Print parameter count. Assert ≤ 500K.                                                         |
| Training loop (cell 7)                                   | No change.                                                                                                                                                           |
| Learning curves (cell 8)                                 | No change.                                                                                                                                                           |
| Validation evaluation (cell 9)                           | No change.                                                                                                                                                           |
| Confusion matrix (cell 10)                               | No change.                                                                                                                                                           |
| Per-class metrics (cell 11)                              | No change. Flag any class with recall < 70%.                                                                                                                         |
| Export (cell 12)                                         | Pass `adjacency` into export payload. Set `artifact_type = "stgcn_lstm_inference"`. Filename: `<version>_stgcn_inference.pt`.                                        |
| Hugging Face Publish (cell 13)                           | Set repo target to `ntsrigaud/maestro-stgcn`. Update model card: STGCN-LSTM, graph-schema-v1, 67 nodes.                                                              |

---

## Phase 4 — Export, Loading & Inference Runtime

### 4.1 Refactor `src/maestro/training/export_checkpoint.py` _(modify existing file)_

**Change `artifact_type`:**

```python
# Old:
artifact_type = "two_stream_lstm_inference"

# New:
artifact_type = "stgcn_lstm_inference"
```

**Add `graph_schema` to payload:**

```python
payload = {
    ...existing keys...,
    "graph_schema": {
        "graph_schema_version": graph_schema.GRAPH_SCHEMA_VERSION,
        "num_nodes": graph_schema.NUM_NODES,
        "num_coords": graph_schema.NUM_COORDS,
        "adjacency": adjacency.tolist(),  # serialise as Python list for portability
        "node_index_map": {k: list(v) for k, v in graph_schema.NODE_INDEX_MAP.items()},
    },
}
```

**Remove or retain `feature_schema`:** Retain for backward-compat readers but update `feature_dim` to `num_nodes * num_coords = 201`. New readers must use `graph_schema`.

---

### 4.2 Refactor `src/maestro/infrastructure/model/checkpoint_loader.py` _(modify existing file)_

**`_build_model_from_artifact()` — add STGCN dispatch:**

```python
def _build_model_from_artifact(
    model_config: dict[str, Any],
    artifact_type: str,
    payload: dict[str, Any],
    device: torch.device,
) -> torch.nn.Module:
    if artifact_type == "stgcn_lstm_inference":
        from maestro.infrastructure.model.stgcn_lstm import STGCN_LSTM

        graph_schema_payload = payload.get("graph_schema", {})
        adjacency = torch.tensor(
            graph_schema_payload["adjacency"], dtype=torch.float32
        )
        return STGCN_LSTM(
            num_nodes=int(model_config["num_nodes"]),
            in_channels=int(model_config["num_coords"]),
            adjacency=adjacency,
            stgcn_channels=list(model_config["stgcn_channels"]),
            lstm_hidden=int(model_config["lstm_hidden"]),
            lstm_layers=int(model_config["lstm_layers"]),
            num_classes=int(model_config["num_classes"]),
            dropout=float(model_config["dropout"]),
        ).to(device)

    elif artifact_type.startswith("two_stream"):
        # Existing TwoStreamGestureLSTM path — preserved for old artifacts.
        from maestro.infrastructure.model.two_stream_lstm import TwoStreamGestureLSTM
        return TwoStreamGestureLSTM(...).to(device)

    else:
        raise ValueError(
            f"Unsupported artifact type {artifact_type!r}. "
            "Supported: 'stgcn_lstm_inference', 'two_stream_lstm_inference'."
        )
```

**`InferenceArtifact` dataclass — add graph fields:**

```python
@dataclass(slots=True)
class InferenceArtifact:
    model: torch.nn.Module
    model_version: str
    class_labels: list[str]
    # Graph schema fields (None for legacy two-stream artifacts):
    graph_schema_version: str | None
    num_nodes: int | None
    num_coords: int | None
    adjacency: np.ndarray | None
    # Legacy fields (retained for two-stream backward compat):
    feature_schema_version: str
    feature_dim: int
    window_length: int
    window_step: int
    orientation_normalization: bool
    artifact_path: Path
```

**`validate_runtime_compatibility()`:** Add a check that `artifact.graph_schema_version == graph_schema.GRAPH_SCHEMA_VERSION`. Fail loudly (raise, not warn) on mismatch.

**Acceptance criterion:** `pytest tests/unit/test_checkpoint_loader.py -q` passes, covering both `stgcn_lstm_inference` and legacy `two_stream_lstm_inference` artifact types.

---

### 4.3 Update `src/maestro/utils/feature_validation.py` _(modify existing file)_

Add a schema compatibility check for graph artifacts:

```python
def validate_graph_schema_compatibility(
    artifact: InferenceArtifact,
    dataset_metadata: dict,
) -> None:
    """Raise ValueError if artifact schema does not match dataset schema."""
    artifact_version = artifact.graph_schema_version
    dataset_version = dataset_metadata.get("graph_schema_version")
    if artifact_version != dataset_version:
        raise ValueError(
            f"Graph schema mismatch: artifact={artifact_version!r}, "
            f"dataset={dataset_version!r}. Rebuild the dataset or re-export the artifact."
        )
```

---

### 4.4 Replace the inference runtime pipeline in `src/maestro/app/bootstrap.py` _(modify existing file)_

**Remove imports:**

```python
# Remove these:
from maestro.infrastructure.mediapipe.hands_extractor import MediaPipeHandsExtractor
from maestro.pipeline.feature_engineering import LandmarkFeatureTransformer
from maestro.pipeline.hand_selection import HandSelector
from maestro.pipeline.buffering import SlidingWindowSequenceBuffer
```

**Add imports:**

```python
from maestro.infrastructure.mediapipe.holistic_extractor import HolisticLandmarkExtractor
from maestro.pipeline.graph_buffering import GraphSequenceBuffer
```

**`AppContainer` dataclass — update fields:**

```python
@dataclass(slots=True)
class AppContainer:
    config: RuntimeConfig
    frame_source: IFrameSource
    gaze_detector: IGazeIntentDetector
    landmark_extractor: HolisticLandmarkExtractor   # was: IHandLandmarkExtractor
    sequence_buffer: GraphSequenceBuffer             # was: ISequenceBuffer
    model_artifact: InferenceArtifact
    stabilizer: PredictionStabilizer
    activation_controller: GestureActivationController
    dispatcher: ICommandDispatcher
    overlay_renderer: IFeedbackRenderer
    benchmark: LatencyBenchmark
    show_window: bool
    # Removed: hand_selector, feature_transformer
```

**`build_container()` — update construction:**

```python
# Old:
landmark_extractor = MediaPipeHandsExtractor(...)
hand_selector = HandSelector(...)
feature_transformer = LandmarkFeatureTransformer(...)
sequence_buffer = SlidingWindowSequenceBuffer(...)

# New:
landmark_extractor = HolisticLandmarkExtractor(
    model_complexity=0,  # lightweight for real-time
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5,
)
sequence_buffer = GraphSequenceBuffer(
    window_length=artifact.window_length,
    window_step=artifact.window_step,
    min_valid_ratio=0.8,
)
```

---

### 4.5 Update inference hot loop in `src/maestro/app/runtime.py` _(modify existing file)_

The runtime loop currently calls:

```python
multi_hand = extractor.extract(frame)
landmark = hand_selector.select(multi_hand)
feature_frame = transformer.transform(landmark)
buffer.push(feature_frame)
```

Replace with:

```python
holistic_packet = extractor.extract(frame)           # returns HolisticPacket
graph_frame = GraphFrame(
    nodes=holistic_packet.keypoints,
    valid_mask=holistic_packet.valid_mask,
    timestamp_ms=holistic_packet.timestamp_ms,
    valid=holistic_packet.valid_mask[:42].any(),     # at least one hand node detected
)
buffer.push(graph_frame)
```

When `buffer.ready()`, call `buffer.window()` to get `(sequences, valid_masks)` tuple. Construct the model input tensor:

```python
sequences_tensor, _ = buffer.window()
x = torch.from_numpy(sequences_tensor).unsqueeze(0).float()   # (1, T, 67, 3)
with torch.no_grad():
    logits = model(x)
probs = torch.softmax(logits, dim=-1).squeeze(0)
```

**Remove:** All references to `hand_selector`, `feature_transformer`, `IFeatureTransformer`, and `SequenceWindow` from the hot loop.

**Update `RuntimeDependencies` Protocol** to remove `hand_selector` and `feature_transformer` fields.

---

### 4.6 Retire pipeline components _(modify existing files, do not delete yet)_

Add a deprecation notice at the top of each retired module. Do not delete — these will be removed in a cleanup PR after Phase 5 passes:

- `src/maestro/pipeline/hand_selection.py` — add `# DEPRECATED: retired in STGCN migration.`
- `src/maestro/pipeline/feature_engineering.py` — add `# DEPRECATED: retired in STGCN migration.`
- `src/maestro/interfaces/feature_transformer.py` — add `# DEPRECATED: retired in STGCN migration.`

**Acceptance criteria:**

1. `python scripts/run_realtime.py` launches without import errors.
2. `pytest tests/integration/test_gesture_dispatch_pipeline.py -q` passes.
3. No reference to `HandSelector`, `LandmarkFeatureTransformer`, or `FeatureFrame` exists in the runtime hot path (verify by `grep -r "HandSelector\|LandmarkFeatureTransformer" src/maestro/app/`).

---

## Phase 5 — Validation, Benchmarking & Regression

### 5.1 Accuracy Evaluation

Run the full evaluation suite on the trained `stgcn_lstm_v1` artifact:

```bash
pytest tests/integration/ -q
```

**Hard pass criteria (block release if any fail):**

| Criterion             | Gate value          |
| --------------------- | ------------------- |
| Overall test accuracy | ≥ 84%               |
| Macro-F1              | ≥ 84%               |
| Worst-class recall    | All 8 classes ≥ 72% |
| Per-class precision   | All 8 classes ≥ 70% |

If `zoom_in` / `zoom_out` recall is below 72%, inspect whether the two-hop graph convolution is active and adjacency is correctly loading in the artifact.

---

### 5.2 Training–Inference Parity Integration Test

Create `tests/integration/test_stgcn_parity.py`:

**Test:** Take a known IPN video frame. Extract keypoints via `HolisticLandmarkExtractor`. Normalise via `graph_schema.normalise_graph_frame()`.  
Separately, load the same frame in the dataset builder path and normalise via `SequenceDatasetBuilder.add_frame()`.  
Assert that the resulting `(67, 3)` arrays are **bitwise identical** (or within floating-point epsilon `1e-6`).

This test must pass before the artifact is published. Any discrepancy indicates a training–inference parity break.

---

### 5.3 CPU Latency Benchmark

```bash
python scripts/benchmark_system.py --model stgcn_lstm_v1 --frames 500
```

**Hard pass criteria:**

| Metric                        | Gate value                                |
| ----------------------------- | ----------------------------------------- |
| Median end-to-end latency     | < 35ms                                    |
| P95 end-to-end latency        | < 50ms                                    |
| MediaPipe Holistic extraction | < 90ms/frame at 720p (model_complexity=0) |
| STGCN-LSTM inference          | < 15ms/window                             |

If P95 > 50ms:

1. Reduce input video resolution to 480p before extraction.
2. Reduce STGCN to 1 layer (`stgcn_channels: [64]`).
3. Re-run benchmark. If still > 50ms, escalate for hardware assessment.

---

### 5.4 Artifact Publication

**Only after all Phase 5 criteria pass:**

```bash
# In notebook 02_model_training.ipynb, Cell 13
# Target: ntsrigaud/maestro-stgcn (NOT the existing LSTM repo)
```

Update `configs/model_source.yaml` to point to the new HuggingFace repo and artifact filename.

Update `src/maestro/training/hf_publish.py` to include graph schema metadata in the model card.

---

## Appendix A — File Change Summary

### New Files to Create

| File                                                         | Phase | Purpose                                                    |
| ------------------------------------------------------------ | ----- | ---------------------------------------------------------- |
| `src/maestro/pipeline/graph_schema.py`                       | 1     | Single source of truth: node map, adjacency, normalisation |
| `src/maestro/infrastructure/mediapipe/holistic_extractor.py` | 1     | MediaPipe Holistic wrapper, returns `HolisticPacket`       |
| `src/maestro/pipeline/graph_buffering.py`                    | 1     | `GraphSequenceBuffer` for `(T, N, C)` windows              |
| `src/maestro/infrastructure/model/stgcn_lstm.py`             | 2     | `STGCN_LSTM` PyTorch module                                |
| `configs/model/stgcn_lstm.yaml`                              | 2     | STGCN architecture & training hyperparameters              |
| `tests/unit/test_holistic_extractor.py`                      | 1     | Unit tests for `HolisticLandmarkExtractor`                 |
| `tests/unit/test_graph_buffering.py`                         | 1     | Unit tests for `GraphSequenceBuffer`                       |
| `tests/unit/test_stgcn_lstm.py`                              | 2     | Unit tests for `STGCN_LSTM`                                |
| `tests/integration/test_stgcn_parity.py`                     | 5     | Training–inference normalisation parity test               |

### Modified Files

| File                                                    | Phase | Change                                                                                       |
| ------------------------------------------------------- | ----- | -------------------------------------------------------------------------------------------- |
| `src/maestro/domain/entities.py`                        | 1     | Add `GraphFrame`, `HolisticPacket`, `LabeledGraphFrame`; add fields to `InferenceArtifact`   |
| `src/maestro/training/dataset_builder.py`               | 1     | Replace transformer-based logic with graph frame construction; update NPZ schema             |
| `scripts/build_ipn_dataset.py`                          | 1     | Replace `MediaPipeHandsExtractor` + `HandSelector` with `HolisticLandmarkExtractor`          |
| `notebooks/01_dataset_preparation.ipynb`                | 1     | Full restructure into 9 sections per plan                                                    |
| `src/maestro/utils/schema_validation.py`                | 2     | Add `STGCNModelConfig` Pydantic schema                                                       |
| `src/maestro/training/datamodule.py`                    | 3     | Add `GraphSequenceTensorDataset`; update shape guards; expose `adjacency`                    |
| `src/maestro/training/augmentation.py`                  | 3     | Rewrite for node-axis `(T, N, C)` augmentation                                               |
| `src/maestro/training/train_lstm.py`                    | 3     | Add `ModelConfig` graph fields; model registry dispatch in `_build_model()`                  |
| `notebooks/02_model_training.ipynb`                     | 3     | Targeted cell updates per Section 3.4                                                        |
| `src/maestro/training/export_checkpoint.py`             | 4     | New `artifact_type`; add `graph_schema` to payload                                           |
| `src/maestro/infrastructure/model/checkpoint_loader.py` | 4     | STGCN dispatch in `_build_model_from_artifact()`; update `InferenceArtifact` dataclass       |
| `src/maestro/utils/feature_validation.py`               | 4     | Add `validate_graph_schema_compatibility()`                                                  |
| `src/maestro/app/bootstrap.py`                          | 4     | Replace `HandSelector` + `LandmarkFeatureTransformer` with holistic extractor + graph buffer |
| `src/maestro/app/runtime.py`                            | 4     | Update hot loop to use `HolisticPacket` → `GraphFrame` → `GraphSequenceBuffer`               |

### Deprecated (not yet deleted)

| File                                            | Note                                          |
| ----------------------------------------------- | --------------------------------------------- |
| `src/maestro/pipeline/hand_selection.py`        | Add deprecation comment; delete after Phase 5 |
| `src/maestro/pipeline/feature_engineering.py`   | Add deprecation comment; delete after Phase 5 |
| `src/maestro/interfaces/feature_transformer.py` | Add deprecation comment; delete after Phase 5 |

---

## Appendix B — Do-Not Rules

These constraints are derived from the audit's `§10.2 Do Not Do` and must be enforced throughout all phases:

1. **Do not reuse any TensorFlow code** from `fokhruli/STGCN-rehab`. It is a conceptual reference only.
2. **Do not store engineered features** alongside raw coordinates in `ipn_graph_v1.npz`. The new schema contains only raw `(67, 3)` coordinates.
3. **Do not concatenate or mix** the old `(N, T, 147)` dataset with the new `(N, T, 67, 3)` dataset. They are semantically incompatible.
4. **Do not use the full 468-point MediaPipe Face mesh**. Use only the 14 pinned face contour keypoints defined in `graph_schema.py`.
5. **Do not process video frames out of sequential order** within a single video clip. MediaPipe's tracker depends on temporal continuity.
6. **Do not hardcode `num_nodes=67`** anywhere except `graph_schema.py`. All other modules must read this constant from the schema or config.
7. **Do not merge `stgcn_lstm_inference` artifacts to the existing `ntsrigaud/maestro-lstm` HF repo**. Use `ntsrigaud/maestro-stgcn`.

---

_See also: [stgcn-lstm-migration-audit.md](stgcn-lstm-migration-audit.md) (source audit), [stgcn-lstm-architecture.md](../architecture/stgcn-lstm-architecture.md) (architecture overview), [data-flow.md](../architecture/data-flow.md) (inference pipeline diagram)._
