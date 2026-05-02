# Work Log — Dual-Branch STGCN-LSTM Architecture and Training Pipeline

**Date:** 2026-05-02
**Branch:** `chore/dual-stream-stgcn-lstm`
**Author:** ntsrigaud

---

## Context

The two-stream LSTM operated on a flat 147-dimensional hand-landmark feature vector extracted from a 21-node single-hand graph. The architecture showed a hard ceiling at Macro-F1 ~0.96 on the IPN Hand training set but lacked the graph-structured inductive bias needed to generalise across varied body orientations and dual-hand gestures. This branch implements the full vertical slice for the dual-branch STGCN-LSTM described in the IEEE 2025 paper "A Study on the STGCN-LSTM Sign Language Recognition Model Based on Phonological Features of Sign Language", adapted to the 67-node IPN Hand graph schema. The two-stream LSTM pipeline is preserved and continues to operate when its checkpoint is loaded; model selection is determined automatically by the artifact type field.

---

## Scope of Work

### Graph schema definition
A centralised `graph_schema` module defines all 67-node layout constants, BFS-derived partitioned adjacency construction, wrist-anchor normalisation, and sliding window parameters. No other module may inline node indices or adjacency logic.

### MediaPipe Holistic extraction
A `HolisticLandmarkExtractor` replaces the per-hand `HandsExtractor` for the STGCN path. It wraps the mediapipe 0.10+ task-based `HolisticLandmarker`, maps hands, pose, and face landmarks to the 67-node layout, and applies wrist-anchor normalisation before returning a `HolisticPacket`.

### 4-D sliding window buffer
`GraphSequenceBuffer` mirrors `SlidingWindowSequenceBuffer` for `(T, N, C)` graph frames, enforcing the same valid-frame ratio gate before emitting a window.

### Graph-space augmentation and data modules
`GraphGestureAugmenter` operates on `(T, N, C)` arrays with temporal resampling, isotropic Gaussian coordinate jitter, and left-right mirror flip (node swap + x-reflection). `GraphGestureDataModule` provides group-aware splitting, augmented training datasets, and `DataLoader` construction for 4-D graph batches.

### Dual-branch classifier architecture
`DualBranchSTGCN_LSTM` implements the paper's two-branch design:
- Branch 1 extracts per-frame phonological features (joint angles, palm normals, hand position, motion) deterministically, then passes the 56-dim sequence through `Conv1d + LSTM + self-attention + Bahdanau attention`.
- Branch 2 applies three `STGCNBlock` layers on the K=3 fully-connected partitioned adjacency, then mean-pools over nodes and time.
- Branches are concatenated and projected to class logits.

### Training pipeline integration
`train_lstm.py` dispatches to `DualBranchSTGCN_LSTM` and `GraphGestureDataModule` based on `model_name`. `ReduceLROnPlateau` monitors `val_accuracy`; early stopping also tracks accuracy. Label smoothing and class weighting are disabled to align with the paper loss.

### Checkpoint export and load
The `dual_stgcn_lstm_inference` artifact type is introduced. Export includes the `graph_schema` dict (adjacency, node/coord counts, schema version). The checkpoint loader dispatches on artifact type, not model name, making the dispatch robust to future renames.

### Runtime integration
`AppContainer` carries optional `holistic_extractor` and `graph_buffer` slots. `build_container` populates them when the loaded artifact reports a non-`"unknown"` graph schema version. The runtime hot loop branches on `graph_buffer is not None` to choose between the STGCN and two-stream paths.

### Iterative bug fixes
Multiple training runs revealed four implementation errors corrected across this branch: incorrect `STGCNBlock` execution order (residual applied before final ReLU), incorrect phonological curvature formula (cross-product vs. signed-angle), early stopping and LR scheduler monitoring val_loss instead of val_accuracy, and label smoothing inadvertently applied to the STGCN path.

### Cleanup
The single-branch STGCN-LSTM architecture, its configs, tests, and training pipeline logic were removed after the dual-branch implementation was validated.

---

## Technical Changes

### `src/maestro/domain/entities.py`
- `HolisticPacket`: `nodes: np.ndarray (67, 3)`, `valid_mask: np.ndarray (67,)`, `timestamp_ms: int`.
- `GraphFrame`: `nodes: np.ndarray (N, C)`, `valid: bool`.
- `LabeledGraphFrame`: extends `GraphFrame` with `label: str`.

### `src/maestro/pipeline/graph_schema.py` (new)
- Constants: `NUM_NODES = 67`, `NUM_COORDS = 3`, `GRAPH_SCHEMA_VERSION = "graph-schema-v1"`, `WINDOW_LENGTH = 24`, `WINDOW_STEP = 3`, `MIN_VALID_RATIO = 0.8`.
- `NODE_INDEX_MAP`: named ranges for right hand (0–20), left hand (21–41), pose subset (42–52), face subset (53–66).
- `POSE_LANDMARK_INDICES`, `FACE_LANDMARK_INDICES`: landmark index mappings into MediaPipe output.
- `build_adjacency()`: 67×67 binary symmetric matrix from hand, pose, face, cross-wrist, and face-pose edges.
- `build_partitioned_adjacency()`: `(3, 67, 67)` K=3 partition tensor (self / centripetal / centrifugal), row-normalised.
- `normalise_graph_frame()`: wrist-anchor normalisation using right or pose wrist as origin.

### `src/maestro/infrastructure/mediapipe/holistic_extractor.py` (new)
- `HolisticLandmarkExtractor`: wraps mediapipe 0.10 task-based `HolisticLandmarker`; auto-downloads model to `~/.cache/maestro/`.
- Maps hand (21 nodes), pose subset (11 nodes), face subset (14 nodes) to 67-node layout; applies `normalise_graph_frame()`.
- `model_complexity` flag: 0 for real-time, 1 for accurate.

### `src/maestro/pipeline/graph_buffering.py` (new)
- `GraphSequenceBuffer`: ring deque of `GraphFrame`; emits `(T, N, C)` numpy arrays via `peek()` when `ready()`.
- Enforces `MIN_VALID_RATIO` gate before emitting.

### `src/maestro/infrastructure/model/stgcn_lstm.py`
- `_GraphConv`: K-partition spatial graph convolution; per-partition Xavier-initialised weight matrices; `einsum`-based forward pass.
- `STGCNBlock`: canonical ST-GCN block (SCN→BN→ReLU→TCN→BN→Dropout→+Residual→ReLU); optional 1×1 projection. **Fixed (`33c0cb7`)**: corrected block execution order and residual connection to match paper schematic.
- `_BahdanauAttention`: additive temporal attention over LSTM output.

### `src/maestro/infrastructure/model/dual_branch_stgcn_lstm.py` (new)
- `_PhonologicalFeatureExtractor`: stateless geometric feature extraction — 15 joint angles + 4 inter-finger angles + 3 palm normal + 3 position + 3 motion per hand × 2 hands = 56 dimensions. Uses `_safe_angle` for numerically stable arccos. **Fixed (`c203074`, `0829985`)**: corrected curvature formula, velocity/acceleration clamping, and SA-ConvLSTM activation flow.
- `DualBranchSTGCN_LSTM`: Branch 1 = phonological SA-ConvLSTM (`Conv1d(56→256)+BN+ReLU → LSTM → self-attention → Bahdanau`) → `(B, 256)`; Branch 2 = 3× `STGCNBlock(3→64→128→256, K=3)` → mean-pool → `(B, 256)`; Fusion = `Linear(512→8)`.
- Node index constants (`_R_WRIST`, `_R_INDEX`, `_L_OFFSET`, etc.) defined at module level for clarity.
- `PHONO_FEAT_DIM = 56` exported for schema documentation.

### `configs/model/dual_branch_stgcn_lstm.yaml` (new)
- `model_name: dual_stgcn_lstm_v1`, `num_nodes: 67`, `num_coords: 3`.
- `stgcn_channels: [64, 128, 256]`, `phono_conv_channels: 256`, `lstm_hidden: 256`, `lstm_layers: 2`, `dropout: 0.3`, `num_classes: 8`.
- `training.learning_rate: 1e-4`, `epochs: 80`, `batch_size: 32`, `early_stopping_patience: 20`. `class_weighting: false`.
- `scheduler: {enabled: true, factor: 0.5, patience: 5, min_lr: 1e-7}`.

### `src/maestro/training/augmentation.py`
- `GraphAugmentationConfig`: `temporal_scale_range`, `spatial_jitter_sigma`, `apply_prob`, `mirror_flip_prob`.
- `GraphGestureAugmenter`: temporal resampling (`np.interp`), Gaussian coordinate jitter `N(0, σ)`, mirror flip (left-right node swap + x-axis negation).

### `src/maestro/training/datamodule.py`
- `GraphSequenceTensorDataset`: wraps `(M, T, N, C)` NPZ arrays.
- `AugmentedGraphSequenceTensorDataset`: applies `GraphGestureAugmenter` on `__getitem__`.
- `GraphDatasetBundle`: `sequences (M, T, N, C)`, `adjacency (N, N)`, `valid_masks (M, T, N)`, `metadata`.
- `GraphGestureDataModule`: group-aware split via `GroupShuffleSplit`; falls back to random split when < 3 recordings; builds augmented train dataset and plain val/test datasets.

### `src/maestro/training/train_lstm.py`
- `_build_model`: dispatches `dual_stgcn_*` → `DualBranchSTGCN_LSTM`, `two_stream_*` → `TwoStreamGestureLSTM`.
- `create_datamodule`: dispatches `dual_stgcn_*` → `GraphGestureDataModule`, otherwise → `GestureDataModule`.
- `_build_criterion`: `CrossEntropyLoss` without label smoothing for all paths. **Fixed (`6bfb6da`)**.
- Training loop: `ReduceLROnPlateau` and early stopping both monitor `val_accuracy`. **Fixed (`5571a5f`)**.
- `SchedulerConfig` added to `TrainingPipelineConfig`; parsed from YAML `scheduler` section.

### `src/maestro/training/export_checkpoint.py`
- `export_inference_artifact`: `is_dual_stgcn` path exports `artifact_type = "dual_stgcn_lstm_inference"` and `graph_schema` dict (adjacency, `num_nodes`, `num_coords`, `graph_schema_version`).
- Two-stream path unchanged.

### `src/maestro/infrastructure/model/checkpoint_loader.py`
- `_build_model_from_artifact`: dispatches on `artifact_type.startswith("dual_stgcn")` → `DualBranchSTGCN_LSTM`; `two_stream` → `TwoStreamGestureLSTM`. Raises `ValueError` for unknown types.
- Removed deprecated single-branch STGCN-LSTM dispatch path. **Refactor (`3711498`)**.
- `InferenceArtifact` dataclass gains `graph_schema_version: str` field. **Fixed (`b10bde8`)**.

### `src/maestro/app/bootstrap.py`
- `AppContainer` gains `holistic_extractor: HolisticLandmarkExtractor | None = None` and `graph_buffer: GraphSequenceBuffer | None = None`.
- `build_container`: `is_stgcn = artifact.graph_schema_version != "unknown"`; STGCN path instantiates `HolisticLandmarkExtractor()` and `GraphSequenceBuffer`; two-stream path leaves slots as `None`.

### `src/maestro/app/runtime.py`
- Hot loop: `is_stgcn = getattr(self._container, "graph_buffer", None) is not None`.
- STGCN branch: `holistic_extractor.extract(frame)` → `graph_buffer.push(graph_frame)` → `graph_buffer.peek()` → `model(window_tensor)`.
- Two-stream branch: unchanged hand-selection + feature-engineering path.

### `notebooks/01_dataset_preparation.ipynb`
- Updated for offline holistic extraction using `mp.solutions`; outputs `(M, T, 67, 3)` NPZ for `GraphGestureDataModule`.

### `notebooks/02_model_training.ipynb`
- Fixed to `DualBranchSTGCN_LSTM`; updated model card generation; documented geometric SA-ConvLSTM features; added `isinstance(model, DualBranchSTGCN_LSTM)` assertion.

### `scripts/benchmark_system.py`
- Updated to use `DualBranchSTGCN_LSTM` forward pass for latency benchmarking.

### Removed
- `src/maestro/infrastructure/model/stgcn_lstm.py` (single-branch LSTM wrapper, not the shared building blocks) — **`820f96b`**.
- Obsolete single-branch STGCN configs and test files — **`a2837cd`**.
- Legacy STGCN dispatch logic from `train_lstm.py` — **`ce75670`**.
- Deprecated STGCN model card template — **`cd2ada0`**.
- Legacy STGCN dispatch test — **`c28a809`**.
- `phono_proj_dim` config binding — **`a5bdfa1`**.

---

## Testing and Validation

- **Unit tests**: `DualBranchSTGCN_LSTM` forward pass shape, parameter count (~1.72 M), and device placement. `STGCNBlock` structural validation. `HolisticLandmarkExtractor` normalisation and empty-hand padding. `GraphSequenceBuffer` emit gating and valid-ratio enforcement. `GraphSequenceTensorDataset` shape and dtype contracts.
- **Integration test (`test_graph_datamodule.py`)**: `GraphGestureDataModule` loads a synthetic NPZ, performs group-aware split, and verifies DataLoader output shape is `(B, T, N, C)`.
- **Training runs**: multiple runs committed as artefacts across the branch; final run `dual_stgcn_lstm_v1_20260502T032810Z` achieves best val loss 0.926 at epoch 8 (val accuracy 71.5%, val Macro-F1 71.0%). Early stopping triggers at epoch 28.

---

## Outcome

The dual-branch STGCN-LSTM is integrated end-to-end in all layers of the system. The two-stream LSTM pipeline is preserved and co-selectable via checkpoint artifact type. The architecture converges to 71% val Macro-F1 at current dataset scale — well below the production target of 0.95 — indicating that the graph pipeline is functionally correct but requires larger graph-schema datasets or additional regularisation. Further tuning is tracked in the model improvement plan.
