# Work Log — EnhancedTwoStreamLSTM and IPN Baseline Comparisons

**Date:** 2026-05-03  
**Branch:** `feat/enhanced-two-stream-LSTM`  
**Author:** ntsrigaud

---

## Context

The baseline `TwoStreamGestureLSTM` was measured against the IPN Hand 32-frame skeleton benchmark to position the Maestro pipeline relative to video-based methods (ResNeXt-101 86.3%, C3D 77.8%). The flat-gated two-stream design had no capacity for cross-stream interaction beyond static scalar gates, and mean-only temporal pooling missed transient gesture events. This branch introduces `EnhancedTwoStreamLSTM` — a BiLSTM + Multi-Head Self-Attention + MLP-gated architecture — through iterative design and root-cause analysis, establishes the 32-frame and 24-frame IPN comparison notebooks, expands the vocabulary to the full 11-class IPN set, adds an 8-class production vocabulary, and fixes strict vocabulary filtering in the dataset assembly script.

The branch also reverts the real-time runtime to the two-stream path: the STGCN hot-loop paths added in the previous branch are stripped from `AppContainer`, `build_container`, and the runtime processing loop. The `DualBranchSTGCN_LSTM` architecture and its training/data infrastructure remain intact for offline research.

---

## Scope of Work

### Runtime reversion to two-stream path

The live inference path is simplified back to the flat two-stream pipeline. `AppContainer` slots for `holistic_extractor` and `graph_buffer` are removed. `build_container` no longer dispatches on `graph_schema_version`. The runtime hot loop contains no STGCN branching.

### IPN 32-frame baseline comparison

A new `TwoStreamGestureLSTM` configuration for 32-frame IPN windows establishes a skeleton-only baseline. Two training runs are committed for benchmarking purposes.

### EnhancedTwoStreamLSTM — architecture design and validation

Three architecture iterations were trained and compared:

- **v1**: feature-projection + BiLSTM + self-attention + scalar stream gating (original enhanced design).
- **v2**: replaced scalar gating with full MHA cross-attention + sinusoidal positional encoding.
- **v3**: root-cause analysis of five regression factors in v2 led to: removal of PE (conflicts with BiLSTM temporal encoding); removal of cross-attention (degenerate mean-pooled query equivalent to a weighted mean); mean+max temporal pooling to capture both sustained and transient events; 2-layer bottleneck MLP cross-stream gates; downsize to `hidden_size=96` (~1.20 M params) to close the overfitting gap.

### Dataset vocabulary expansion and strict filtering fix

The IPN vocabulary was expanded to the full 11-class set. An 8-class production vocabulary (no click variants) was added. The dataset assembly script was corrected to filter assembled segments against the target vocabulary, preventing label errors when building subsets.

### AdamW migration and training improvements

The optimiser was changed from `Adam` to `AdamW` across all models. Per-epoch timing was added to training history. Temporal augmentation interpolation was vectorised.

---

## Technical Changes

### `src/maestro/app/bootstrap.py` and `src/maestro/app/runtime.py`

- `AppContainer`: removed `holistic_extractor` and `graph_buffer` slots; reverted to flat two-stream container.
- `build_container`: removed STGCN dispatch block; always builds `SlidingWindowSequenceBuffer` and `HandsExtractor`.
- Runtime hot loop: removed `is_stgcn` branch; flat hand-selection + feature-engineering path only.

### `src/maestro/infrastructure/model/two_stream_attn_lstm.py` (new)

- `EnhancedTwoStreamLSTM`: input `(B, T, 147)` → split into pose stream (73 dims) and motion stream (74 dims).
- `_split`: static method; pose = positions `[0:63]` + angles `[134:144]`; motion = spread `[63:68]` + trajectory `[68:71]` + velocity `[71:134]` + wrist_vel `[144:147]`.
- `proj_a` / `proj_b`: `Linear(73/74 → proj_dim) + LN + GELU`.
- `lstm_a` / `lstm_b`: `nn.LSTM(proj_dim, hidden_size, num_layers, bidirectional=True, batch_first=True)`; BiLSTM output = `2 × hidden_size`.
- `ln_a` / `ln_b`: pre-attention LayerNorm.
- `mha_a` / `mha_b`: `nn.MultiheadAttention(embed_dim=2H, num_heads, dropout, batch_first=True)` — Q/K/V self-attention with residual + post-LN (`ln_post_a/b`).
- `pool_ln_a` / `pool_ln_b`: LayerNorm applied after `mean + max` temporal pooling.
- `gate_a` / `gate_b`: 2-layer bottleneck MLP (`2H → H → 2H`) with `Tanh + Sigmoid`; `gate_a` reads `ctx_b`; gated residual applied to `ctx_a`; followed by `ln_cross_a/b`.
- `head`: `LN(4H) → Linear(4H → 2H) → GELU → Dropout → Linear(2H → C)`.
- Guards: raises `ValueError` if `2 × hidden_size` is not divisible by `num_heads`.

### `src/maestro/training/train_lstm.py`

- `ModelConfig`: added `num_heads: int = 4`, `proj_dim: int = 64`.
- `TrainingConfig`: added `label_smoothing: float = 0.0`; parsed from YAML `training.label_smoothing`.
- `_build_model`: added `two_stream_attn_*` dispatch (checked before `two_stream_*`) → `EnhancedTwoStreamLSTM`.
- Optimiser changed from `Adam` to `AdamW` for all model types.
- `epoch_secs` logged per-epoch in training history dictionary.
- Augmentation temporal interpolation vectorised using `np.interp` instead of a Python loop.
- Removed `create_datamodule` graph-model dispatch (consistent with runtime revert; STGCN training still works via direct script invocation).

### `src/maestro/training/export_checkpoint.py`

- Added `is_two_stream_attn = config.model.model_name.startswith("two_stream_attn")` detection.
- `two_stream_attn_inference` artifact type: `model_config` dict includes `num_heads` and `proj_dim` in addition to standard fields.
- Non-attn two-stream and dual-stgcn paths unchanged.

### `src/maestro/infrastructure/model/checkpoint_loader.py`

- `_build_model_from_artifact`: dispatches `two_stream_attn_inference` (and fallback `model_name.startswith("two_stream_attn")`) → `EnhancedTwoStreamLSTM(num_heads, proj_dim)`.
- Removed `dual_stgcn_lstm_inference` dispatch (runtime revert).

### `scripts/build_ipn_dataset.py`

- **Fix**: added `if IPN_TO_VOCAB[row.ipn_label] not in vocabulary_set: continue` in `assemble_graph_dataset_from_holistic` before pushing segments to the builder. Previously, segments whose mapped label was absent from the target vocabulary (e.g. `click_one` not in the 8-class vocabulary) were passed to the builder, which raised `ValueError: Unknown label not found in vocabulary`.
- Added `--format flat` CLI flag: builds a flat feature-schema-v5 NPZ (suitable for `GestureDataModule`) reusing holistic landmark cache; produces same output shape as the existing two-stream dataset pipeline.
- `IPN_TO_VOCAB` expanded with G01 `click_one`, G02 `click_two`, G08 `double_click_one`, G09 `double_click_two` (included in the 12-class vocabulary; excluded by the strict vocabulary filter for the 8-class path).

### `configs/gestures/vocabulary_ipn.yaml`

- Expanded from 8 to 12 classes: added `click_one`, `click_two`, `double_click_one`, `double_click_two` from IPN G01/G02/G08/G09.

### `configs/gestures/vocabulary_ipn_8class.yaml` (new)

- 7 presentation-control gestures + `unknown` = 8 classes.
- IPN source mapping documented inline.
- `--vocab-config` CLI usage documented.

### `configs/model/two_stream_lstm_ipn32.yaml` (new)

- Baseline 32-frame IPN configuration for `TwoStreamGestureLSTM` (12-class).

### `configs/model/two_stream_attn_ipn32.yaml` (new)

- `EnhancedTwoStreamLSTM` 32-frame IPN comparison configuration.
- Documents six root causes analysed: PE+BiLSTM redundancy, out-dropout interference, degenerate cross-attention query, model capacity vs dataset size, mean-only pooling, LR schedule aggressiveness.
- Final hyperparameters: `hidden_size=96`, `proj_dim=96`, `num_heads=8`, `dropout=0.35`, `lr=9e-5`, `weight_decay=0.0005`, `scheduler.patience=8`, `label_smoothing=0.05`.

### `configs/model/two_stream_attn.yaml` (new)

- Production 8-class 24-frame configuration for `EnhancedTwoStreamLSTM` v3.
- `num_classes: 8`, `window_length=24`, `batch_size=64`, early stopping `patience=25`.

### `src/maestro/training/hf_publish.py`

- Added `EnhancedTwoStreamLSTM` model card template.
- `publish_to_huggingface` selects model card based on artifact type.
- Feature schema description table updated for feature-schema-v5 partition layout.

### `notebooks/03_dataset_preparation_ipn32.ipynb` (new)

- Offline MediaPipe Holistic landmark extraction for IPN Hand 32-frame comparison.

### `notebooks/04_model_training_ipn32.ipynb` (new)

- Trains and evaluates `TwoStreamGestureLSTM` at 32-frame IPN (12-class).
- Records baseline to position against video-model benchmarks.

### `notebooks/05_model_training_attn_ipn32.ipynb` (new)

- Trains and evaluates `EnhancedTwoStreamLSTM` at 32-frame IPN.
- §5.5 documents per-class recall analysis and six regression root causes addressed in v3.
- Overfitting gap tracking plots included.

### `notebooks/03_model_training_two-stream-attn-lstm.ipynb` (new)

- Trains and evaluates `EnhancedTwoStreamLSTM` on 8-class 24-frame production dataset.
- Section 12: side-by-side architecture comparison against `TwoStreamGestureLSTM` baseline.
- Sections: config, dataset load, model architecture, training, learning curves, test evaluation, confusion matrix, per-class recall, export, HF publish, comparison.

---

## Testing and Validation

- **Unit tests** (`tests/unit/test_enhanced_two_stream_attn.py`, new):
  - Forward pass shape for varying `(batch, seq, classes)` combinations including `T=24`.
  - Output is raw logits (not probabilities).
  - `_split` coverage: all 147 dims accounted for, no overlap.
  - `num_heads` divisibility guard.
  - Parameter count: small config 400 K–700 K; large config 900 K–1.5 M.
  - BiLSTM bidirectionality; MHA `embed_dim = 2 × hidden_size`.
  - `pool_ln` and `gate` module type assertions.
  - Gradient flow with random input (all parameters reached, no dead gradients).
  - Checkpoint round-trip (numerical equivalence within `atol=1e-6`).
  - `checkpoint_loader` integration test for `two_stream_attn_inference` artifact type.

- **Training runs**: 12 runs across all architecture versions committed as artefacts:
  - v1 lr sweep: 3 runs, val F1 ~73%.
  - v2 cross-attention: 2 runs, val F1 71.6%–73.7%.
  - v3 final 8-class: best run `064347Z` at **val Macro-F1 80.8%**, epoch 38.

---

## Outcome

`EnhancedTwoStreamLSTM` (v3) establishes a new performance ceiling for the flat-feature two-stream pipeline. At 8-class 24-frame it achieves 80.8% val Macro-F1 — a 9-point improvement over the comparable flat-gated `TwoStreamGestureLSTM` baseline. The mean + max pooling change was confirmed critical by the v3 ablation: the mean-only v3 run (`050237Z`) scored only 68.9% on 12 classes. The model is a drop-in replacement for `TwoStreamGestureLSTM` in the inference runtime and is the new production-ready architecture for real-time gesture control.
