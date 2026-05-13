# Work Log — Jester-Pretrained Hybrid Training Iterations

**Date**: 2026-05-13  
**Branch**: `test/jester-pretrained-model`  
**Author**: @ntsrigaud

---

## Context

This branch consolidates the post-migration training work after the hybrid vocabulary rollout. It merges `refactor/hybrid-vocabulary-migration` and then focuses on stabilizing the Jester-pretrained → hybrid fine-tuning pipeline, correcting hybrid dataset construction logic, and promoting the latest stable inference artifact for runtime testing.

---

## Scope of Work

### 1. Hybrid dataset preparation hardening

- Added dedicated hybrid dataset builder: `scripts/build_hybrid_dataset.py`
- Added class filtering controls and exclusion support for dataset builders (including `--exclude-class` path and filtering logic)
- Corrected Jester/IPN hybrid merge behavior and class balancing constraints
- Updated notebooks for dataset preparation alignment:
  - `notebooks/04_dataset_preparation_jester_ipn.ipynb`
  - `notebooks/05_dataset_preparation_hybrid.ipynb`

### 2. Fine-tuning stabilization and transfer learning fixes

- Added linear warmup for Stage B backbone LR in `finetune_lstm()` to avoid early loss spikes caused by cold AdamW moments on previously frozen params
- Added/retained replay controls and lower sample caps for stability under mixed-source hybrid training
- Refined training config for hybrid fine-tuning:
  - `configs/model/two_stream_attn_hybrid.yaml`
  - lowered `max_samples_per_class`
  - disabled label-unsafe mirror augmentation path for directional classes
- Added `flatten_parameters()` handling in the model path to keep LSTM weights contiguous and reduce runtime/perf warnings during training

### 3. Vocabulary and neutral-class consistency

- Carried forward hybrid vocabulary migration (`unknown` → `no_gesture`) from merged PR #202 and ensured continued consistency in subsequent notebook and training iterations
- Reverted temporary regression that reintroduced `unknown` in hybrid flows

### 4. Runtime model source and threshold iteration

- Updated `configs/model_source.yaml` multiple times to point to latest training artifacts, ending with the latest stable hybrid inference checkpoint
- Temporarily lowered production thresholds for test cycles to validate runtime behavior under new training artifacts

### 5. Hygiene and formatting

- Applied ruff formatting and lint cleanup across code and notebooks

---

## Technical Changes

| File | Change |
|---|---|
| `scripts/build_hybrid_dataset.py` | New hybrid NPZ builder, source-aware merge, per-class balancing |
| `scripts/build_jester_dataset.py` | Dataset prep alignment and filtering support |
| `src/maestro/training/train_lstm.py` | Stage B backbone LR linear warmup and fine-tuning stability updates |
| `src/maestro/infrastructure/model/two_stream_attn_lstm.py` | LSTM contiguity/runtime robustness refinements |
| `configs/model/two_stream_attn_hybrid.yaml` | Hybrid fine-tune config updates (caps/augmentation/training controls) |
| `configs/model_source.yaml` | Advanced to latest stable hybrid inference artifact |
| `configs/thresholds/production_hybrid.yaml` | Lowered thresholds for testing iteration |
| `notebooks/04_dataset_preparation_jester_ipn.ipynb` | Class exclusion/filtering alignment |
| `notebooks/05_dataset_preparation_hybrid.ipynb` | Hybrid dataset prep workflow updates |
| `notebooks/05_model_training_hybrid.ipynb` | Hybrid fine-tuning iteration updates |
| `notebooks/05b_model_training_hybrid_scratch.ipynb` | Scratch training notebook lint/style and workflow alignment |

---

## Testing and Validation

- Completed multiple end-to-end training runs on hybrid subsets
- Confirmed training completion after Stage B warmup integration
- Validated artifact loading path via `configs/model_source.yaml` updates
- Notebook pipelines updated and linted for reproducible reruns

---

## Outcome

This branch establishes a stable and reproducible hybrid fine-tuning baseline on Jester-pretrained weights. The data preparation path is now safer (class exclusion/filtering + balancing), Stage B optimization is more stable (linear warmup), and runtime testing is aligned with the latest stable hybrid artifact. The result is a cleaner handoff point for production calibration and final threshold tuning.
