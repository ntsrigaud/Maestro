# Work Log — Baseline Comparisons Integration and Runtime Model Consolidation

**Date:** 2026-05-03  
**Updated:** 2026-05-06  
**Branch:** `docs/baseline-comparisons`  
**Author:** ntsrigaud

---

## Context

This branch integrates the post-main model evolution stream into a single reviewed baseline-comparisons line: dual-branch STGCN research enablement, EnhancedTwoStreamLSTM implementation and evaluation, repository-wide default alignment to the enhanced runtime artifact, and CI trigger hardening. The latest update extends this baseline-comparison line with large-scale grid search runs, 32-frame training-default consolidation, dynamic class exclusion, and memory-safe streaming for large dataset training. The objective remains convergence of architecture, training, runtime loading, tests, and documentation onto a coherent production path while preserving comparative baseline evidence.

---

## Scope of Work

### Model architecture and pipeline development

- Implemented and integrated dual-branch STGCN-LSTM research pipeline, including graph schema, graph buffering, graph datamodules, checkpoint export/load support, and runtime integration path.
- Implemented EnhancedTwoStreamLSTM and iterated stream-fusion design to final MLP-gated combined-pooling architecture.
- Added partitioned model configs for 32-frame baseline comparisons and 8-class production-oriented runs.

### Runtime and default alignment

- Set EnhancedTwoStreamLSTM as runtime/training default architecture.
- Updated model source defaults to enhanced inference artifacts from Hugging Face.
- Tightened runtime checkpoint-loader expectations to enhanced artifact path in aligned repository phase.

### Dataset, vocabulary, and experiment baselines

- Expanded IPN vocabulary coverage for baseline comparisons (11-class and then 14-class baseline contexts).
- Added explicit 14-class baseline vocabulary/config and a 10-class production vocabulary path.
- Removed click/double-click classes from selected training flows via config-driven exclusion while preserving full baseline vocab artifacts.
- Added pointing gestures to baseline vocabulary mappings.
- Added strict vocabulary filtering in dataset assembly to prevent out-of-scope label leakage.
- Corrected unknown-segment subsampling granularity (per-video rather than global-label collapse).
- Added legacy cache compatibility in dataset loading (promotion of legacy right-hand cache layout to holistic layout).
- Added flat-format dataset output mode for feature-schema-v5 compatibility.
- Added and retained comparative experiment artifacts for STGCN, two-stream LSTM baseline, and enhanced two-stream runs.

### Hyperparameter search and training-scale optimization

- Added full 24-frame and 32-frame grid-search execution tracks with persisted run metadata/results.
- Added learning-curve visualization for top grid-search candidates and synchronized notebooks.
- Updated model/training defaults from grid-search optima (including 32-frame context-window defaults).
- Increased BiLSTM depth for expressivity in comparison runs and later tuned dropout to reduce over-regularization.
- Introduced dataset-level max-samples-per-class controls (8,000 and 2,400 regimes) to stabilize split balance and class coverage.

### Testing and quality hardening

- Added/updated unit and integration test suites for dual-branch STGCN and EnhancedTwoStreamLSTM.
- Updated checkpoint loader and HF loader tests to match enhanced model artifact contracts.
- Applied lint/style normalization passes (ruff formatting and follow-up cleanups).

### Documentation and architecture synchronization

- Added dedicated EnhancedTwoStreamLSTM architecture documentation.
- Updated system/data-flow/logical/process/physical/development/training architecture views to reflect current defaults.
- Added feature-schema-v5 documentation and appended May 2026 ablation/training history summary.
- Added branch-level work-log entries for major model milestones.

### CI and workflow controls

- Deferred sklearn imports to runtime path to stabilize CI environments.
- Restricted CI workflow push triggers to the main branch to reduce unnecessary pipeline runs.

---

## Technical Changes

### Core model and training modules

- `src/maestro/infrastructure/model/dual_branch_stgcn_lstm.py`: dual-branch STGCN-LSTM classifier and phonological branch implementation.
- `src/maestro/infrastructure/model/stgcn_lstm.py`: STGCN blocks, graph-conv components, temporal attention, and architecture fixes.
- `src/maestro/infrastructure/model/two_stream_attn_lstm.py`: EnhancedTwoStreamLSTM with projected two-stream encoding, self-attention, cross-stream gating, and fused head.
- `src/maestro/training/train_lstm.py`: model dispatch, optimizer/training loop updates, scheduler/early-stopping fixes, enhanced model support, and additional training controls (`max_samples_per_class`, `exclude_classes`) for balanced large-scale experiments.
- `src/maestro/training/export_checkpoint.py`, `src/maestro/infrastructure/model/checkpoint_loader.py`: enhanced and dual-branch artifact contract support and loader/export alignment.

### Graph and data pipeline infrastructure

- `src/maestro/pipeline/graph_schema.py`, `src/maestro/pipeline/graph_buffering.py`: graph-schema constants, adjacency logic, normalization, and sequence buffering.
- `src/maestro/training/datamodule.py`: dynamic class exclusion, max-samples-per-class handling, and streaming NPZ sequence iteration to avoid full compressed-array decompression spikes.
- `src/maestro/training/dataset_builder.py`: memory-optimized dataset persistence/loading paths for large sequence artifacts.
- `src/maestro/training/augmentation.py`: maintained augmentation parity while scaling larger grid-search workloads.
- `scripts/build_ipn_dataset.py`: strict vocabulary filtering, flat output mode, corrected unknown subsampling logic, and legacy-cache compatibility.

### Configuration defaults and runtime source

- `configs/model/two_stream_attn.yaml`: updated to 10-class production track and 32-frame context defaults derived from grid-search results.
- `configs/model/two_stream_attn_ipn32.yaml`: explicit class exclusion, sample-cap, and tuned hyperparameter defaults for 32-frame baselines.
- `configs/model/two_stream_lstm_ipn32.yaml`, `configs/model/dual_branch_stgcn_lstm.yaml`: retained as comparative references.
- `configs/gestures/vocabulary_ipn_14class.yaml` and `configs/gestures/vocabulary_ipn_10class.yaml`: explicit baseline vs production vocabulary separation.
- `configs/gestures/vocabulary_ipn.yaml`: adjusted class set for the active baseline flow.
- `configs/model_source.yaml`, `src/maestro/_defaults/configs/model_source.yaml`: default artifact target aligned to enhanced runtime model.

### Documentation and benchmark evidence

- `docs/architecture/*`: architecture views aligned to enhanced defaults and updated dataflow semantics.
- `docs/ml-system/feature-schema.md`, `docs/ml-system/training-history-summary.md`: schema reference and consolidated training/ablation history.
- `notebooks/04_model_training_two-stream-attn-lstm_gridsearch.ipynb` and related notebooks: added full grid-search workflows and learning-curve analysis for top runs.
- `experiments/gridsearch_results.csv`, `experiments/gridsearch_results.json`, `experiments/gridsearch_runs/*`: persisted search-space outcomes and per-run snapshots.
- `experiments/runs/*`: additional 24/32-frame baseline and optimal-config iterations.

### CI workflow

- `.github/workflows/ci.yml`: push trigger scope restricted to main branch.

---

## Testing and Validation

- Expanded unit test coverage for:
  - EnhancedTwoStreamLSTM behavior and artifact compatibility.
  - Dual-branch STGCN-LSTM model structure and graph pipeline components.
  - Graph dataset builder and graph sequence buffering.
- Added/updated integration tests for graph datamodule and phase-2 runtime compatibility.
- Validated checkpoint export/load paths for enhanced and graph-model workflows across updated artifact contracts.
- Validated large-dataset training stability after streaming/memory optimizations in datamodule and dataset builder paths.
- Validated dynamic class exclusion and per-class sample capping behavior through full 24-frame and 32-frame grid-search/optimal reruns.
- Applied lint/format normalization and CI-focused fixes, including deferred sklearn import behavior for non-training execution paths.
- CI workflow trigger scope reduced to main push events to improve signal-to-noise and pipeline stability.

---

## Outcome

The branch consolidates model-research progression and production alignment into a single coherent system state: comparative baseline infrastructure is preserved, EnhancedTwoStreamLSTM is established as the operational default, architecture and schema documentation are synchronized with runtime behavior, and CI execution scope is hardened. The updated revision further adds scalable 24/32-frame search workflows, explicit baseline-vs-production vocabulary separation, and memory-safe large-dataset training mechanics. No public API-breaking changes were introduced for end-user runtime commands; changes are primarily internal architecture, model-contract, training-data controls, and documentation alignment.
