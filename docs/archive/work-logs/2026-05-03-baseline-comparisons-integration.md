# Work Log — Baseline Comparisons Integration and Runtime Model Consolidation

**Date:** 2026-05-03  
**Branch:** `docs/baseline-comparisons`  
**Author:** ntsrigaud

---

## Context

This branch integrates the post-main model evolution stream into a single reviewed baseline-comparisons line: dual-branch STGCN research enablement, EnhancedTwoStreamLSTM implementation and evaluation, repository-wide default alignment to the enhanced runtime artifact, and CI trigger hardening. The objective was to converge architecture, training, runtime loading, tests, and documentation onto a coherent production path while preserving comparative baseline evidence.

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
- Added strict vocabulary filtering in dataset assembly to prevent out-of-scope label leakage.
- Added flat-format dataset output mode for feature-schema-v5 compatibility.
- Added and retained comparative experiment artifacts for STGCN, two-stream LSTM baseline, and enhanced two-stream runs.

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
- `src/maestro/training/train_lstm.py`: model dispatch, optimizer/training loop updates, scheduler/early-stopping fixes, and enhanced model support.
- `src/maestro/training/export_checkpoint.py`, `src/maestro/infrastructure/model/checkpoint_loader.py`: enhanced and dual-branch artifact contract support and loader/export alignment.

### Graph and data pipeline infrastructure
- `src/maestro/pipeline/graph_schema.py`, `src/maestro/pipeline/graph_buffering.py`: graph-schema constants, adjacency logic, normalization, and sequence buffering.
- `src/maestro/training/datamodule.py`, `src/maestro/training/augmentation.py`, `src/maestro/training/dataset_builder.py`: graph/flat data modules, augmentation, and dataset building paths.
- `scripts/build_ipn_dataset.py`: strict vocabulary filtering and flat output mode.

### Configuration defaults and runtime source
- `configs/model/two_stream_attn.yaml`, `configs/model/two_stream_attn_ipn32.yaml`, `configs/model/two_stream_lstm_ipn32.yaml`, `configs/model/dual_branch_stgcn_lstm.yaml`: comparative and production-aligned model configs.
- `configs/model_source.yaml`, `src/maestro/_defaults/configs/model_source.yaml`: default artifact target aligned to enhanced runtime model.

### Documentation and benchmark evidence
- `docs/architecture/*`: architecture views aligned to enhanced defaults and updated dataflow semantics.
- `docs/ml-system/feature-schema.md`, `docs/ml-system/training-history-summary.md`: schema reference and consolidated training/ablation history.
- `notebooks/*` and `experiments/runs/*`: baseline-comparison notebooks and supporting run artifacts for reproducible comparisons.

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
- Applied lint/format normalization and CI-focused fixes, including deferred sklearn import behavior for non-training execution paths.
- CI workflow trigger scope reduced to main push events to improve signal-to-noise and pipeline stability.

---

## Outcome

The branch consolidates model-research progression and production alignment into a single coherent system state: comparative baseline infrastructure is preserved, EnhancedTwoStreamLSTM is established as the operational default, architecture and schema documentation are synchronized with runtime behavior, and CI execution scope is hardened. No public API-breaking changes were introduced for end-user runtime commands; changes are primarily internal architecture, model-contract, and documentation alignment.
