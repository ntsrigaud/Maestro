# Work Log — Repository Cleanup: Presentation Pivot, Two-Stream Consolidation, and Artifact Pruning

**Date:** 2026-04-23  
**Branch:** `refactor/repository-cleanup`  
**Author:** @ntsrigaud

---

## Context

This branch completes the cleanup pass that follows the real-time performance and
two-stream LSTM rollout. The repository had accumulated multiple obsolete runtime
paths: desktop-specific command adapters and configuration profiles, single-stream
and TCN model support, local registry export logic, and retained experiment run
artifacts that no longer matched the production deployment model. The objective of
this branch is to converge the codebase on the current supported operating mode:
presentation control backed by a Hugging Face-hosted two-stream LSTM artifact.

---

## Scope of Work

### Runtime and Application Refactor

- Replaced the desktop-oriented dispatch path with a dedicated
  `PresentationAdapter` in
  `src/gesture_control/infrastructure/dispatch/app_adapters/presentation_adapter.py`.
- Updated `src/gesture_control/app/bootstrap.py` to build the dispatcher around
  `PresentationAdapter` and `KeyboardAdapter`; removed `DesktopAdapter`
  construction and all desktop-only tuning inputs.
- Removed the deprecated `--registry` CLI flag from
  `src/gesture_control/app/main.py` and the corresponding bootstrap override.
- Switched `configs/app.yaml` from `presentation-meeting-control` to
  `presentation-controller` and replaced the active command map with
  `command_map_presentation.yaml`.

### Model and Training Consolidation

- Removed legacy single-stream LSTM and TCN runtime support from
  `src/gesture_control/infrastructure/model/checkpoint_loader.py`; artifact
  loading now rejects non-two-stream artifact types explicitly.
- Removed `src/gesture_control/infrastructure/model/lstm_model.py` and
  `src/gesture_control/infrastructure/model/tcn_model.py` from the supported
  inference stack.
- Restricted `src/gesture_control/training/train_lstm.py`,
  `src/gesture_control/training/export_checkpoint.py`, and
  `src/gesture_control/training/hf_publish.py` to the two-stream LSTM workflow.
- Changed `scripts/train_model.py` to default to
  `configs/model/two_stream_lstm.yaml`.

### Configuration Cleanup

- Deleted deprecated desktop config overlays:
  `configs/desktop/app.yaml`, `configs/desktop/inference.yaml`.
- Deleted deprecated model/training configuration:
  `configs/model/lstm_default.yaml`.
- Removed local artifact registry configuration from
  `configs/model_source.yaml`; runtime model loading is now Hugging Face-only.
- Finalized production swipe thresholds in
  `configs/thresholds/production_ipn.yaml` by replacing the diagnostic note with
  the confirmed stable values.
- Simplified `src/gesture_control/utils/schema_validation.py` by removing the
  `DesktopConfig` schema, desktop-specific app fields, and local registry model
  source fields.

### Test and Documentation Alignment

- Added `tests/unit/test_presentation_adapter.py` and updated dispatch
  integration tests to cover the presentation-controller use case.
- Removed obsolete test suites for `DesktopAdapter`, legacy desktop
  configuration, and local model registry export.
- Updated architecture documents (`data-flow.md`, `development-view.md`,
  `logical-view.md`, `process-view.md`, `system-overview.md`) to reflect the
  presentation pivot and two-stream-only model path.
- Added `docs/ml-system/training-history-summary.md` to preserve the promoted
  two-stream training results before deleting historical run artifacts.
- Archived deprecated desktop interaction planning documents under
  `docs/archive/plans/`.

---

## Technical Changes

**`src/gesture_control/infrastructure/dispatch/app_adapters/presentation_adapter.py`**
adds a cross-platform presentation controller adapter that maps the 8-class IPN
gesture vocabulary to universal keyboard shortcuts for slide navigation,
scrolling, view zoom, and slideshow start/stop. Platform-specific modifier keys
and slideshow shortcuts are resolved internally; unknown commands return a
non-executed `DispatchResult` rather than interrupting the runtime loop.

**`src/gesture_control/app/bootstrap.py`** now assumes a single supported
dispatch target and a single supported model source. `_build_dispatcher()` wraps
`PresentationAdapter` directly, and `_load_model_from_source()` unconditionally
loads the configured Hugging Face artifact. The legacy local-registry branch was
removed completely.

**`src/gesture_control/infrastructure/model/checkpoint_loader.py`** was reduced
to a two-stream-only loader. The constructor path now raises `ValueError` for any
artifact whose `artifact_type` or `model_name` does not identify the
two-stream architecture, preventing accidental fallback to retired model types.

**`src/gesture_control/training/export_checkpoint.py`** always exports
`two_stream_lstm_inference` artifacts and no longer updates a local registry.
`scripts/train_model.py` and `src/gesture_control/training/train_lstm.py` were
aligned to the same assumption, removing orphaned TCN and legacy LSTM branches.

**`src/gesture_control/utils/schema_validation.py`** now enforces a narrower
runtime contract: `AppConfig.command_map` defaults to
`command_map_presentation.yaml`, `ModelSourceConfig.backend` is constrained to
`huggingface`, and the default filename is the promoted production artifact
`two_stream_lstm_v1_20260420T091839Z_inference.pt`.

**Repository pruning** removed tracked experiment payloads under
`experiments/runs/` and the older `experiment-model/` snapshot while preserving a
concise audit document in `docs/ml-system/training-history-summary.md`.

---

## Testing and Validation

- Added a dedicated unit test suite for `PresentationAdapter` covering command
  routing and non-executed fallback behavior.
- Updated `tests/integration/test_gesture_dispatch_pipeline.py` and related
  model-loading tests to use the presentation adapter and two-stream inference
  artifacts.
- Updated checkpoint loader, Hugging Face loader, and training export tests to
  assert two-stream-only behavior and to reject legacy artifact types.
- Removed obsolete tests for `DesktopAdapter`, desktop configuration schemas,
  and local registry export paths after the corresponding code paths were
  deleted.
- Applied Black formatting to Python source and test files to restore style
  consistency after the cleanup pass.

---

## Outcome

The repository now reflects a single supported product posture: presentation
control driven by a Hugging Face-hosted two-stream LSTM model. Legacy desktop,
single-stream, TCN, and local-registry paths have been removed from runtime,
training, configuration, tests, and architecture documentation. The result is a
smaller and less ambiguous codebase with tighter interface boundaries and no
breaking change within the supported deployment model, because the active runtime
path has been updated consistently end-to-end. Net change: 106 files changed,
622 insertions, 9,219 deletions.
