# Work Log - Dataset Generation Foundation

## Context

This work establishes the dataset-generation baseline for LSTM training by introducing sequence-dataset construction from labeled landmark frames, integrating a CLI build script, and validating end-to-end dataset export behavior.

## Scope of Work

- Implemented a training dataset builder for windowed sequence sample creation.
- Exported dataset builder components through the training package boundary.
- Implemented a dataset build script for JSONL-to-NPZ conversion.
- Added unit tests for builder behavior, label strategy, validation, and serialization.
- Added integration test coverage for script-level dataset generation flow.
- Applied formatting normalization across training components and tests.

## Technical Changes

- Added `SequenceDatasetBuilder`, `LabeledLandmarkFrame`, and `SequenceSample` in the training dataset builder module.
- Implemented per-frame ingestion flow that transforms landmarks into feature frames, buffers temporal windows, aligns labels, and emits sequence samples.
- Implemented configurable label alignment strategies (`last`, `majority`) with vocabulary-to-label-id mapping.
- Implemented recording-boundary state reset to avoid cross-recording temporal leakage.
- Implemented compressed NPZ export with sequence tensors, labels, label IDs, sample metadata, and JSON-encoded dataset metadata.
- Implemented NPZ loading utility for downstream training/evaluation consumption.
- Added CLI dataset build path that reads labeled JSONL frames, loads runtime/window configuration, resolves vocabulary, and writes training-ready NPZ artifacts.
- Exported training dataset builder symbols from the training package initializer.

## Testing and Validation

- Added unit tests for sequence shape correctness, label-strategy behavior, unknown-label validation, and serialization roundtrip.
- Added integration test executing the dataset build script through subprocess and validating generated artifact schema/shape.
- Executed `pytest -q` successfully.
- Executed `ruff check .` successfully.
- Executed `mypy src/gesture_control scripts/build_dataset.py` successfully.
- No test suppressions were added or removed.
- No dependency or build-system modifications were introduced in this branch.

## Outcome

The branch delivers a reproducible dataset-generation foundation that converts labeled landmark streams into model-ready sequence datasets with explicit metadata contracts and validated script automation. This extends training readiness without introducing breaking API changes.
