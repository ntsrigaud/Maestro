# Work Log - LSTM Training Pipeline Foundation

## Context

This work establishes the first end-to-end LSTM training pipeline baseline, including deterministic dataset loading and split management, model training and evaluation, inference artifact export, and registry updates, while preserving feature-schema parity checks across data and runtime paths.

## Scope of Work

- Implemented baseline LSTM training pipeline components and training loop orchestration.
- Implemented data module support for deterministic train/validation/test splitting and DataLoader construction.
- Implemented model evaluation utility with accuracy, macro-F1, confusion matrix, and calibration metrics.
- Implemented inference artifact export and model registry update utilities.
- Wired training CLI driver for dataset validation, training execution, export, and registration.
- Expanded LSTM configuration schema and default model training configuration.
- Added unit and integration tests for datamodule behavior, export/registry utilities, config parity, and pipeline smoke execution.
- Added and registered an `lstm_v1` trained artifact with accompanying run metadata.
- Applied Black formatting across touched modules.

## Technical Changes

- Added training core modules:
- `src/gesture_control/training/train_lstm.py` for config parsing, model definition, training loop, early stopping, scheduler handling, checkpointing, metrics emission, and run artifact generation.
- `src/gesture_control/training/datamodule.py` for NPZ ingestion, deterministic split logic, class distribution inspection, and Torch dataloader setup.
- `src/gesture_control/training/evaluate.py` for classification and calibration reporting.
- `src/gesture_control/training/export_checkpoint.py` for inference artifact export, SHA256 checksums, and registry append behavior.
- Updated `src/gesture_control/training/__init__.py` to export LSTM pipeline symbols through package boundary.
- Updated `scripts/train_model.py` to execute full training lifecycle:
- runtime feature compatibility validation,
- config resolution and device selection,
- training execution,
- inference artifact export,
- model registry update,
- structured JSON result output.
- Updated `configs/model/lstm_default.yaml` with expanded structured sections (`model`, `training`, `splits`, `scheduler`, `export`).
- Updated `src/gesture_control/training/dataset_builder.py` to include `feature_schema_version` in dataset metadata.
- Updated project dependencies in `pyproject.toml` to include `torch` and `scikit-learn` and added mypy override for `sklearn` imports.
- Updated `models/registry.json` with a new registered entry for `lstm_v1_20260328T100850Z` and corresponding exported/checkpoint checksums.
- Added generated experiment artifacts under `experiment-model/lstm_v1_20260328T100850Z/` and exported inference artifact under `models/exported/`.

## Testing and Validation

- Added unit tests:
- `tests/unit/test_training_datamodule.py` for shape checks and deterministic splitting.
- `tests/unit/test_training_export.py` for checksum and registry update behavior.
- `tests/unit/test_config_loader.py` extension verifying orientation normalization loading.
- Added integration smoke test:
- `tests/integration/test_training_pipeline_smoke.py` validating minimal end-to-end training and artifact outputs.
- Executed `ruff check .` successfully.
- Executed `mypy src/gesture_control scripts/train_model.py` successfully.
- Executed `pytest -q` successfully.
- No test suppressions were added or removed.

## Outcome

The branch delivers a production-aligned LSTM training foundation with deterministic data handling, reproducible run artifacts, model export and registry integration, and validated quality gates. The system now supports train-to-register flow for `lstm_v1` without breaking existing runtime interfaces.
