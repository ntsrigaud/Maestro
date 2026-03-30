# Work Log - Training Pipeline Improvements

## Context

This work resolves a critical data leakage issue in the LSTM training pipeline by replacing random window-level dataset splitting with recording-level group splitting. It also introduces on-the-fly data augmentation (temporal speed scaling, spatial jitter) to improve model generalization and implements an interactive data collection mode to facilitate the creation of properly segmented gesture datasets.

## Scope of Work

- Implemented recording-level group splitting in the data module to prevent data leakage between train, validation, and test sets.
- Implemented on-the-fly gesture sequence augmentation with temporal speed scaling and spatial jitter.
- Wired augmentation configuration into the training pipeline, enabled by default for the training set.
- Implemented an interactive data collection mode in `scripts/collect_data.py` to enable segmented gesture recording.
- Added a comprehensive data collection and training guide documenting the new workflow.
- Added unit tests for the new augmentation module and group splitting logic.
- Trained and registered an updated LSTM model (`lstm_v1_20260330T104422Z`) on a new dataset created with the improved collection and splitting methodology.

## Technical Changes

- Updated `src/gesture_control/training/datamodule.py` to use `sklearn.model_selection.GroupShuffleSplit`, using `recording_id` to ensure that all frames from a single recording are confined to a single data split (train, validation, or test). Includes a fallback to random splitting with a `UserWarning` if fewer than 3 unique recording IDs are present.
- Added `src/gesture_control/training/augmentation.py` containing `GestureAugmenter` to perform on-the-fly temporal speed scaling (simulating faster/slower gestures) and spatial jitter (adding Gaussian noise to position features).
- Created `AugmentedSequenceTensorDataset` in `datamodule.py` to apply augmentations to the training set during `__getitem__` calls.
- Updated `src/gesture_control/training/train_lstm.py` to construct and pass the `GestureAugmenter` to the training dataset based on the `augmentation` configuration.
- Extended `configs/model/lstm_default.yaml` with an `augmentation` section to control `enabled`, `temporal_scale_range`, `spatial_jitter_sigma`, and `apply_prob`.
- Enhanced `scripts/collect_data.py` with an `--interactive` mode that provides a live camera preview and uses the SPACE key to start and stop individual gesture recordings, automatically creating unique `recording_id`s for each segment.
- Created `docs/DATA_COLLECTION_GUIDE.md`, a detailed guide explaining the new data collection protocol, dataset building process, and updated training pipeline behavior.
- Updated `models/registry.json` to register the new `lstm_v1_20260330T104422Z` model.

## Testing and Validation

- Added `tests/unit/test_augmentation.py` with comprehensive tests for temporal scaling (speed up/slow down) and spatial jitter, including deterministic seed validation.
- Extended `tests/unit/test_training_datamodule.py` with tests for `GroupShuffleSplit` logic, verifying that recording IDs are not shared across splits.
- Executed `ruff check .` successfully after fixing one import order issue.
- Executed `mypy src/gesture_control --ignore-missing-imports` successfully.
- Executed `pytest -q` successfully, with 78 tests passing (20 new tests). A `UserWarning` related to the group split fallback is expected in the smoke test due to limited synthetic data.
- No test suppressions were added or removed.

## Outcome

The branch delivers a significantly more robust training pipeline that prevents data leakage and improves model generalization through data augmentation. The new interactive data collection script and accompanying guide establish a repeatable, high-quality workflow for creating gesture datasets, addressing the root cause of previously inflated validation metrics. The system is now capable of training models that learn generalizable gesture patterns rather than memorizing specific recording trajectories. No breaking API changes were introduced.
