# Work Log - Notebook-Based ML Workflow and Hugging Face Publishing

## Context

This work introduces a three-layer ML workflow architecture that replaces the previous
script-only training approach. The motivation is to separate dataset preparation, model
training and experimentation, and production inference into explicit, decoupled stages
that are individually reproducible. The only coupling between layers is file artifacts:
an NPZ dataset file between Layer 1 and Layer 2, and an exported `.pt` inference artifact
between Layer 2 and Layer 3. The production runtime path has zero imports from the
training package.

The branch also introduces Hugging Face Hub publishing as an optional output step, enabling
trained artifacts to be stored and shared through a versioned model repository alongside a
rendered model card.

## Scope of Work

**Architecture and Planning**

- Authored the `docs/development/ml-workflow-refactoring-plan.md` implementation plan
  describing the three-layer architecture, the concrete deliverables, and the ordered
  implementation phases for the full workflow refactoring.

**Dependency Groups**

- Added a `notebooks` optional dependency group to `pyproject.toml` including
  `scikit-learn`, `matplotlib`, `seaborn`, `pandas`, `jupyter`, and `ipykernel`.
- Added a `huggingface` optional dependency group including `huggingface_hub>=0.22.0`.
- Added a `training` pytest marker so training-specific tests can be filtered
  independently from the core runtime test suite.

**Hugging Face Hub Publishing Module**

- Implemented `src/gesture_control/training/hf_publish.py` as a new publishing module
  that wraps `HfApi` and produces a reproducible two-commit upload of a trained model.
- The first commit uploads a rendered `README.md` model card and a `config.json`
  structured metadata snapshot; the second commit uploads the binary inference artifact.
- The model card includes YAML front-matter metadata for the Hugging Face model index,
  a gesture class description table, the feature schema breakdown, per-epoch training
  configuration, and per-class evaluation metrics from the held-out test set.
- The module supports both LSTM and dilated TCN architectures and branches the model card
  content accordingly.
- `huggingface_hub` is treated as a soft dependency: the module raises a clear
  `ImportError` when the optional group is not installed rather than failing at import
  time.
- Exposed `build_model_card()` as a public helper so callers can preview the rendered
  card without uploading.

**Dataset Preparation Notebook**

- Added `notebooks/01_dataset_preparation.ipynb` as the Layer 1 canonical workflow for
  building, validating, and analyzing the NPZ training dataset.
- The notebook covers dataset loading, class distribution visualization, per-split class
  balance verification, and recording-level leakage checks.

**Model Training Notebook**

- Added `notebooks/02_model_training.ipynb` as the Layer 2 canonical workflow for model
  construction, training, evaluation, artifact export, and optional Hugging Face publishing.
- The notebook covers forward-pass verification, epoch-level training loop with early
  stopping, learning-curve and confusion-matrix visualization, inference artifact export,
  and a publishing cell wired to `publish_to_huggingface()`.

## Technical Changes

- `docs/development/ml-workflow-refactoring-plan.md`: Added implementation plan
  documenting the three-layer architecture, deliverable status table, and per-phase
  implementation steps.
- `pyproject.toml`: Added `notebooks` and `huggingface` optional dependency groups;
  added a `training` pytest marker for filtering training-dependent tests.
- `src/gesture_control/training/hf_publish.py`: New module implementing
  `publish_to_huggingface()` and `build_model_card()`. Handles repository creation,
  model card rendering from a template, `config.json` snapshot assembly, and
  two-stage artifact upload. Token resolution falls back to the `HF_TOKEN` environment
  variable.
- `notebooks/01_dataset_preparation.ipynb`: Layer 1 dataset preparation and validation
  workflow.
- `notebooks/02_model_training.ipynb`: Layer 2 model training, evaluation, export, and
  publishing workflow.

## Pending Work (Tracked in Implementation Plan)

The following items from the plan are out of scope for this branch and remain pending:

- Phase 1: Expose `train_indices`, `val_indices`, and `test_indices` as public
  `@property` accessors on `GestureDataModule`.
- Phase 2: Correct the `update_model_registry()` call in notebook cell 11 to match the
  actual function signature (`exported_path` instead of `artifact_path`;
  `class_labels`, `feature_schema_version`, and `training_data_version` as explicit
  arguments).
- Phase 3: Add an `--unknown-subsample-ratio` flag to `scripts/build_ipn_dataset.py`
  to address `unknown`-class overrepresentation.

## Testing and Validation

- Executed `pytest`: all existing tests pass.
- Executed `ruff check .`: all checks pass.
- No existing tests were modified. The new `huggingface` dependency group is an
  optional install and the `hf_publish` module is not exercised by the current test suite.

## Outcome

The branch establishes the structural foundation for a reproducible, notebook-driven ML
workflow. Dataset preparation and model training are now available as interactive,
self-documenting notebooks with clear artifact handoff points. Trained models can be
published to Hugging Face Hub with a rendered model card and structured metadata in a
single function call. The production runtime is unchanged and retains zero coupling to
the training package. The pending implementation plan phases address the remaining
integration gaps before the full workflow is end-to-end executable.
