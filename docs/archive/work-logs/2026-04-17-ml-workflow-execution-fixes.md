# Work Log — ML Workflow Execution Fixes

**Date:** 2026-04-17
**Branch:** `fix/ML-workflow`
**Author:** @blackbird410

---

## Context

The `refactor/ml-workflow` branch established the notebook-driven ML workflow and the
Hugging Face publishing module, but left three items explicitly noted as pending in its
implementation plan before the notebooks were end-to-end executable:

1. `GestureDataModule` did not expose `train_indices`, `val_indices`, or `test_indices`
   as public properties, so the notebooks could not index into the raw NPZ arrays to
   obtain split-specific labels, recording IDs, or feature statistics.
2. The `update_model_registry()` call in the training notebook used the wrong argument
   names (`artifact_path` instead of `exported_path`, and omitted the required
   `class_labels`, `feature_schema_version`, and `training_data_version` keyword
   arguments), causing a `TypeError` at runtime.
3. `scripts/build_ipn_dataset.py` had no mechanism to reduce `unknown`-class
   over-representation in the built dataset; the class had no subsample flag.

This branch resolves all three blockers and makes the full ML workflow executable from
start to finish.

---

## Scope of Work

| Area                                     | Change type                                                                                                     |
| ---------------------------------------- | --------------------------------------------------------------------------------------------------------------- |
| `GestureDataModule`                      | Enhancement — three public index `@property` accessors                                                          |
| `scripts/build_ipn_dataset.py`           | Enhancement — `--unknown-subsample-ratio` flag + post-build filter                                              |
| `notebooks/01_dataset_preparation.ipynb` | Fix — use `dm.train_indices` / `dm.val_indices` / `dm.test_indices` for split-level analysis and leakage checks |
| `notebooks/02_model_training.ipynb`      | Fix — correct `update_model_registry()` call signature                                                          |

---

## Technical Changes

### `GestureDataModule` — Split Index Properties

**File:** `src/gesture_control/training/datamodule.py`

Three `@property` accessors added:

```python
@property
def train_indices(self) -> np.ndarray: ...

@property
def val_indices(self) -> np.ndarray: ...

@property
def test_indices(self) -> np.ndarray: ...
```

Each raises `RuntimeError("DataModule not initialized. Call setup() first.")` if
accessed before `setup()` has been called, giving a clear error instead of a bare
`AttributeError` on `None`. The underlying `_train_idx`, `_val_idx`, and `_test_idx`
attributes were already populated by `setup()`; this change only surfaces them through
a stable public interface.

These properties allow notebook cells to index directly into the raw NPZ arrays:

```python
dm.setup()
train_labels = labels[dm.train_indices]
train_recs = set(rec_ids[dm.train_indices].tolist())
```

This is used by both notebooks: `01_dataset_preparation.ipynb` for per-split class
balance and recording-level leakage verification; `02_model_training.ipynb` for split
size reporting.

### `build_ipn_dataset.py` — Unknown-Class Subsampling

**File:** `scripts/build_ipn_dataset.py`

A new `--unknown-subsample-ratio` CLI argument (type `float`, default `1.0`) applies a
post-build, sequence-level random subsample of the `unknown` class:

```
--unknown-subsample-ratio R
    Fraction of unknown-class sequences to keep (0.0–1.0).
    Applied after all other filtering. Default: 1.0 (keep all).
```

The implementation shuffles `unknown` sequences using the existing `--seed` value for
reproducibility, retains the first `max(1, int(N * R))` sequences, and rebuilds
`builder._samples` excluding the dropped entries. A log line at `INFO` level reports
the before/after count.

The `feature_schema_version` argument is also now propagated from the CLI
(`--feature-schema-version`) through to `_save_metadata()`, which previously hardcoded
`"feature-schema-v1"` regardless of what schema was actually used to build the dataset.

### Notebook Execution Fixes

**`notebooks/01_dataset_preparation.ipynb`**

The split-analysis section now calls `dm.setup()` and uses `dm.train_indices`,
`dm.val_indices`, and `dm.test_indices` to produce split-level label distributions and
recording-level leakage checks. Previously these cells referenced undefined variables
and would raise `NameError` on execution.

**`notebooks/02_model_training.ipynb`**

The `update_model_registry()` call was corrected to match the function signature:

```python
# Before (broken):
update_model_registry(
    ...,
    artifact_path=artifact_path,
)

# After (correct):
update_model_registry(
    ...,
    exported_path=artifact_path,
    class_labels=result.class_labels,
    feature_schema_version=config.export.feature_schema_version,
    training_data_version=str(result.dataset_metadata.get("build_timestamp", "unknown")),
)
```

The cell also adds a round-trip artifact verification block (load → forward pass → shape
check) immediately after registration.

---

## Testing and Validation

- **203 tests pass, 0 failures.**
- `ruff check src/ tests/` — clean.
- No new tests were added; the `GestureDataModule` index properties are covered
  implicitly by the existing datamodule integration tests that call `setup()` and
  inspect split sizes.

---

## Outcome

The ML workflow is now end-to-end executable. Running `notebooks/01_dataset_preparation.ipynb`
followed by `notebooks/02_model_training.ipynb` completes without errors from dataset
inspection through model training, artifact export, registry update, and optional Hugging
Face publishing. The `unknown`-class imbalance can be controlled at dataset build time
via `--unknown-subsample-ratio`. The `feature_schema_version` written into NPZ metadata
now reflects the schema actually used rather than a hardcoded default.
