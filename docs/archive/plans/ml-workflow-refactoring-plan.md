# ML Workflow Refactoring — Implementation Plan

**Status**: In progress  
**Branch**: `build/model-improvement`  
**Author**: Neil  
**Last updated**: 2026-04-17

---

## Overview

The goal of this refactoring is to replace the current script-only ML workflow with a
three-layer architecture that separates dataset preparation, model training, and production
inference. The table below shows the current state of each deliverable.

| Deliverable                                          | File                                         | Status     |
| ---------------------------------------------------- | -------------------------------------------- | ---------- |
| Dataset preparation notebook                         | `notebooks/01_dataset_preparation.ipynb`     | ✅ Done    |
| Model training notebook                              | `notebooks/02_model_training.ipynb`          | ✅ Done    |
| Hugging Face publishing module                       | `src/gesture_control/training/hf_publish.py` | ✅ Done    |
| `pyproject.toml` extras                              | `notebooks`, `huggingface` dependency groups | ✅ Done    |
| Public split-index properties on `GestureDataModule` | `datamodule.py`                              | ⬜ Pending |
| `update_model_registry()` call in notebook cell 11   | `notebooks/02_model_training.ipynb`          | ⬜ Pending |
| Unknown-class subsample flag                         | `scripts/build_ipn_dataset.py`               | ⬜ Pending |

The remaining items are described as concrete, ordered steps below.

---

## Three-layer architecture

```
Layer 1 — Dataset Preparation
    notebooks/01_dataset_preparation.ipynb
    scripts/build_ipn_dataset.py
    ─────────────────────────────────────────
    Output: data/processed/ipn_dataset_v4.npz

        │  (NPZ artifact — no code coupling)
        ▼

Layer 2 — Model Training and Experimentation
    notebooks/02_model_training.ipynb
    src/gesture_control/training/  (train_lstm, datamodule, evaluate, export_checkpoint)
    src/gesture_control/training/hf_publish.py  (new)
    ─────────────────────────────────────────
    Output: models/exported/*_inference.pt  +  optional HF Hub commit

        │  (.pt artifact — no training dependency)
        ▼

Layer 3 — Production Inference  (unchanged)
    src/gesture_control/{pipeline,infrastructure,app}/
    scripts/run_realtime.py
```

The runtime path (`run_realtime.py`) has zero imports from `training/`. The only
coupling between Layer 2 and Layer 3 is the `_inference.pt` file loaded by
`load_inference_artifact()`.

---

## Phase 1 — Expose Public Split Indices on `GestureDataModule`

### Why

The dataset preparation notebook (`01_dataset_preparation.ipynb`) calls
`dm.train_indices`, `dm.val_indices`, and `dm.test_indices` to build per-split class
distribution charts and to verify recording-level leakage. These names are used publicly
throughout both notebooks but the underlying attributes (`_train_idx`, `_val_idx`,
`_test_idx`) are private to `GestureDataModule`.

### File

`src/gesture_control/training/datamodule.py`

### Step 1.1 — Add three `@property` accessors

Locate the block that ends with the existing `num_classes` property (around line 175).
Add the following three properties immediately after it:

```python
@property
def train_indices(self) -> np.ndarray:
    if self._train_idx is None:
        raise RuntimeError("DataModule not initialized. Call setup() first.")
    return self._train_idx

@property
def val_indices(self) -> np.ndarray:
    if self._val_idx is None:
        raise RuntimeError("DataModule not initialized. Call setup() first.")
    return self._val_idx

@property
def test_indices(self) -> np.ndarray:
    if self._test_idx is None:
        raise RuntimeError("DataModule not initialized. Call setup() first.")
    return self._test_idx
```

No other changes are needed — `_train_idx`, `_val_idx`, and `_test_idx` are already
set during `setup()`.

### Step 1.2 — Validation

```bash
python - <<'EOF'
import sys; sys.path.insert(0, "src")
from gesture_control.training.train_lstm import load_training_config, create_datamodule
from pathlib import Path

config = load_training_config("configs/model/lstm_default.yaml")
dm = create_datamodule(dataset_path=Path("data/processed/ipn_dataset_v4.npz"), config=config)
print("train:", dm.train_indices.shape)
print("val  :", dm.val_indices.shape)
print("test :", dm.test_indices.shape)
# Verify no overlap between splits
import numpy as np
assert len(np.intersect1d(dm.train_indices, dm.val_indices))   == 0
assert len(np.intersect1d(dm.train_indices, dm.test_indices))  == 0
assert len(np.intersect1d(dm.val_indices,   dm.test_indices))  == 0
print("OK — no index overlap between splits")
EOF
```

### Step 1.3 — Unit test

Add to `tests/unit/test_datamodule.py`:

```python
def test_split_index_properties_are_disjoint(tmp_npz_fixture):
    """train/val/test index sets must be mutually exclusive."""
    dm = GestureDataModule(dataset_path=tmp_npz_fixture, ...)
    dm.setup()
    assert len(np.intersect1d(dm.train_indices, dm.val_indices))  == 0
    assert len(np.intersect1d(dm.train_indices, dm.test_indices)) == 0
    assert len(np.intersect1d(dm.val_indices,   dm.test_indices)) == 0
```

---

## Phase 2 — Fix `update_model_registry()` Call in Training Notebook

### Why

The notebook (`notebooks/02_model_training.ipynb`, cell 11) calls `update_model_registry()`
with a signature that does not match the function defined in
`src/gesture_control/training/export_checkpoint.py`.

**Current notebook call (incorrect):**

```python
update_model_registry(
    registry_path=REPO_ROOT / config.export.registry_path,
    model_version=result.model_version,
    checkpoint_path=result.checkpoint_path,
    artifact_path=artifact_path,        # ← wrong param name
    config=config,                      # ← not a parameter
    dataset_metadata=result.dataset_metadata,
    test_report=result.test_report,     # ← not a parameter
)
```

**Actual function signature:**

```python
def update_model_registry(
    *,
    registry_path: str | Path,
    model_version: str,
    checkpoint_path: str | Path,
    exported_path: str | Path,          # correct param name
    class_labels: list[str],            # must be provided explicitly
    dataset_metadata: dict[str, Any],
    feature_schema_version: str,        # must be provided explicitly
    training_data_version: str,         # must be provided explicitly
) -> dict[str, Any]:
```

### File

`notebooks/02_model_training.ipynb` — cell 11 (export artifact cell)

### Step 2.1 — Replace the registry call

Replace the incorrect call with:

```python
registry_entry = update_model_registry(
    registry_path=REPO_ROOT / config.export.registry_path,
    model_version=result.model_version,
    checkpoint_path=result.checkpoint_path,
    exported_path=artifact_path,
    class_labels=result.class_labels,
    dataset_metadata=result.dataset_metadata,
    feature_schema_version=config.export.feature_schema_version,
    training_data_version=str(result.dataset_metadata.get("build_timestamp", "unknown")),
)
print(f"Registry entry : {registry_entry['model_version']}")
```

### Step 2.2 — Validation

After running the cell, verify `models/registry.json` contains a new entry:

```bash
python -c "
import json
reg = json.load(open('models/registry.json'))
latest = reg['models'][-1]
print('Registered:', latest['model_version'])
print('Exported  :', latest['exported_path'])
"
```

---

## Phase 3 — Address Unknown-Class Overrepresentation

### Why

The `unknown` class has 4,052 sequences vs 2,406 for `swipe_right` (1.68× ratio).
`class_weighting=true` in the training config provides partial compensation through loss
reweighting, but the absolute volume asymmetry still biases the model toward seeing more
`unknown` examples per epoch. Reducing `unknown` to ≈2,400 sequences equalises the per-epoch
distribution without any loss reweighting changes.

### File

`scripts/build_ipn_dataset.py`

### Step 3.1 — Add `--unknown-subsample-ratio` CLI argument

In `build_ipn_dataset.py`, locate the `argparse` setup block and add:

```python
parser.add_argument(
    "--unknown-subsample-ratio",
    type=float,
    default=1.0,
    metavar="RATIO",
    help=(
        "Subsample the 'unknown' class to this fraction of its natural size. "
        "E.g. 0.5 keeps half. Default 1.0 (no subsampling)."
    ),
)
```

In the dataset builder invocation, filter `unknown` sequences after building:

```python
if args.unknown_subsample_ratio < 1.0:
    rng = np.random.default_rng(seed=args.seed)
    unknown_mask = np.array([s.label == "unknown" for s in sequences])
    unknown_idx  = np.where(unknown_mask)[0]
    keep_n       = int(len(unknown_idx) * args.unknown_subsample_ratio)
    drop_idx     = set(rng.choice(unknown_idx, size=len(unknown_idx) - keep_n, replace=False))
    sequences    = [s for i, s in enumerate(sequences) if i not in drop_idx]
```

### Step 3.2 — Rebuild the dataset

```bash
python scripts/build_ipn_dataset.py \
  --unknown-subsample-ratio 0.55 \
  --feature-schema-version feature-schema-v4 \
  --output data/processed/ipn_dataset_v4_balanced.npz
```

Target: `unknown` count ≈ 2,230 (≤ 1.1× of `swipe_right`).

### Step 3.3 — Verify with the dataset notebook

Open `notebooks/01_dataset_preparation.ipynb`, update `DATASET_PATH` to point to the new
balanced NPZ, and run all cells. Check that the imbalance table shows all ratios ≤ 1.15×.

### Step 3.4 — Retrain and compare

```bash
python scripts/train_model.py \
  --config configs/model/lstm_default.yaml \
  --dataset data/processed/ipn_dataset_v4_balanced.npz
```

Compare test accuracy, macro F1, and per-class `unknown` recall against baseline
(`lstm_v1_20260417T052358Z`, 77.8% test accuracy). Accept the change only if macro F1
does not regress by more than 0.5%.

---

## Phase 4 — Verify Notebooks Run End-to-End

### Why

The notebooks were authored programmatically and have not yet been executed in the target
environment. This phase validates that all cells complete without errors and that the
outputs are correct before the notebooks are used as the canonical training workflow.

### Step 4.1 — Install notebook dependencies

```bash
pip install -e ".[notebooks]"
pip install -e ".[huggingface]"   # optional, needed for cell 12
```

### Step 4.2 — Run `01_dataset_preparation.ipynb`

```bash
jupyter nbconvert --to notebook --execute \
  --ExecutePreprocessor.timeout=120 \
  --output notebooks/01_dataset_preparation.executed.ipynb \
  notebooks/01_dataset_preparation.ipynb
```

**Expected outputs (check each):**

| Cell                     | Expected output                                        |
| ------------------------ | ------------------------------------------------------ |
| Cell 4 (load NPZ)        | `sequences shape: (N, 24, 144)` with N > 20,000        |
| Cell 5 (class dist.)     | Bar chart with 8 classes; `unknown` tallest bar        |
| Cell 6 (imbalance table) | `ratio_to_min` column; `unknown` row flagged if > 1.5× |
| Cell 12 (split verify)   | `train_indices shape` printed; leakage count = 0       |

### Step 4.3 — Run `02_model_training.ipynb`

```bash
jupyter nbconvert --to notebook --execute \
  --ExecutePreprocessor.timeout=3600 \
  --output notebooks/02_model_training.executed.ipynb \
  notebooks/02_model_training.ipynb
```

**Expected outputs (check each):**

| Cell                      | Expected output                                                  |
| ------------------------- | ---------------------------------------------------------------- |
| Cell 6 (model build)      | `Forward pass OK: input [2, 24, 144] → output [2, 8]`            |
| Cell 7 (training)         | Per-epoch log lines; terminates before epoch 30 (early stopping) |
| Cell 8 (learning curves)  | Three-subplot figure; no diverging train/val loss                |
| Cell 9 (confusion matrix) | Heatmap with 8×8 grid; diagonal dominant                         |
| Cell 11 (export)          | `Artifact verified:` line + `Schema version: feature-schema-v4`  |

### Step 4.4 — Run the full test suite

```bash
pytest tests/ -q
```

All 204 tests must pass. No regressions from the `datamodule.py` property additions (Phase 1).

---

## Phase 5 — Publish Current Best Model to Hugging Face

### Prerequisites

- `HF_TOKEN` environment variable set (`hf_…`)
- A Hugging Face repository created (can be private): `{username}/gesture-control-lstm-v4`
- `02_model_training.ipynb` has been executed and `artifact_path` is set

### Step 5.1 — Set environment variables

```bash
export HF_TOKEN=hf_...
export HF_REPO_ID=blackbird410/gesture-control-lstm-v4
```

### Step 5.2 — Run notebook cell 12 or script equivalent

```python
from gesture_control.training.hf_publish import publish_to_huggingface
import os

url = publish_to_huggingface(
    result=result,
    config=config,
    test_report=result.test_report,
    inference_artifact_path=artifact_path,
    repo_id=os.environ["HF_REPO_ID"],
    token=os.environ["HF_TOKEN"],
    private=True,
)
print(f"Published: {url}")
```

### Step 5.3 — Verify the repository

Open `https://huggingface.co/{HF_REPO_ID}` and confirm:

- `README.md` renders with the model card (YAML metadata + gesture table + how-to-use)
- `config.json` is present and contains `feature_schema_version: feature-schema-v4`
- `lstm_v1_*_inference.pt` is present

### Step 5.4 — Tag the commit

```bash
git tag v1.0-lstm-v4
git push origin v1.0-lstm-v4
```

---

## Phase 6 — Update Production Model Pointer

### Why

`scripts/run_realtime.py` loads the artifact specified in `models/registry.json` under the
`production_model` key. After retraining with the balanced dataset (Phase 3), the registry
pointer must be updated if the new model version is selected for production.

### Step 6.1 — Update `models/registry.json`

```bash
python - <<'EOF'
import json
from pathlib import Path

reg_path = Path("models/registry.json")
reg = json.loads(reg_path.read_text())
reg["production_model"] = "lstm_v1_20260417T052358Z"   # replace with new version if applicable
reg_path.write_text(json.dumps(reg, indent=2))
print("Updated production pointer to:", reg["production_model"])
EOF
```

### Step 6.2 — Smoke test the runtime

```bash
python scripts/run_realtime.py --dry-run --max-frames 30 2>&1 | tail -5
```

Expected: `Inference pipeline ready` without schema validation errors.

---

## Phase 7 — Update Branch and Open Pull Request

### Step 7.1 — Commit all changes

```bash
git add \
  src/gesture_control/training/datamodule.py \
  src/gesture_control/training/hf_publish.py \
  notebooks/01_dataset_preparation.ipynb \
  notebooks/02_model_training.ipynb \
  scripts/build_ipn_dataset.py \
  models/registry.json \
  pyproject.toml \
  docs/development/ml-workflow-refactoring-plan.md

git commit -m "refactor: notebook-based ML workflow with HF publishing"
```

### Step 7.2 — Final quality gate

```bash
ruff check src/ scripts/ notebooks/
mypy src/ --ignore-missing-imports
pytest tests/ -q
```

All three commands must exit 0 before opening a PR.

### Step 7.3 — Open a pull request

Branch: `build/model-improvement` → `main`  
Title: `refactor(training): notebook workflow, HF publishing, split-index API`

---

## Appendix — Key API Reference

### `GestureDataModule` (after Phase 1)

| Member                 | Type                  | Description                                          |
| ---------------------- | --------------------- | ---------------------------------------------------- |
| `train_indices`        | `np.ndarray`          | Indices into the full dataset for the training split |
| `val_indices`          | `np.ndarray`          | Indices for the validation split                     |
| `test_indices`         | `np.ndarray`          | Indices for the test split                           |
| `label_distribution()` | `dict[str, int]`      | Full-dataset class → count mapping                   |
| `split_info()`         | `dict[str, set[str]]` | Recording IDs per split (leakage audit)              |

### `TrainingResult` dataclass

| Field              | Type               | Notes                                                                                                      |
| ------------------ | ------------------ | ---------------------------------------------------------------------------------------------------------- |
| `model`            | `nn.Module`        | Best-checkpoint model (eval mode after `train_lstm()`)                                                     |
| `model_version`    | `str`              | `lstm_v1_YYYYMMDDTHHMMSSZ` timestamp key                                                                   |
| `checkpoint_path`  | `Path`             | Path to the raw `.pt` checkpoint                                                                           |
| `run_dir`          | `Path`             | Experiment run directory with logs                                                                         |
| `train_history`    | `list[dict]`       | Keys: `epoch`, `train_loss`, `train_accuracy`, `val_loss`, `val_accuracy`, `val_macro_f1`, `learning_rate` |
| `val_report`       | `EvaluationReport` | Validation set metrics (best epoch)                                                                        |
| `test_report`      | `EvaluationReport` | Test set metrics                                                                                           |
| `class_labels`     | `list[str]`        | Ordered class names matching logit indices                                                                 |
| `dataset_metadata` | `dict[str, Any]`   | Forwarded from NPZ metadata                                                                                |

### `EvaluationReport` dataclass

| Field                 | Type               |
| --------------------- | ------------------ |
| `accuracy`            | `float`            |
| `macro_f1`            | `float`            |
| `loss`                | `float`            |
| `calibration_ece`     | `float`            |
| `mean_confidence`     | `float`            |
| `per_class_precision` | `dict[str, float]` |
| `per_class_recall`    | `dict[str, float]` |
| `confusion_matrix`    | `list[list[int]]`  |

### `update_model_registry()` signature

```python
update_model_registry(
    *,
    registry_path: str | Path,
    model_version: str,
    checkpoint_path: str | Path,
    exported_path: str | Path,           # NOT artifact_path
    class_labels: list[str],
    dataset_metadata: dict[str, Any],
    feature_schema_version: str,
    training_data_version: str,
) -> dict[str, Any]
```

### `publish_to_huggingface()` signature

```python
publish_to_huggingface(
    *,
    result: TrainingResult,
    config: TrainingPipelineConfig,
    test_report: EvaluationReport,
    inference_artifact_path: str | Path,
    repo_id: str,
    token: str | None = None,            # falls back to HF_TOKEN env var
    private: bool = True,
    commit_message: str = "...",
) -> str                                 # returns repository URL
```
