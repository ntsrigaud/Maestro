# Work Log — Horizontal Swipe Confusion Fix and Feature Schema v5

**Date:** 2026-04-18  
**Branch:** `fix/horizontal-swipes-confusion`  
**Author:** @ntsrigaud

---

## Context

`swipe_left` and `swipe_right` had a persistently elevated confusion rate across all
LSTM runs on `feature-schema-v4`. Root cause analysis identified two coupled issues:

1. **Missing translational signal.** The wrist-relative velocity block (dims 71–133)
   normalises all landmarks against the wrist origin, so for a pure lateral swipe—where
   all landmarks translate together—the velocity features are near-zero and provide no
   directional information. The classifier was forced to rely on fingertip spread changes
   and joint angles, both of which are ambiguous for mirror-symmetric swipe directions.

2. **Incorrect on-the-fly mirror augmentation.** The `mirror_flip_prob` augmentation in
   `AugmentationConfig` flipped x-coordinates but returned the original label unchanged,
   directly injecting label noise into `swipe_left` and `swipe_right` and compounding
   the confusion rather than helping the model learn the distinction.

This branch resolves both issues and fixes two related infrastructure bugs discovered
during the investigation: incorrect relative path resolution in the registry loader, and
a `KeyError` when loading NPZ files that lack optional timestamp / sample-ID fields.

---

## Scope of Work

| Area | Change type |
|---|---|
| Feature engineering | Enhancement — 147-dim `feature-schema-v5` (global wrist velocity) |
| Augmentation | Fix — disable on-the-fly mirror flip; extend `_X_COORD_DIMS` for v5 |
| Dataset notebook | Enhancement — offline label-aware mirror augmentation section |
| NPZ loader | Fix — tolerate missing optional fields (`sample_ids`, timestamps) |
| Registry export | Fix — store paths relative to registry file |
| Registry loader | Fix — resolve relative paths against registry directory |
| Config | Update — `input_size` 144 → 147; `mirror_flip_prob` → 0.0; schema v4 → v5 |
| Tests | Alignment — all tests updated to 147-dim / `feature-schema-v5` |
| Experiment artefacts | Added — 3 LSTM v5 runs registered |

---

## Technical Changes

### Feature Engineering — Global Wrist Velocity (dims 144–146)

**File:** `src/gesture_control/pipeline/feature_engineering.py`

A new three-dimensional block appended to the feature vector captures the instantaneous
velocity of the wrist landmark in MediaPipe's camera-normalized coordinate system
(independent of the wrist-centered normalisation applied to all other features):

```
Feature vector layout (feature-schema-v5, 147 dims):
  [0:63]    normalised positions (21 × 3)
  [63:68]   fingertip spread (5)
  [68:71]   wrist trajectory — net displacement from oldest frame (3)
  [71:134]  wrist-relative velocity (21 × 3)
  [134:144] joint angles (10)
  [144:147] wrist velocity in camera-normalized coordinates (3)  ← v5
```

Implementation: `_prev_raw_wrist` (a `list[float] | None`) stores the preceding frame's
raw wrist position. `wrist_vel_raw` is computed before the wrist history is updated so
it reflects the true per-frame delta. On the first frame, or after a gesture boundary
reset via `_clear_velocity_state()`, the block is zeroed.

`_zero_vector()` now returns `[0.0] * 147`. `_clear_velocity_state()` sets
`_prev_raw_wrist = None`.

### Augmentation — Mirror Flip Corrected

**File:** `src/gesture_control/training/augmentation.py`

The `_X_COORD_DIMS` constant is extended with `_WRIST_VEL_START` (index 144) so the
camera-space wrist velocity x-coordinate is also negated during a mirror flip,
ensuring a mirrored `swipe_right` carries a correctly signed directional signal.

**File:** `configs/model/lstm_default.yaml`

`mirror_flip_prob` is set to `0.0` with an explanatory comment:

> Mirror flip is intentionally disabled here.  
> Applying it without label remapping injects label noise into swipe_left/right.  
> Use the label-aware mirror augmentation in 01_dataset_preparation.ipynb instead.

### Offline Label-Aware Mirror Augmentation (Notebook)

**File:** `notebooks/01_dataset_preparation.ipynb` — new section "Label-Aware Horizontal Mirror Augmentation"

The dataset notebook now produces a mirror-augmented NPZ as an explicit offline step:

- Every `swipe_left` sequence is x-flipped and re-labelled as `swipe_right`, and
  vice versa, producing a corresponding doubled count for both classes.
- The augmented NPZ is saved alongside the base dataset and can be passed to the
  training notebook as the dataset path.
- Two verification plots are generated: (1) a before/after class distribution bar
  chart, and (2) a wrist-trajectory x-trace comparison confirming sign inversion
  between original and mirrored samples.

### NPZ Loader — Tolerate Missing Optional Fields

**File:** `src/gesture_control/training/dataset_builder.py`

`SequenceDatasetBuilder.load_npz()` now checks for the presence of `sample_ids`,
`start_timestamp_ms`, and `end_timestamp_ms` before accessing them, falling back to
empty / zero arrays when absent. These fields are not written by the notebook's
offline augmentation step, and their absence previously caused a `KeyError` when the
augmented NPZ was passed to the training pipeline.

### Registry Infrastructure — Relative Path Storage and Resolution

**File:** `src/gesture_control/training/export_checkpoint.py`

`update_model_registry()` now stores `checkpoint_path` and `exported_path` as paths
relative to the registry file's parent directory (using `Path.relative_to()`, falling
back to the absolute path if the file is on a different volume). This makes the registry
portable across machines.

**File:** `src/gesture_control/infrastructure/model/checkpoint_loader.py`

`load_latest_inference_artifact()` resolves relative paths against the registry file's
own parent directory (corrected from `registry_path.parent.parent`, which pointed one
level too high).

---

## Testing and Validation

All tests were updated to the 147-dim feature vector:

| Test file | Change |
|---|---|
| `tests/unit/test_feature_engineering.py` | Zero-vector assertions updated to 147 dims |
| `tests/unit/test_augmentation.py` | Fixtures rebuilt for 147-dim tensors; `_X_COORD_DIMS` count updated |
| `tests/unit/test_dataset_builder.py` | Schema version updated to `feature-schema-v5` |
| `tests/unit/test_training_datamodule.py` | Feature dim assertion updated to 147 |
| `tests/integration/test_phase2_pipeline.py` | Forward-pass input dim updated to 147 |
| `tests/integration/test_phase3_feature_pipeline.py` | Feature dim assertion updated |

| Metric | Result |
|---|---|
| `pytest tests/` | **203 passed, 0 failed** |
| `ruff check src/ tests/` | **All checks passed** |
| `ruff check notebooks/` | **All checks passed** |

### Experiment Artefacts Registered

Three LSTM runs on `feature-schema-v5` registered in `models/registry.json`:

| Artefact ID | Schema |
|---|---|
| `lstm_v1_20260418T002927Z` | feature-schema-v5 |
| `lstm_v1_20260418T004149Z` | feature-schema-v5 |
| `lstm_v1_20260418T023120Z` | feature-schema-v5 |

---

## Outcome

The `swipe_left` / `swipe_right` confusion is addressed at its root: the model now
receives an explicit per-frame translational signal (camera-space wrist velocity) for
every frame, and mirror augmentation is applied offline with correct label remapping
rather than on-the-fly with a static label. The registry is portable across machines.
The NPZ loader is backward-compatible with derived datasets that omit optional fields.
