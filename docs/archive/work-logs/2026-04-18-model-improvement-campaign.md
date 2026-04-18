# Work Log — Model Improvement Campaign: Feature Schema v5, TCN Evaluation, and HF Hub Deployment

**Date:** 2026-04-18  
**Branch:** `build/model-improvement`  
**Author:** @ntsrigaud

---

## Context

This branch contains the complete model improvement campaign, from the initial quality
analysis through to a production-deployed model on the Hugging Face Hub. The work
proceeded in six logical phases:

1. **Feature engineering and architecture exploration** — systematic expansion of the
   feature vector from the v1 baseline through three intermediate schemas while exploring
   a Temporal Convolutional Network architecture in parallel with the LSTM.
2. **ML workflow operationalisation** — extraction of dataset preparation and model
   training into reproducible, notebook-driven layers with Hugging Face Hub publishing.
3. **Feature schema v5 and horizontal swipe fix** — final 147-dimensional feature vector
   with global wrist velocity, label-aware mirror augmentation, and corrected swipe
   confusion.
4. **Repository cleanup and HF Hub migration** — documentation archival, promotion of
   `huggingface_hub` to a core dependency, removal of tracked binary artifacts from VCS,
   and TCN retirement.

**Total scope:** 144 files changed, 12 471 insertions(+), 194 deletions(-)

---

## Phase 1 — Feature Engineering and Architecture Exploration (direct commits)

### Enhancement Planning

Three planning documents authored before any implementation:

- `docs/archive/research/model-improvement-analysis.md` (323 lines) — technical review
  covering LSTM architecture limitations, IPN gesture class confusion matrix analysis,
  feature quality assessment, and a prioritised remediation plan.
- `docs/archive/plans/gesture-recognition-enhancement-plan.md` (936 lines) — full
  implementation plan for the model enhancement campaign, including ordered phases,
  deliverable acceptance criteria, and rollback checkpoints.

### Feature Vector Expansion — v1 → v2 (126-dim)

**`src/gesture_control/pipeline/feature_engineering.py`** — feature vector expanded from
the v1 baseline. Added: wrist trajectory delta over the buffer window (`Δwx, Δwy, Δwz`,
3 dims); fingertip spread distances (`d₀₁–d₀₅`, 5 dims). Total: 126 dims (`feature-schema-v2`).

**`configs/model/feature_pipeline.yaml`** — `feature_schema_version: v2`, dimension
counts and block names updated.

**`configs/model/lstm_default.yaml`** — `input_size` updated to match v2 dims.

**`src/gesture_control/training/augmentation.py`** — mirror flip augmentation added with
a configurable `mirror_flip_probability`; augmentation config centralised.

**`src/gesture_control/training/train_lstm.py`** — criterion construction centralised;
label smoothing added (`label_smoothing=0.1`).

**`src/gesture_control/training/export_checkpoint.py`** — model architecture metadata
(`model_type`, `input_size`, `hidden_size`, `num_layers`) included in exported artifact.

**`scripts/build_ipn_dataset.py`** — `--feature-schema-version` argument added;
feature schema version propagated into NPZ metadata.

### Joint Angle Features — v2 → v3 (144-dim)

**`src/gesture_control/pipeline/feature_engineering.py`** — ten joint angles extracted
from MCP and PIP landmark triplets (`θ₀–θ₉`, 10 dims). Per-joint landmark triplet
constants defined in a dedicated table. Total: 144 dims (`feature-schema-v3`).

### Temporal Convolutional Network (TCN) — Exploration and Retirement

**`src/gesture_control/infrastructure/model/tcn_model.py`** (120 lines) — TCN
architecture with dilated causal convolutions, residual connections, weight normalisation,
and a configurable channel depth and kernel size.

**`src/gesture_control/infrastructure/model/lstm_model.py`** — temporal attention added
to `GestureLSTMClassifier`: a learned query vector projects hidden states to attention
weights and scales the final hidden representation.

**`src/gesture_control/infrastructure/model/checkpoint_loader.py`** — multi-architecture
dispatch added; `load_inference_artifact()` reads `model_type` from the artifact metadata
and routes to the appropriate constructor.

**`configs/model/lstm_default.yaml`** — updated for v3/144-dim.

**Training runs:** two TCN runs and four LSTM runs committed to
`experiments/runs/` (JSON config snapshots, dataset metadata, epoch-level training
metrics).

**TCN investigation documents** (→ `docs/archive/research/`):

| File | Content |
|---|---|
| `tcn-runtime-bug-investigation.md` | Off-by-one error in wrist trajectory window; orientation normalisation interaction |
| `tcn-quality-gap-investigation.md` | Root-cause analysis: receptive field underfit on 30-frame sequences |
| `tcn-performance-analysis-report.md` | Benchmark comparison LSTM vs TCN; LSTM Macro-F1 0.94 vs TCN 0.87 |

**Decision:** TCN architecture retired; LSTM retained as the sole production model.
`configs/model/tcn_default.yaml` deleted in the final cleanup PR.

---

## Phase 2 — ML Workflow Operationalisation (PRs #131, #134, #135)

### PR #131 — Notebook-Based ML Workflow and HF Publishing

**`src/gesture_control/training/hf_publish.py`** (448 lines) — new Hugging Face Hub
publishing module:
- `publish_to_huggingface()`: two-stage commit upload — model card + `config.json`
  metadata snapshot first, binary inference artifact second.
- `build_model_card()`: renders a model card with YAML front-matter, gesture class
  description table, 147-dim feature schema breakdown, per-epoch training config, and
  per-class evaluation metrics.
- Soft dependency on `huggingface_hub`; raises a clear `ImportError` when absent.

**`notebooks/01_dataset_preparation.ipynb`** — Layer 1: dataset build, class
distribution visualization, per-split balance verification, recording-level leakage check.

**`notebooks/02_model_training.ipynb`** — Layer 2: forward-pass verification, training
loop with early stopping, learning-curve and confusion-matrix visualisation, inference
artifact export, and optional publishing via `publish_to_huggingface()`.

**`pyproject.toml`** — `notebooks` optional dependency group (`scikit-learn`,
`matplotlib`, `seaborn`, `pandas`, `jupyter`, `ipykernel`); `huggingface` optional group
(`huggingface_hub>=0.22.0`); `training` pytest marker added.

**`.github/workflows/ci.yml`** — Git LFS checkout enabled to unblock `ruff` lint step.

**`docs/archive/plans/ml-workflow-refactoring-plan.md`** — implementation plan (541
lines) documenting the three-layer architecture and per-phase deliverable status table.

### PR #134 — ML Workflow Execution Fixes

Three blockers discovered during end-to-end notebook execution, resolved by @blackbird410:

**`src/gesture_control/training/datamodule.py`** — `train_indices`, `val_indices`, and
`test_indices` exposed as `@property` accessors (with `RuntimeError` guard for
pre-`setup()` access). Notebooks use these to index raw NPZ arrays for split-level
analysis and leakage checking.

**`scripts/build_ipn_dataset.py`** — `--unknown-subsample-ratio FLOAT` flag added;
post-build filter removes excess `unknown`-class samples to reduce class imbalance.

**Notebooks** — `update_model_registry()` call corrected; split index properties wired
into leakage checks.

### PR #135 — v4 Training Artifacts

Eleven LSTM training runs from the notebook-based ML workflow committed to
`experiments/runs/` (JSON config snapshots, dataset metadata, per-epoch training
metrics). These constitute the `feature-schema-v4` training baseline.

### PR #136 — Registry Path Fix

**`src/gesture_control/infrastructure/model/checkpoint_loader.py`** — `load_latest_inference_artifact()` resolves artifact paths relative to the registry file directory rather than the CWD. Fixes a crash when the runtime process is started from any directory other than the repository root.

---

## Phase 3 — Feature Schema v5 (PR #138)

### Global Wrist Velocity — v4 → v5 (147-dim)

**`src/gesture_control/pipeline/feature_engineering.py`** — global wrist velocity in
camera-normalised space added as the final three features (`vwx, vwy, vwz`). Camera
normalisation makes the feature invariant to hand scale and position. Total: 147 dims (`feature-schema-v5`).

**`configs/model/feature_pipeline.yaml`** — `feature_schema_version: v5`, `input_dim: 147`.

**`configs/model/lstm_default.yaml`** — `input_size: 147`.

### Horizontal Swipe Confusion Fix

The `SWIPE_LEFT` / `SWIPE_RIGHT` gesture pair shared overlapping trajectory peaks in
the wrist trajectory sub-vector (3 dims, schema-v2 block). The fix:

- `src/gesture_control/training/augmentation.py` — mirror flip updated to flip the new
  wrist velocity x-component (`vwx`) in addition to the existing trajectory delta
  (`Δwx`); label-aware flip introduced — swipe class labels are toggled when the mirror
  is applied so the augmented sample has the correct label.
- `configs/model/feature_pipeline.yaml` — mirror flip axis list updated to include
  wrist velocity x-index (146).

**Four final v5 training runs** committed to `experiments/runs/`; best checkpoint
(`lstm_v1_20260418T023120Z`, Macro-F1 0.96) published to
`ntsrigaud/gesture-control-lstm` on Hugging Face Hub.

---

## Phase 4 — Repository Cleanup and HF Hub Migration (PR #144)

Full detail in the dedicated meta work log
`docs/archive/work-logs/2026-04-18-repository-cleanup-hf-migration-and-documentation-archival.md`.
Summary:

| Sub-PR | Key change |
|---|---|
| #139 | 47 historical docs → `docs/archive/`; `feature-schema-v5` spec published; `docs/work-logs/` removed |
| #140 | `ModelSourceConfig`, `hf_loader.py`, bootstrap dispatch; 21 `.pt` + `registry.json` deleted from VCS |
| #141 | Model card usage snippet updated to HF Hub download |
| #142 | `--hf-revision` CLI flag; `--registry` deprecated; 3 unit tests for HF loader |
| #143 | `configs/model/tcn_default.yaml` deleted; `gesture-lstm-v5` → `gesture-control-lstm` |

---

## Outcome

| Metric | Before | After |
|---|---|---|
| Feature schema | v1 baseline | `feature-schema-v5`, 147-dim |
| Model Macro-F1 (IPN test) | ~0.87 LSTM v1 (from analysis) | **0.96** LSTM v5 |
| Architecture | LSTM only | LSTM + TCN explored; TCN retired |
| Training workflow | `scripts/train_model.py` only | Notebook Layer 1 + Layer 2 + HF publishing |
| Model deployment | Local `models/` directory | Hugging Face Hub (`ntsrigaud/gesture-control-lstm`) |
| Tracked binary weight size | ~22 MB `.pt` files | 0 MB — all weights excluded from VCS |
| Architecture documentation | v1 feature schema | 147-dim, `feature-schema-v5` throughout |
