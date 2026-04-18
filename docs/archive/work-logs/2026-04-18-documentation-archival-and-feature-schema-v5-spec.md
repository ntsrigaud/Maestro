# Work Log — Documentation Archival and Feature Schema v5 Specification

**Date:** 2026-04-18  
**Branch:** `docs/documentation-archival`  
**Author:** @ntsrigaud

---

## Context

As the codebase reached `feature-schema-v5` and the swipe-confusion fix branch was
merged, the documentation tree had accumulated two distinct problems:

1. **Stale working documents scattered in live directories.** Historical work logs,
   research investigations, and completed planning documents were mixed into the same
   directories as actively maintained architecture references, making it difficult to
   distinguish canonical sources of truth from transient analysis.

2. **Architecture docs still reflected the 126-dimensional feature vector.** After the
   expansion to 147 dims in `feature-schema-v5`, `data-flow.md`, `system-overview.md`,
   and `README.md` still cited the old numbers and the old two-block feature layout.
   No single canonical specification for `feature-schema-v5` existed anywhere in the
   repository.

This branch performs a combined archival pass and documentation update to address both
issues.

---

## Scope of Work

| Area | Change type |
|---|---|
| 40 historical work logs | Move → `docs/archive/work-logs/` |
| 4 research investigation reports | Move → `docs/archive/research/` |
| 3 completed development plans | Move → `docs/archive/plans/` |
| `docs/architecture/data-flow.md` | Update — 126 → 147 dims; six-block v5 layout |
| `docs/architecture/system-overview.md` | Update — feature engineering description; HF Hub model export |
| `README.md` | Update — architecture reference; models directory description |
| `docs/ml-system/feature-schema-v5.md` | New — canonical ML feature specification |
| `docs/development/repository-refactoring-plan.md` | New — repository refactoring and model migration plan |

---

## Technical Changes

### Archive: Historical Work Logs (40 files)

All work logs from 2026-03-16 through 2026-04-18 moved from `docs/work-logs/` to
`docs/archive/work-logs/`. File content is unchanged; this is a structural move only.

### Archive: Research and Investigation Reports (4 files)

Moved to `docs/archive/research/`:

- `model-improvement-analysis.md`
- `tcn-performance-analysis-report.md`
- `tcn-quality-gap-investigation.md`
- `tcn-runtime-bug-investigation.md`

These documents captured the TCN evaluation and debugging work that led to the final
LSTM-based architecture. They are preserved as historical record, not as active guidance.

### Archive: Completed Development Plans (3 files)

Moved to `docs/archive/plans/`:

- `gaze-aware-activation.md`
- `gesture-recognition-enhancement-plan.md`
- `ml-workflow-refactoring-plan.md`

All three plans have been fully implemented and superseded by the current architecture.

### Architecture Docs Updated to Feature-Schema-v5

**File:** `docs/architecture/data-flow.md`

- Feature Engineering node: `126-d` → `147-d`
- `FeatureFrame` data table: vector length `126` → `147`
- Temporal Buffer data table: `24×126` → `24×147`
- LSTM input section: tensor shape updated from `(1, 24, 126)` to `(1, 24, 147)`;
  `input_size` updated from `126` to `147`
- Feature vector layout in prose replaced with the canonical six-block diagram
  matching `feature-schema-v5`:

```
[x₀…z₂₀]                        ← 63 dims  (positions)
[d₀₁, d₀₂, d₀₃, d₀₄, d₀₅]      ←  5 dims  (fingertip spread)
[Δwx, Δwy, Δwz]                 ←  3 dims  (wrist trajectory in window)
[vx₀…vz₂₀]                      ← 63 dims  (velocity)
[θ₀…θ₉]                         ← 10 dims  (MCP + PIP joint angles)
[vwx, vwy, vwz]                 ←  3 dims  (wrist velocity, camera-normalised)
                                 ─────────
                                   147 dims total (feature-schema-v5)
```

**File:** `docs/architecture/system-overview.md`

- Feature Engineering step (step 4): description updated to list all six feature
  types and reference `feature-schema-v5`.
- LSTM Inference step (step 6): sequence shape updated to `(24×147)`.
- Model Export step (step 3): description updated to reference the Hugging Face Hub
  (`ntsrigaud/gesture-lstm-v5`) as the publication target and describe the
  download-on-first-use runtime behaviour.

**File:** `README.md`

- Architecture reference: `ARCHITECTURE_SPECIFICATION.md` → `docs/architecture/`
- Models directory description updated to note local artifacts are `.gitignore`d and
  the production model is hosted on
  [Hugging Face](https://huggingface.co/ntsrigaud/gesture-lstm-v5).

### New: Canonical Feature Schema v5 Specification

**File:** `docs/ml-system/feature-schema-v5.md`

The first canonical reference document for the 147-dimensional feature schema. Covers:

- Feature block table with slice indices, dimensionality, and descriptions
- Step-by-step preprocessing pipeline with mathematical notation (wrist-relative
  centering, scale normalisation, positions, fingertip spread, wrist trajectory,
  velocity, joint angles, wrist velocity raw)
- Orientation normalisation rationale (disabled in production)
- Invalid frame handling policy
- Schema version history table (`feature-schema-v1` through `feature-schema-v5`)

### New: Repository Refactoring Plan

**File:** `docs/development/repository-refactoring-plan.md`

Outlines the planned repository reorganisation and model migration work, including
Hugging Face Hub publication strategy, directory restructuring for the archive
hierarchy, and naming conventions for future model registry entries.

---

## Outcome

The repository documentation is now in a consistent state with the deployed codebase.
The archive hierarchy separates historical material from current references. All
architecture documents cite the correct 147-dimensional feature schema. Engineers
joining the project have a single canonical document (`docs/ml-system/feature-schema-v5.md`)
as the reference for the ML feature space, and a clear distinction between active
documentation and historical record.
