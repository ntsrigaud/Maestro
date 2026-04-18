# Work Log — Repository Cleanup: HF Hub Migration, Documentation Archival, and TCN Retirement

**Date:** 2026-04-18  
**Branch:** `refactor/repository-cleanup`  
**Author:** @ntsrigaud

---

## Context

This branch consolidates four sequential sub-PRs that together complete the
post-`feature-schema-v5` housekeeping pass on the repository. The goals were:

1. Archive all historical working documents so the live documentation tree contains
   only actively maintained references.
2. Update all architecture documentation to the 147-dimensional `feature-schema-v5`
   feature vector and add a canonical ML specification document.
3. Migrate the runtime model loader from the local artifact registry to the Hugging
   Face Hub, removing ~22 MB of tracked binary files in the process.
4. Complete CLI and deprecation hygiene so the local-registry code path is clearly
   marked for future removal and the training script does not attempt registry updates
   in Hub-backed workflows.
5. Remove the last TCN artefact (its training config) and fix a stale Hugging Face
   repository ID that had propagated across several files.

---

## Scope of Work (88 files, +1196 / −449)

| Sub-PR | Title | Key deliverables |
|---|---|---|
| #139 | Documentation archival and feature-schema-v5 rollout | 47 files moved to `docs/archive/`; `data-flow.md` + `system-overview.md` updated to 147-dim; `docs/ml-system/feature-schema-v5.md` created; `docs/work-logs/` directory removed |
| #140 | HF Hub model loader and local artifact cleanup | `ModelSourceConfig`, `configs/model_source.yaml`, `hf_loader.py`, bootstrap dispatch; 21 `.pt` files + `registry.json` deleted; `huggingface_hub` promoted to core dep |
| #141 | Model card usage snippet update | `hf_publish.py` usage snippet updated to HF Hub download |
| #142 | Runtime CLI deprecations and HF loader tests | `--hf-revision` flag; `--registry` deprecated; `load_latest_inference_artifact()` deprecation notice; `train_model.py` conditional registry update; 3 offline unit tests for `hf_loader.py` |
| #143 | TCN config cleanup and HF repo ID correction | `configs/model/tcn_default.yaml` deleted; `gesture-lstm-v5` → `gesture-control-lstm` across 4 files |

---

## Technical Changes

### Documentation Archival and Feature-Schema-v5 Update (PR #139)

**`docs/archive/` hierarchy established:**
- 41 historical work logs → `docs/archive/work-logs/`
- 4 research investigation reports → `docs/archive/research/`
- 3 completed development plans → `docs/archive/plans/`
- `docs/work-logs/` directory removed; all work logs henceforth written directly to `docs/archive/work-logs/`

**Architecture docs updated to 147-dim / `feature-schema-v5`:**
- `docs/architecture/data-flow.md`: feature node, `FeatureFrame` table, Temporal Buffer table, LSTM input tensor shape all updated; feature vector prose replaced with the canonical six-block layout diagram.
- `docs/architecture/system-overview.md`: Feature Engineering step and LSTM Inference step updated; Model Export step references the Hugging Face Hub.
- `README.md`: architecture pointer updated from `ARCHITECTURE_SPECIFICATION.md` → `docs/architecture/`; models directory description updated to reference HF Hub.

**New files:**
- `docs/ml-system/feature-schema-v5.md`: canonical 147-dim feature specification with block table, preprocessing pipeline with mathematical notation, orientation normalisation rationale, invalid frame policy, and schema version history.
- `docs/development/repository-refactoring-plan.md`: plan document covering HF Hub publication strategy, archive directory structure, and model registry naming conventions.

### HF Hub Model Loader (PR #140)

**`src/gesture_control/utils/schema_validation.py`** — new `ModelSourceConfig` Pydantic model (`extra="forbid"`):
- `backend: "local" | "huggingface"` (regex-validated)
- HF Hub fields: `repo_id`, `revision`, `filename`, `cache_dir`
- Local fields: `registry_path`
- `RuntimeConfig.model_source` field added (default factory)

**`configs/model_source.yaml`** — new file shipped with `backend: huggingface`, pinned to `lstm_v1_20260418T023120Z_inference.pt`.

**`src/gesture_control/utils/config_loader.py`** — `load_runtime_config()` loads `model_source.yaml`; falls back to `ModelSourceConfig()` defaults when absent (backwards-compatible).

**`src/gesture_control/infrastructure/model/hf_loader.py`** — new module:
- `load_artifact_from_hub()`: calls `hf_hub_download()` then delegates to `load_inference_artifact()` for unpacking and device placement.
- Guards against absent `huggingface_hub` with an informative `ImportError`.

**`src/gesture_control/app/bootstrap.py`** — `build_container()` updated:
- `registry_path: str | None = None`; `hf_revision_override: str | None = None` added.
- Direct `load_latest_inference_artifact()` call replaced with `_load_model_from_source()` routing on `config.model_source.backend`.

**`pyproject.toml`** — `huggingface_hub>=0.22.0` moved to `[project.dependencies]`; `[huggingface]` extra retained as empty for backwards compatibility.

**`.gitignore`** — `models/exported/*.pt`, `models/checkpoints/*.pt`, `models/registry.json`, `models/stage/` excluded.

**Deleted:** 21 inference `.pt` artifacts (~22 MB) and `models/registry.json` (372 lines).

### Model Card Usage Snippet (PR #141)

**`src/gesture_control/training/hf_publish.py`** — model card usage code snippet updated to show `hf_hub_download()` rather than the local registry path.

### CLI Deprecations and HF Loader Tests (PR #142)

**`src/gesture_control/app/main.py`:**
- `--hf-revision REVISION` flag added (forwarded to `build_container(hf_revision_override=...)`).
- `--registry` default changed to `None`; help text marks it as deprecated.
- `--model-artifact` help text updated to reference `model_source.yaml`.

**`src/gesture_control/infrastructure/model/checkpoint_loader.py`** — `load_latest_inference_artifact()` gains a `.. deprecated::` docstring pointing to `hf_loader.load_artifact_from_hub`.

**`scripts/train_model.py`** — `update_model_registry()` called conditionally: `... if args.registry else None`.

**`tests/unit/test_hf_loader.py`** — three offline unit tests using `unittest.mock.patch`:
- `test_load_artifact_from_hub_calls_hf_hub_download` — call args and returned artifact fields
- `test_load_artifact_from_hub_passes_cache_dir` — `cache_dir` and `revision` forwarded correctly
- `test_load_artifact_from_hub_raises_on_missing_huggingface_hub` — `ImportError` on absent library

### TCN Cleanup and Repo ID Fix (PR #143)

**`configs/model/tcn_default.yaml`** — deleted (50 lines; `feature-schema-v4`, `input_size: 144`).

**HF repo ID** `ntsrigaud/gesture-lstm-v5` → `ntsrigaud/gesture-control-lstm` in `README.md`, `docs/architecture/system-overview.md`, two prior work logs in `docs/archive/work-logs/`, and `docs/development/repository-refactoring-plan.md`.

---

## Outcome

- The repository clones ~22 MB lighter with no tracked binary model weights.
- `scripts/run_realtime.py` works on any machine with network access without a prior training run.
- All documentation references the correct 147-dimensional feature schema and the correct HF Hub repository.
- The archive hierarchy cleanly separates historical material from active references.
- The local registry code path is marked deprecated at every entry point, with a clear migration path, pending end-to-end `backend: huggingface` validation.
