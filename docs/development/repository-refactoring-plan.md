# Repository Refactoring Plan

> **Date:** 2026-04-18  
> **Branch:** `build/model-improvement`  
> **Status:** Pending execution

---

## 1 — Documentation Audit

### 1.1 Classification Table

| Document                                                      | Category              | Action      | Reason                                                                                          |
| ------------------------------------------------------------- | --------------------- | ----------- | ----------------------------------------------------------------------------------------------- |
| `docs/architecture/system-overview.md`                        | Authoritative         | **Update**  | References 126-dim feature vector and old pipeline description                                  |
| `docs/architecture/data-flow.md`                              | Authoritative         | **Update**  | Stage 4 output `length 126`, Stage 6 tensor `(1, 24, 126)` — both stale                         |
| `docs/architecture/development-view.md`                       | Authoritative         | **Update**  | Feature dim references are indirect but stale                                                   |
| `docs/architecture/logical-view.md`                           | Authoritative         | **Keep**    | Component interfaces are implementation-agnostic                                                |
| `docs/architecture/process-view.md`                           | Authoritative         | **Keep**    | State machine logic unchanged since gaze integration                                            |
| `docs/architecture/physical-view.md`                          | Authoritative         | **Keep**    | Deployment topology unchanged                                                                   |
| `docs/architecture/gaze-aware-activation.md`                  | Historical design doc | **Archive** | Feature fully implemented; content is now covered by `system-overview.md` and `process-view.md` |
| `docs/architecture/model-improvement-analysis.md`             | Historical research   | **Archive** | Pre-v5 analysis on 126-dim schema; all findings have been acted on                              |
| `docs/debugging/tcn-quality-gap-investigation.md`             | Historical debug log  | **Archive** | TCN evaluated and rejected; move to `docs/archive/research/`                                    |
| `docs/debugging/tcn-performance-analysis-report.md`           | Historical debug log  | **Archive** | TCN evaluation record                                                                           |
| `docs/debugging/tcn-runtime-bug-investigation.md`             | Historical debug log  | **Archive** | TCN runtime bug; resolved                                                                       |
| `docs/development/ml-workflow-refactoring-plan.md`            | Engineering plan      | **Archive** | All deliverables complete                                                                       |
| `docs/implementation/gesture-recognition-enhancement-plan.md` | Engineering plan      | **Archive** | Superseded by feature-schema-v5; baseline metrics are stale (78.8% on 126-dim)                  |
| `docs/work-logs/` (40 files)                                  | Development logs      | **Archive** | Session logs from the development sprint; move to `docs/archive/work-logs/`                     |
| `docs/IPN_INTEGRATION_GUIDE.md`                               | Operational guide     | **Keep**    | Dataset pipeline from IPN source videos; still fully relevant                                   |
| `README.md`                                                   | Authoritative         | **Update**  | References 126-dim in pipeline summary; add HF model link                                       |

### 1.2 Critical Stale References

| Stale statement                           | Correct value                                                                             | Found in                                          |
| ----------------------------------------- | ----------------------------------------------------------------------------------------- | ------------------------------------------------- |
| Feature vector: `126-dimensional`         | `147-dimensional (feature-schema-v5)`                                                     | `system-overview.md`, `data-flow.md`, `README.md` |
| Input tensor: `(1, 24, 126)`              | `(1, 24, 147)`                                                                            | `data-flow.md` Stage 6                            |
| Stage 4 output `length 126`               | `length 147`                                                                              | `data-flow.md` Stage 4                            |
| Feature blocks: positions + velocity only | positions + fingertip spread + wrist trajectory + velocity + joint angles + wrist_vel_raw | `data-flow.md`                                    |

---

## 2 — Proposed Documentation Structure

```
docs/
├── architecture/              # authoritative system description (keep + update)
│   ├── system-overview.md
│   ├── data-flow.md
│   ├── logical-view.md
│   ├── process-view.md
│   ├── development-view.md
│   └── physical-view.md
│
├── ml-system/                 # new: ML-specific reference docs
│   ├── feature-schema-v5.md   # canonical 147-dim schema spec
│   ├── dataset-pipeline.md    # IPN source → build_ipn_dataset.py → NPZ spec
│   ├── training-guide.md      # how to retrain end-to-end from notebooks
│   └── model-versioning.md    # HF repo layout, revision policy, artifact format
│
├── operations/                # new: how to run the system
│   ├── quickstart.md          # install + run in 5 commands
│   ├── configuration.md       # YAML config reference (all configs/ keys)
│   └── model-loading.md       # HF vs local loading, cache location
│
├── development/               # coding standards, contributing guide
│   └── contributing.md
│
├── IPN_INTEGRATION_GUIDE.md   # keep at top level
│
└── archive/
    ├── work-logs/             # all 40 session logs
    ├── research/              # tcn-*.md, model-improvement-analysis.md,
    │                          # gesture-recognition-enhancement-plan.md
    └── plans/                 # ml-workflow-refactoring-plan.md,
                               # gaze-aware-activation.md (design doc)
```

---

## 3 — Hugging Face Model Loading Architecture

### 3.1 Target Config Schema

Add a `model_source` block to `configs/model/lstm_default.yaml`:

```yaml
model_source:
  backend: huggingface # or "local" for development/offline
  repo_id: ntsrigaud/gesture-control-lstm
  revision: main # or a pinned git tag, e.g. "v1.0.0"
  cache_dir: null # null = HF default (~/.cache/huggingface)
  filename: lstm_v1_inference.pt
```

The `backend: local` path retains the existing registry flow so notebook-based training continues to work unmodified during migration.

### 3.2 New Loader: `hf_loader.py`

```python
# src/gesture_control/infrastructure/model/hf_loader.py
from __future__ import annotations
from pathlib import Path
import torch
from huggingface_hub import hf_hub_download
from gesture_control.infrastructure.model.checkpoint_loader import (
    InferenceArtifact,
    load_inference_artifact,
)

def load_artifact_from_hub(
    *,
    repo_id: str,
    filename: str,
    revision: str = "main",
    cache_dir: str | Path | None = None,
    device: torch.device,
) -> InferenceArtifact:
    """Download (or return cached) an inference artifact from Hugging Face Hub."""
    local_path = hf_hub_download(
        repo_id=repo_id,
        filename=filename,
        revision=revision,
        cache_dir=str(cache_dir) if cache_dir else None,
    )
    return load_inference_artifact(artifact_path=Path(local_path), device=device)
```

### 3.3 Bootstrap Dispatch

```python
def _load_model(cfg: ModelSourceConfig, device: torch.device) -> InferenceArtifact:
    if cfg.backend == "huggingface":
        from gesture_control.infrastructure.model.hf_loader import load_artifact_from_hub
        return load_artifact_from_hub(
            repo_id=cfg.repo_id,
            filename=cfg.filename,
            revision=cfg.revision,
            cache_dir=cfg.cache_dir,
            device=device,
        )
    return load_latest_inference_artifact(registry_path=cfg.registry_path, device=device)
```

### 3.4 HF Repository Artifact Layout

```
ntsrigaud/gesture-control-lstm/
├── README.md                  # model card (generated by hf_publish.py)
├── lstm_v1_inference.pt       # inference artifact (state dict + metadata)
├── config.json                # model hyperparameters
├── feature_schema.json        # 147-dim schema spec
├── label_map.json             # integer → class label mapping
└── metrics.json               # test accuracy, per-class recall, F1
```

### 3.5 Model Versioning Policy

| Mechanism                                             | Use                                                |
| ----------------------------------------------------- | -------------------------------------------------- |
| HF `revision=main`                                    | Points to the latest production model              |
| HF git tags (e.g. `v1.0.0`)                           | Pinned reproducible references                     |
| Git tags in this repo (e.g. `model/lstm-v5-20260418`) | Cross-references HF revision for full auditability |

---

## 4 — Artifact Registry Removal Plan

| Item                                                         | Action                                                                                     |
| ------------------------------------------------------------ | ------------------------------------------------------------------------------------------ |
| `models/exported/` (19 `.pt` files)                          | Delete all but the current production artifact; add `models/exported/*.pt` to `.gitignore` |
| `models/checkpoints/` (empty)                                | Remove directory; add `models/checkpoints/*.pt` to `.gitignore`                            |
| `models/registry.json`                                       | Keep during migration; delete after `backend: huggingface` is validated end-to-end         |
| `update_model_registry()` in `export_checkpoint.py`          | Gate behind `backend == "local"` check; remove once local backend is retired               |
| `--registry` CLI argument in `main.py`                       | Deprecate with a warning; remove in the following release                                  |
| `load_latest_inference_artifact()` in `checkpoint_loader.py` | Mark `@deprecated`; remove once all callers have migrated                                  |

**Migration sequence:**

1. Add `hf_loader.py` and update `bootstrap.py` — `backend: local` remains default (no behaviour change).
2. Validate `backend: huggingface` end-to-end with `run_realtime.py`.
3. Switch `lstm_default.yaml` default to `backend: huggingface`.
4. Update `main.py` to remove `--registry`; add optional `--hf-revision`.
5. Delete artifact files; add entries to `.gitignore`.
6. Remove `update_model_registry()` call from `train_lstm.py`.
7. Delete `load_latest_inference_artifact()` and registry infrastructure.

---

## 5 — Notebook Upload Workflow

### 5.1 Staging Cell

```python
import json
STAGE_DIR = REPO_ROOT / "models" / "stage" / MODEL_VERSION
STAGE_DIR.mkdir(parents=True, exist_ok=True)

(STAGE_DIR / "config.json").write_text(json.dumps({
    "model_name": config.model.model_name,
    "input_size": config.model.input_size,
    "hidden_size": config.model.hidden_size,
    "num_layers": config.model.num_layers,
    "dropout": config.model.dropout,
    "num_classes": config.model.num_classes,
    "feature_schema_version": config.export.feature_schema_version,
}, indent=2))

(STAGE_DIR / "feature_schema.json").write_text(json.dumps({
    "version": "feature-schema-v5",
    "feature_dim": 147,
    "blocks": {
        "position":         [0, 63],
        "fingertip_spread": [63, 68],
        "wrist_trajectory": [68, 71],
        "velocity":         [71, 134],
        "joint_angles":     [134, 144],
        "wrist_vel_raw":    [144, 147],
    },
}, indent=2))

(STAGE_DIR / "label_map.json").write_text(
    json.dumps({str(i): lbl for i, lbl in enumerate(result.class_labels)}, indent=2)
)

(STAGE_DIR / "metrics.json").write_text(json.dumps({
    "test_accuracy": test_report.accuracy,
    "test_macro_f1": test_report.macro_f1,
    "per_class_recall": test_report.per_class_recall,
    "model_version": MODEL_VERSION,
    "feature_schema_version": "feature-schema-v5",
}, indent=2))
```

### 5.2 Upload Cell

```python
from huggingface_hub import HfApi
from gesture_control.training.hf_publish import build_model_card

(STAGE_DIR / "README.md").write_text(
    build_model_card(result=result, test_report=test_report, config=config)
)

api = HfApi()
url = api.upload_folder(
    repo_id=HF_REPO_ID,
    folder_path=str(STAGE_DIR),
    repo_type="model",
    commit_message=f"Upload {MODEL_VERSION} — feature-schema-v5",
    token=HF_TOKEN,
)
print(f"Published: {url}")
```

`upload_folder()` is diff-based — only changed files are transferred. `build_model_card()` is a public wrapper to add to `hf_publish.py`.

---

## 6 — Implementation Roadmap

### Phase 1 — Documentation Cleanup _(no code changes)_

| Step | Action                                                                            | Target                                               |
| ---- | --------------------------------------------------------------------------------- | ---------------------------------------------------- |
| 1.1  | Create `docs/archive/work-logs/`, `docs/archive/research/`, `docs/archive/plans/` | —                                                    |
| 1.2  | `git mv` 40 work-log files                                                        | `docs/work-logs/` → `docs/archive/work-logs/`        |
| 1.3  | `git mv` debugging + plan files                                                   | → `docs/archive/research/` and `docs/archive/plans/` |
| 1.4  | `git mv` `gaze-aware-activation.md`, `model-improvement-analysis.md`              | → `docs/archive/research/`                           |
| 1.5  | Update `data-flow.md` and `system-overview.md`                                    | Fix 147-dim, update Stage 4/6 tables                 |
| 1.6  | Update `README.md`                                                                | Fix 126-dim reference, add HF model link             |
| 1.7  | Create `docs/ml-system/feature-schema-v5.md`                                      | Extract schema table from `hf_publish.py`            |

### Phase 2 — Model Storage Migration _(runtime behaviour change)_

| Step | Action                                                                                        | Target                                      |
| ---- | --------------------------------------------------------------------------------------------- | ------------------------------------------- |
| 2.1  | Add `model_source` block to `lstm_default.yaml` with `backend: local`                         | `configs/model/lstm_default.yaml`           |
| 2.2  | Add `hf_loader.py`                                                                            | `src/gesture_control/infrastructure/model/` |
| 2.3  | Update `bootstrap.py` to dispatch on `backend`                                                | `src/gesture_control/app/bootstrap.py`      |
| 2.4  | Add `huggingface_hub` to runtime dependencies                                                 | `pyproject.toml`                            |
| 2.5  | Validate `backend: huggingface` end-to-end                                                    | integration test                            |
| 2.6  | Switch default to `backend: huggingface`                                                      | `configs/model/lstm_default.yaml`           |
| 2.7  | Add `models/exported/*.pt`, `models/checkpoints/*.pt`, `models/registry.json` to `.gitignore` | `.gitignore`                                |
| 2.8  | Delete 18 obsolete `.pt` artifacts                                                            | `models/exported/`                          |

### Phase 3 — Training Pipeline _(additive)_

| Step | Action                                                  | Target                                       |
| ---- | ------------------------------------------------------- | -------------------------------------------- |
| 3.1  | Add `metrics.json` export cell after evaluation         | `notebooks/02_model_training.ipynb`          |
| 3.2  | Add staging directory cell                              | `notebooks/02_model_training.ipynb`          |
| 3.3  | Replace HF publish cell with `upload_folder()` approach | `notebooks/02_model_training.ipynb`          |
| 3.4  | Expose `build_model_card()` as public function          | `src/gesture_control/training/hf_publish.py` |
| 3.5  | Add `models/stage/` to `.gitignore`                     | `.gitignore`                                 |

### Phase 4 — Runtime Refactor

| Step | Action                                                                            | Target                                       |
| ---- | --------------------------------------------------------------------------------- | -------------------------------------------- |
| 4.1  | Deprecate `load_latest_inference_artifact()`                                      | `checkpoint_loader.py`                       |
| 4.2  | Remove `--registry`; add `--hf-revision`                                          | `src/gesture_control/app/main.py`            |
| 4.3  | Remove `update_model_registry()` call                                             | `src/gesture_control/training/train_lstm.py` |
| 4.4  | Add unit test for `hf_loader.py` (mock `hf_hub_download`)                         | `tests/unit/test_hf_loader.py`               |
| 4.5  | Update integration test for bootstrap with `backend: huggingface` offline fixture | `tests/integration/`                         |

### Phase 5 — Repository Hygiene

| Step | Action                                                                        | Target                                      |
| ---- | ----------------------------------------------------------------------------- | ------------------------------------------- |
| 5.1  | Add `docs/development/contributing.md`                                        | New file                                    |
| 5.2  | Update `README.md` with new directory layout and HF model badge               | `README.md`                                 |
| 5.3  | Delete `configs/model/tcn_default.yaml`                                       | `configs/model/`                            |
| 5.4  | Evaluate `src/gesture_control/infrastructure/model/tcn_model.py` for deletion | `src/gesture_control/infrastructure/model/` |
| 5.5  | Tag current production commit: `git tag model/lstm-v5-20260418`               | Git                                         |
| 5.6  | Open PR from `build/model-improvement` → `main`                               | GitHub                                      |

### Priority Summary

| Phase                 | Risk   | Effort | Recommended order    |
| --------------------- | ------ | ------ | -------------------- |
| 1 — Documentation     | None   | Low    | First                |
| 3 — Training pipeline | Low    | Low    | Second               |
| 2 — Model migration   | Medium | Medium | Third (with Phase 4) |
| 4 — Runtime refactor  | Medium | Low    | Third (with Phase 2) |
| 5 — Hygiene           | None   | Low    | Last                 |
