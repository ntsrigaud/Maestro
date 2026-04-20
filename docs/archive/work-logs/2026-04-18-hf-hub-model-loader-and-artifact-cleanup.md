# Work Log — Hugging Face Hub Model Loader and Local Artifact Cleanup

**Date:** 2026-04-18  
**Branch:** `refactor/model-loader-and-local-artifact-cleanup`  
**Author:** @ntsrigaud

---

## Context

With `feature-schema-v5` stabilised and `lstm_v1_20260418T023120Z` confirmed as the
production artifact, the repository was still carrying 21 binary `.pt` files
(~22 MB) and a 372-line `registry.json` in version control. These artifacts had
already been published to the Hugging Face Hub; storing them in git was redundant and
prevented clean repository clones.

Separately, the runtime bootstrap had no abstraction for the artifact source: it
always loaded from the local registry, making it impossible to deploy a clone of the
repository that had never run a training job.

This branch removes all local artifacts from git, exposes a two-backend model source
configuration (`local` / `huggingface`), implements the Hugging Face loader, and
promotes `huggingface_hub` to a core runtime dependency.

---

## Scope of Work

| Area                              | Change type                                            |
| --------------------------------- | ------------------------------------------------------ |
| `ModelSourceConfig` schema        | New — Pydantic model for artifact source configuration |
| `configs/model_source.yaml`       | New — runtime configuration file for model source      |
| `RuntimeConfig`                   | Update — add `model_source` field                      |
| `load_runtime_config()`           | Update — load and validate `model_source.yaml`         |
| `hf_loader.py`                    | New — Hugging Face Hub inference artifact loader       |
| `bootstrap.py`                    | Refactor — dispatch on `model_source.backend`          |
| `pyproject.toml`                  | Update — promote `huggingface_hub` to core dependency  |
| `.gitignore`                      | Update — exclude local model artifacts                 |
| `models/exported/*.pt` (21 files) | Delete — removed from version control                  |
| `models/registry.json`            | Delete — removed from version control                  |

---

## Technical Changes

### Schema: `ModelSourceConfig`

**File:** `src/gesture_control/utils/schema_validation.py`

New Pydantic model (`extra="forbid"`) with two mutually exclusive configuration
groups:

```
backend: "local" | "huggingface"   # validated via regex pattern

# Hugging Face Hub (active when backend == "huggingface")
repo_id:    str   = "ntsrigaud/gesture-control-lstm"
revision:   str   = "main"
filename:   str   = "lstm_v1_inference.pt"
cache_dir:  str | None = None      # null → ~/.cache/huggingface/hub

# Local registry (active when backend == "local")
registry_path: str = "models/registry.json"
```

`RuntimeConfig` extended with `model_source: ModelSourceConfig = Field(default_factory=ModelSourceConfig)`. The `feature_pipeline` field was also converted from a bare default to `Field(default_factory=...)` for consistency.

### Config: `configs/model_source.yaml`

New file shipped with `backend: huggingface` and pinned to
`lstm_v1_20260418T023120Z_inference.pt` on revision `main`. The `local` backend path
is preserved in the file and documented; switching back requires only a one-line
change.

### Config Loader: `load_runtime_config()`

**File:** `src/gesture_control/utils/config_loader.py`

Loads `model_source.yaml` from the config root directory and passes the resulting
`ModelSourceConfig` into `RuntimeConfig`. When the file is absent, `ModelSourceConfig()`
defaults are used (`backend: local`, pointing to `models/registry.json`), which
preserves backward compatibility for configurations that predate this change.

### New: Hugging Face Hub Loader

**File:** `src/gesture_control/infrastructure/model/hf_loader.py`

`load_artifact_from_hub()` — downloads (or returns cached) an inference artifact from
a Hugging Face Hub repository:

- Calls `hf_hub_download()` with `repo_id`, `filename`, `revision`, and `cache_dir`.
- Delegates to `load_inference_artifact()` (the existing local loader) with the
  resolved local path, so the artifact unpacking, device placement, and
  `InferenceArtifact` construction are reused without duplication.
- After the first download the artifact is served from the HF cache; no network
  round-trip occurs on subsequent calls for the same `revision`.
- Guards against a missing `huggingface_hub` installation with a clear
  `ImportError` message.

### Bootstrap: Dispatch on `model_source.backend`

**File:** `src/gesture_control/app/bootstrap.py`

`build_container()` signature updated:

- `registry_path: str | None = None` (was `str = "models/registry.json"`)
- `hf_revision_override: str | None = None` (new — for CLI `--hf-revision` flag)

The direct `load_latest_inference_artifact()` call is replaced with a new private
helper `_load_model_from_source()` that routes based on `config.model_source.backend`:

```
backend == "huggingface"  →  load_artifact_from_hub(repo_id, filename,
                              revision=hf_revision_override or src.revision,
                              cache_dir, device)

backend == "local"        →  load_latest_inference_artifact(
                              registry_path=registry_path_override or src.registry_path,
                              device)
```

CLI-level overrides (`registry_path`, `hf_revision_override`) take precedence over
values in the config file, supporting manual testing and one-off deployments.

### Dependency: `huggingface_hub` Promoted to Core

**File:** `pyproject.toml`

`huggingface_hub>=0.22.0` moved from `[project.optional-dependencies].huggingface` to
`[project.dependencies]`. The `[huggingface]` optional extra is retained as an empty
group with a backwards-compatibility comment so existing install commands
(`pip install -e ".[huggingface]"`) continue to work without error.

### Gitignore: Exclude Local Artifacts

**File:** `.gitignore`

Four new patterns added under a `# Model artifacts` comment block:

```
models/exported/*.pt
models/checkpoints/*.pt
models/registry.json
models/stage/
```

### Deleted: Local Artifacts Removed from Version Control

21 binary `.pt` inference artifacts (18 LSTM, 2 TCN) and `models/registry.json` (372
lines) removed from the repository. All artifacts are preserved on Hugging Face Hub;
the registry is regenerated locally on each training run.

---

## Outcome

Fresh clones of the repository are ~22 MB lighter and do not require a local registry
or pre-downloaded weights to run the application. The `backend: huggingface` default
in `model_source.yaml` means `scripts/run_realtime.py` works out of the box on any
machine with network access. The `local` path is still fully functional for offline
development or when running against a freshly trained checkpoint.
