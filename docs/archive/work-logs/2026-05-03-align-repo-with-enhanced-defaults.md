# Work Log — Repository Alignment to EnhancedTwoStreamLSTM Defaults

**Date:** 2026-05-03  
**Branch:** `refactor/align-repo-with-new-model`  
**Author:** ntsrigaud

---

## Context

After introducing `EnhancedTwoStreamLSTM`, the repository still contained mixed defaults, mixed runtime expectations, and stale architecture/training references pointing to legacy `two_stream_lstm` artifacts. This branch aligns runtime loading, training defaults, tests, and architecture documentation around a single production authority:

- Runtime model: `EnhancedTwoStreamLSTM`
- Runtime artifact type: `two_stream_attn_inference`
- Default model source artifact: `two_stream_attn_v1_20260503T065204Z_inference.pt`

In parallel, documentation was expanded to include the full May 2026 ablation/training series and a dedicated feature-schema specification for the enhanced two-stream partitioning rationale.

---

## Scope of Work

### Default model source alignment

Set repository-level and packaged default `model_source.yaml` to the new HF artifact filename.

### Training default alignment

Switch CLI training default config from legacy `two_stream_lstm.yaml` to `two_stream_attn.yaml`.

### Runtime loader hardening

Restrict runtime checkpoint loading to the enhanced artifact type only (`two_stream_attn_inference`) and remove legacy baseline runtime dispatch.

### Test suite migration

Update integration/unit tests to construct and load `EnhancedTwoStreamLSTM` artifacts, including `num_heads` and `proj_dim` config fields, and explicitly reject legacy runtime artifact types.

### Architecture and ML documentation refresh

Synchronize all architectural views to the enhanced two-stream runtime path, add dedicated model architecture documentation, add a formal feature-schema-v5 document, and append May 2026 cross-family training history/ablation summaries.

### Formatting sweep

Apply repository-wide style normalization (Prettier-driven formatting commit) across documentation and experiment artifacts.

---

## Technical Changes

### `configs/model_source.yaml`

### `src/maestro/_defaults/configs/model_source.yaml`

- Default HF artifact filename changed:
  - from `two_stream_lstm_v1_20260420T091839Z_inference.pt`
  - to `two_stream_attn_v1_20260503T065204Z_inference.pt`
- Ensures runtime/model bootstrap pull the enhanced model by default.

### `scripts/train_model.py`

- Default config path changed to `configs/model/two_stream_attn.yaml` in:
  - `_resolve_config_path()` fallback
  - CLI `--config` default
  - user-facing FileNotFoundError hint text
- Training entrypoint now defaults to enhanced architecture without requiring explicit CLI override.

### `src/maestro/utils/schema_validation.py`

- `ModelSourceConfig.filename` default updated to `two_stream_attn_v1_20260503T065204Z_inference.pt`.
- Aligns schema-level defaults with config-level defaults and runtime behavior.

### `src/maestro/infrastructure/model/checkpoint_loader.py`

- Runtime loader scope narrowed to enhanced artifacts:
  - keeps `two_stream_attn_inference` dispatch → `EnhancedTwoStreamLSTM`
  - removes legacy `two_stream_lstm_inference` runtime dispatch path
- Error messaging updated to single supported runtime artifact type.
- Makes runtime compatibility failures explicit instead of silently accepting legacy formats.

### `src/maestro/app/bootstrap.py`

- Loader docstring updated to reflect enhanced artifact as runtime source of truth.

### `tests/integration/test_phase2_pipeline.py`

- Mock artifact model switched from `TwoStreamGestureLSTM` to `EnhancedTwoStreamLSTM`.
- Test artifact constructor now includes enhanced-only config fields (`num_heads`, `proj_dim`).

### `tests/unit/test_checkpoint_loader.py`

- Artifact fixture switched to `two_stream_attn_inference` + `two_stream_attn_v1`.
- Positive-path assertions updated to enhanced model versioning.
- Negative-path test updated to reject legacy `two_stream_lstm_inference` artifacts.

### `tests/unit/test_hf_loader.py`

- HF artifact fixture switched from legacy to enhanced architecture.
- Artifact filename and payload model config aligned to enhanced defaults.

### `docs/architecture/enhanced-two-stream-lstm-architecture.md` (new)

- New authoritative architecture document:
  - input/output contract
  - end-to-end mermaid forward graph
  - component decomposition and rationale
  - runtime compatibility rules
  - training/runtime default references

### `docs/ml-system/feature-schema.md` (new)

- New formal `feature-schema-v5` specification:
  - 147-dim block-by-block schema
  - preprocessing steps and equations
  - invalid-frame handling and schema versioning
  - enhanced two-stream partition section (`73/74`) with implementation snippet and rationale

### `docs/ml-system/training-history-summary.md`

- Reframed as legacy snapshot + expanded May 2026 experimental appendix.
- Added explicit runtime authority note pointing to enhanced artifact defaults.
- Appended representative runs across model families:
  - `stgcn_lstm`
  - `dual_stgcn_lstm`
  - `two_stream_lstm`
  - `two_stream_attn`
- Includes per-run split metrics, per-class precision/recall, and observation summaries.
- Documents best validation Macro F1 in this series as `two_stream_attn_v1_20260503T064347Z` (~80.82%).

### Architecture view alignment

Updated architecture views to consistently reference the enhanced model and revised flow semantics:

- `docs/architecture/system-overview.md`
  - runtime model text switched to `EnhancedTwoStreamLSTM`
  - production 8-class runtime vocabulary vs full 14-class baseline vocabulary clarified
  - pipeline diagram updated with explicit gates (`gaze_permitted?`, `window_ready?`, `command_emitted?`)

- `docs/architecture/data-flow.md`
  - stage 7 renamed to enhanced inference
  - old two-stream-LSTM block replaced with enhanced model description
  - runtime artifact compatibility text switched to `two_stream_attn_inference`

- `docs/architecture/logical-view.md`
  - infrastructure model component changed to `EnhancedTwoStreamLSTM`
  - pipeline contract corrected to include explicit `HandSelector.select()` step (`MultiHandPacket` → `LandmarkPacket`)

- `docs/architecture/process-view.md`
  - inference stage and training loop labels updated (EnhancedTwoStream + AdamW)

- `docs/architecture/physical-view.md`
  - default runtime artifact reference updated to enhanced HF checkpoint
  - training command example default config switched to `two_stream_attn.yaml`

- `docs/architecture/development-view.md`
  - default model config declared as `two_stream_attn.yaml`
  - vocabulary description updated to full IPN 14-class baseline for dataset workflows

- `docs/architecture/training-pipeline.md`
  - production artifact reference switched to `two_stream_attn_v1`
  - IPN mapping block updated to full 14-class baseline mapping

### Formatting commit (`f063f88`)

- Repository-wide style normalization applied, including architecture docs, work logs, and committed experiment artifacts.
- No algorithmic behavior change introduced by this formatting-only commit.

---

## Testing and Validation

- Unit loader tests validated enhanced artifact round-trip expectations.
- Legacy runtime artifact type (`two_stream_lstm_inference`) is now intentionally rejected by loader tests.
- Integration phase-2 pipeline fixture updated to enhanced model instantiation and remains contract-compatible with runtime path.
- Documentation consistency validated by cross-references between:
  - `enhanced-two-stream-lstm-architecture.md`
  - `feature-schema.md`
  - architecture view set (`system-overview`, `data-flow`, `logical`, `process`, `physical`, `development`, `training-pipeline`).

---

## Outcome

The repository now has a single coherent default path for training and runtime inference centered on `EnhancedTwoStreamLSTM`. Config defaults, schema defaults, loader expectations, and tests are synchronized. Architecture documentation is now aligned with executable behavior, and May 2026 model-family ablation history is preserved in a single ML-system record for future model selection and regression analysis.
