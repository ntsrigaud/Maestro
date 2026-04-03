# Work Log - Feature Space Parity Hardening

## Context

This work resolves a critical training/inference feature-space inconsistency by centralizing feature preprocessing configuration, wiring runtime and dataset generation to the same config source, and enforcing compatibility validation before model usage.

## Scope of Work

- Added a centralized feature pipeline configuration file for preprocessing parity.
- Extended runtime configuration schema and loader to include feature pipeline settings.
- Updated runtime bootstrap to use config-driven orientation normalization.
- Added dataset compatibility validation utilities and bootstrap integration checks.
- Updated dataset generation script to resolve orientation normalization from config with explicit CLI override support.
- Added dedicated parity validation tests.
- Applied Black formatting normalization across touched modules.

## Technical Changes

- Added `configs/model/feature_pipeline.yaml` as the single source of truth for feature preprocessing controls.
- Extended schema models in `src/gesture_control/utils/schema_validation.py` with `FeaturePipelineConfig` and integrated it into `RuntimeConfig`.
- Updated `src/gesture_control/utils/config_loader.py` to load feature pipeline config from `configs/model/feature_pipeline.yaml` with safe defaults when absent.
- Updated `src/gesture_control/app/bootstrap.py` to construct `LandmarkFeatureTransformer` using `config.feature_pipeline.orientation_normalization`.
- Added `validate_container_dataset(...)` in `src/gesture_control/app/bootstrap.py` for pre-inference dataset compatibility checks.
- Added `src/gesture_control/utils/feature_validation.py` with metadata loading and feature-schema parity checks that raise explicit errors for critical mismatches.
- Updated `scripts/build_dataset.py` to resolve orientation normalization from config by default.
- Added explicit CLI override support in `scripts/build_dataset.py` via `--enable-orientation-normalization` and `--disable-orientation-normalization`.
- Added `tests/unit/test_feature_validation.py` covering metadata loading, schema mismatch detection, compatibility checks, and parity validation paths.

## Testing and Validation

- Executed `ruff check .` successfully.
- Executed `mypy src/gesture_control scripts/build_dataset.py scripts/collect_data.py` successfully.
- Executed `pytest -q` successfully.
- Added and validated new parity-specific unit tests in `tests/unit/test_feature_validation.py`.
- No test suppressions were added or removed.
- No dependency or build-system changes were introduced in this branch.

## Outcome

The branch restores deterministic training/inference feature-space parity by removing hardcoded preprocessing behavior and replacing it with validated, configuration-driven controls. Runtime initialization now has explicit safety checks against dataset incompatibility, reducing silent model-quality regressions. No breaking API changes were introduced.
