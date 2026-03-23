from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml

from gesture_control.utils.schema_validation import (
    AppConfig,
    FeaturePipelineConfig,
    InferenceConfig,
    RuntimeConfig,
)


def _read_yaml(path: Path) -> dict[str, Any]:
    if not path.exists():
        raise FileNotFoundError(f"Config file not found: {path}")

    with path.open("r", encoding="utf-8") as f:
        data = yaml.safe_load(f) or {}

    if not isinstance(data, dict):
        raise ValueError(f"Config file must contain a YAML mapping: {path}")
    return data


def load_runtime_config(config_dir: str | Path) -> RuntimeConfig:
    root = Path(config_dir)
    app_cfg = AppConfig.model_validate(_read_yaml(root / "app.yaml"))
    infer_cfg = InferenceConfig.model_validate(_read_yaml(root / "inference.yaml"))

    # Load feature pipeline config with defaults if file doesn't exist
    feature_pipeline_path = root / "model" / "feature_pipeline.yaml"
    if feature_pipeline_path.exists():
        feature_cfg = FeaturePipelineConfig.model_validate(
            _read_yaml(feature_pipeline_path)["feature_pipeline"]
        )
    else:
        feature_cfg = FeaturePipelineConfig()

    return RuntimeConfig(app=app_cfg, inference=infer_cfg, feature_pipeline=feature_cfg)
