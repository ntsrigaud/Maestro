from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch

from maestro.utils.schema_validation import RuntimeConfig

_logger = logging.getLogger(__name__)


def _build_model_from_artifact(
    model_config: dict[str, Any],
    artifact_type: str,
    device: torch.device,
) -> torch.nn.Module:
    """Instantiate the correct model class from *artifact_type* / *model_config*.

    Runtime supports only the production EnhancedTwoStreamLSTM artifact:
    * ``two_stream_attn_inference``
    """
    model_name = str(model_config.get("model_name", ""))

    if artifact_type == "two_stream_attn_inference" or model_name.startswith("two_stream_attn"):
        from maestro.infrastructure.model.two_stream_attn_lstm import EnhancedTwoStreamLSTM

        return EnhancedTwoStreamLSTM(
            input_size=int(model_config["input_size"]),
            hidden_size=int(model_config["hidden_size"]),
            num_layers=int(model_config["num_layers"]),
            dropout=float(model_config["dropout"]),
            num_classes=int(model_config["num_classes"]),
            num_heads=int(model_config.get("num_heads", 4)),
            proj_dim=int(model_config.get("proj_dim", 64)),
            motion_temporal_conv=bool(model_config.get("motion_temporal_conv", False)),
            motion_attn_pool=bool(model_config.get("motion_attn_pool", False)),
        ).to(device)

    raise ValueError(
        f"Unsupported artifact type {artifact_type!r} (model_name={model_name!r}). "
        "Supported type: 'two_stream_attn_inference'."
    )


@dataclass(slots=True)
class InferenceArtifact:
    model: torch.nn.Module
    model_version: str
    class_labels: list[str]
    artifact_path: Path
    # Flat-feature schema fields (two_stream_* models).
    feature_schema_version: str = "unknown"
    feature_dim: int = 0
    window_length: int = 16
    window_step: int = 3
    orientation_normalization: bool = False


def load_inference_artifact(
    *,
    artifact_path: str | Path,
    device: torch.device,
) -> InferenceArtifact:
    path = Path(artifact_path)
    if not path.exists():
        raise FileNotFoundError(f"Inference artifact not found: {path}")

    payload = torch.load(path, map_location=device, weights_only=False)
    if not isinstance(payload, dict):
        raise ValueError(f"Unexpected artifact payload format: {path}")

    model_config = payload.get("model_config")
    class_labels = payload.get("class_labels")
    state_dict = payload.get("model_state_dict")

    if not isinstance(model_config, dict):
        raise ValueError(f"Artifact missing model_config: {path}")
    if not isinstance(class_labels, list) or not class_labels:
        raise ValueError(f"Artifact missing class_labels: {path}")
    if state_dict is None:
        raise ValueError(f"Artifact missing model_state_dict: {path}")

    artifact_type = str(payload.get("artifact_type", ""))
    model = _build_model_from_artifact(model_config, artifact_type, device)
    model.load_state_dict(state_dict)
    model.eval()

    base = InferenceArtifact(
        model=model,
        model_version=str(payload.get("artifact_version", "unknown")),
        class_labels=[str(x) for x in class_labels],
        artifact_path=path,
    )

    _model_name_loaded = str(model_config.get("model_name", ""))

    fs = payload.get("feature_schema", {})
    if not isinstance(fs, dict):
        raise ValueError(f"Artifact missing feature_schema: {path}")
    base.feature_schema_version = str(fs.get("feature_schema_version", "unknown"))
    base.feature_dim = int(fs.get("feature_dim") or model_config.get("input_size", 0))
    base.window_length = int(fs.get("window_length", 16))
    base.window_step = int(fs.get("window_step", 3))
    base.orientation_normalization = bool(fs.get("orientation_normalization", False))

    return base


def validate_runtime_compatibility(
    *,
    runtime_config: RuntimeConfig,
    artifact: InferenceArtifact,
) -> None:
    if (
        runtime_config.feature_pipeline.orientation_normalization
        != artifact.orientation_normalization
    ):
        raise ValueError(
            "Feature schema mismatch: orientation_normalization differs between runtime "
            f"({runtime_config.feature_pipeline.orientation_normalization}) and model artifact "
            f"({artifact.orientation_normalization})."
        )

    if runtime_config.inference.window_length != artifact.window_length:
        raise ValueError(
            "Inference window_length mismatch: runtime "
            f"{runtime_config.inference.window_length} vs artifact {artifact.window_length}"
        )

    if runtime_config.inference.window_step != artifact.window_step:
        raise ValueError(
            "Inference window_step mismatch: runtime "
            f"{runtime_config.inference.window_step} vs artifact {artifact.window_step}"
        )
