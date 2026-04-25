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
    """Instantiate :class:`TwoStreamGestureLSTM` from *artifact_type*.

    Raises:
        ValueError: If *artifact_type* does not identify a two-stream artifact.
            This prevents silent failure when an unexpected artifact is loaded.
    """
    model_name = str(model_config.get("model_name", ""))
    if not (artifact_type.startswith("two_stream") or model_name.startswith("two_stream")):
        raise ValueError(
            f"Unsupported artifact type {artifact_type!r} (model_name={model_name!r}). "
            "Only two_stream_lstm artifacts are supported. "
            "Verify that model_source.yaml points to the correct HF Hub filename."
        )

    from maestro.infrastructure.model.two_stream_lstm import (
        TwoStreamGestureLSTM,  # noqa: PLC0415
    )

    return TwoStreamGestureLSTM(
        input_size=int(model_config["input_size"]),
        hidden_size=int(model_config["hidden_size"]),
        num_layers=int(model_config["num_layers"]),
        dropout=float(model_config["dropout"]),
        num_classes=int(model_config["num_classes"]),
    ).to(device)


@dataclass(slots=True)
class InferenceArtifact:
    model: torch.nn.Module
    model_version: str
    class_labels: list[str]
    feature_schema_version: str
    feature_dim: int
    window_length: int
    window_step: int
    orientation_normalization: bool
    artifact_path: Path


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
    feature_schema = payload.get("feature_schema")
    state_dict = payload.get("model_state_dict")

    if not isinstance(model_config, dict):
        raise ValueError(f"Artifact missing model_config: {path}")
    if not isinstance(class_labels, list) or not class_labels:
        raise ValueError(f"Artifact missing class_labels: {path}")
    if not isinstance(feature_schema, dict):
        raise ValueError(f"Artifact missing feature_schema: {path}")
    if state_dict is None:
        raise ValueError(f"Artifact missing model_state_dict: {path}")

    artifact_type = str(payload.get("artifact_type", ""))
    model = _build_model_from_artifact(model_config, artifact_type, device)
    model.load_state_dict(state_dict)
    model.eval()

    return InferenceArtifact(
        model=model,
        model_version=str(payload.get("artifact_version", "unknown")),
        class_labels=[str(x) for x in class_labels],
        feature_schema_version=str(feature_schema.get("feature_schema_version", "unknown")),
        feature_dim=int(feature_schema.get("feature_dim") or model_config["input_size"]),
        window_length=int(feature_schema.get("window_length", 24)),
        window_step=int(feature_schema.get("window_step", 3)),
        orientation_normalization=bool(feature_schema.get("orientation_normalization", False)),
        artifact_path=path,
    )


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
