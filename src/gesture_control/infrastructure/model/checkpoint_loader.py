from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch

from gesture_control.training.train_lstm import GestureLSTMClassifier
from gesture_control.utils.schema_validation import RuntimeConfig


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


def load_latest_inference_artifact(
    *,
    registry_path: str | Path,
    device: torch.device,
) -> InferenceArtifact:
    registry = _load_registry(registry_path)
    models = registry.get("models")
    if not isinstance(models, list) or not models:
        raise ValueError(f"No models available in registry: {registry_path}")

    latest = models[-1]
    if not isinstance(latest, dict):
        raise ValueError(f"Invalid registry entry in {registry_path}")

    exported_path = latest.get("exported_path")
    if not isinstance(exported_path, str) or not exported_path:
        raise ValueError(f"Missing exported_path in latest model entry: {registry_path}")

    return load_inference_artifact(artifact_path=exported_path, device=device)


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

    model = GestureLSTMClassifier(
        input_size=int(model_config["input_size"]),
        hidden_size=int(model_config["hidden_size"]),
        num_layers=int(model_config["num_layers"]),
        dropout=float(model_config["dropout"]),
        num_classes=int(model_config["num_classes"]),
    ).to(device)
    model.load_state_dict(state_dict)
    model.eval()

    return InferenceArtifact(
        model=model,
        model_version=str(payload.get("artifact_version", "unknown")),
        class_labels=[str(x) for x in class_labels],
        feature_schema_version=str(feature_schema.get("feature_schema_version", "unknown")),
        feature_dim=int(feature_schema.get("feature_dim", model_config["input_size"])),
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


def _load_registry(registry_path: str | Path) -> dict[str, Any]:
    path = Path(registry_path)
    if not path.exists():
        raise FileNotFoundError(f"Registry not found: {path}")

    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"Invalid registry payload: {path}")
    return payload
