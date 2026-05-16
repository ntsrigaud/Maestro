from __future__ import annotations

import hashlib
import json
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import torch

from maestro.training.train_lstm import TrainingPipelineConfig, TrainingResult


def compute_sha256(path: str | Path) -> str:
    hasher = hashlib.sha256()
    with Path(path).open("rb") as f:
        for chunk in iter(lambda: f.read(8192), b""):
            hasher.update(chunk)
    return hasher.hexdigest()


def _two_stream_attn_model_config(
    *,
    model: torch.nn.Module,
    config: TrainingPipelineConfig,
) -> dict[str, Any]:
    """Serialize the actual EnhancedTwoStreamLSTM architecture into metadata."""

    return {
        "model_name": config.model.model_name,
        "input_size": int(getattr(model, "input_size", config.model.input_size)),
        "hidden_size": int(getattr(model, "hidden_size", config.model.hidden_size)),
        "num_layers": int(getattr(model, "num_layers", config.model.num_layers)),
        "dropout": float(getattr(model, "dropout", config.model.dropout)),
        "num_classes": int(getattr(model, "num_classes", config.model.num_classes)),
        "num_heads": int(getattr(model, "num_heads", config.model.num_heads)),
        "proj_dim": int(getattr(model, "proj_dim", config.model.proj_dim)),
    }


def export_inference_artifact(
    *,
    result: TrainingResult,
    config: TrainingPipelineConfig,
    export_dir: str | Path,
) -> Path:
    """Export production inference artifact with model + feature schema metadata."""
    path = Path(export_dir)
    path.mkdir(parents=True, exist_ok=True)

    artifact_path = path / f"{result.model_version}_inference.pt"

    is_two_stream_attn = config.model.model_name.startswith("two_stream_attn")

    if is_two_stream_attn:
        artifact_type = "two_stream_attn_inference"
    else:
        raise ValueError(
            f"Unsupported model_name {config.model.model_name!r} for inference export. "
            "Supported prefix: 'two_stream_attn_'."
        )

    model_config_dict: dict[str, Any] = _two_stream_attn_model_config(
        model=result.model,
        config=config,
    )
    payload: dict[str, Any] = {
        "artifact_type": artifact_type,
        "artifact_version": result.model_version,
        "created_at": datetime.now(UTC).isoformat(),
        "model_state_dict": result.model.state_dict(),
        "model_config": model_config_dict,
        "class_labels": result.class_labels,
    }

    def _meta_int(key: str, default: int) -> int:
        """Return int from dataset_metadata[key], falling back to *default* on None or missing."""
        val = result.dataset_metadata.get(key)
        return int(val) if val is not None else default

    def _meta_bool(key: str, default: bool = False) -> bool:
        val = result.dataset_metadata.get(key)
        return bool(val) if val is not None else default

    payload["feature_schema"] = {
        "feature_schema_version": config.export.feature_schema_version,
        "feature_dim": _meta_int("feature_dim", 126),
        "window_length": _meta_int("window_length", 24),
        "window_step": _meta_int("window_step", 3),
        "orientation_normalization": _meta_bool("enable_orientation_normalization"),
    }

    torch.save(payload, artifact_path)
    return artifact_path


def update_model_registry(
    *,
    registry_path: str | Path,
    model_version: str,
    checkpoint_path: str | Path,
    exported_path: str | Path,
    class_labels: list[str],
    dataset_metadata: dict[str, Any],
    feature_schema_version: str = "",
    training_data_version: str = "unknown",
) -> None:
    """Append a new model entry to the JSON registry file.

    Supports feature-schema based two-stream attention models.
    """
    path = Path(registry_path)
    if path.exists():
        registry: dict[str, Any] = json.loads(path.read_text(encoding="utf-8"))
    else:
        registry = {"models": []}

    entry: dict[str, Any] = {
        "model_version": model_version,
        "training_data_version": training_data_version,
        "class_labels": class_labels,
        "checkpoint_path": str(checkpoint_path),
        "exported_path": str(exported_path),
        "checkpoint_checksum": compute_sha256(checkpoint_path),
        "created_at": datetime.now(UTC).isoformat(),
        "window_length": int(dataset_metadata.get("window_length", 24)),
        "window_step": int(dataset_metadata.get("window_step", 3)),
    }

    entry["feature_schema_version"] = feature_schema_version
    entry["feature_dim"] = int(dataset_metadata.get("feature_dim", 0))

    registry.setdefault("models", []).append(entry)
    path.write_text(json.dumps(registry, indent=2, default=str), encoding="utf-8")
