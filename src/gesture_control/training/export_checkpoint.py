from __future__ import annotations

import hashlib
import json
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import torch

from gesture_control.training.train_lstm import TrainingPipelineConfig, TrainingResult


def compute_sha256(path: str | Path) -> str:
    hasher = hashlib.sha256()
    with Path(path).open("rb") as f:
        for chunk in iter(lambda: f.read(8192), b""):
            hasher.update(chunk)
    return hasher.hexdigest()


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
    artifact_type = (
        "tcn_inference"
        if config.model.model_name.startswith("tcn")
        else "lstm_inference"
    )
    payload: dict[str, Any] = {
        "artifact_type": artifact_type,
        "artifact_version": result.model_version,
        "created_at": datetime.now(UTC).isoformat(),
        "model_state_dict": result.model.state_dict(),
        "model_config": {
            "model_name": config.model.model_name,
            "input_size": config.model.input_size,
            "hidden_size": config.model.hidden_size,
            "num_layers": config.model.num_layers,
            "dropout": config.model.dropout,
            "num_classes": config.model.num_classes,
        },
        "class_labels": result.class_labels,
        "feature_schema": {
            "feature_schema_version": config.export.feature_schema_version,
            "feature_dim": int(result.dataset_metadata.get("feature_dim", 126)),
            "window_length": int(result.dataset_metadata.get("window_length", 24)),
            "window_step": int(result.dataset_metadata.get("window_step", 3)),
            "orientation_normalization": bool(
                result.dataset_metadata.get("enable_orientation_normalization", False)
            ),
        },
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
    feature_schema_version: str,
    training_data_version: str,
) -> dict[str, Any]:
    path = Path(registry_path)
    path.parent.mkdir(parents=True, exist_ok=True)

    if path.exists():
        existing = json.loads(path.read_text(encoding="utf-8"))
    else:
        existing = {"models": []}

    if not isinstance(existing, dict) or "models" not in existing:
        existing = {"models": []}

    checkpoint_checksum = compute_sha256(checkpoint_path)
    exported_checksum = compute_sha256(exported_path)

    entry = {
        "model_version": model_version,
        "training_data_version": training_data_version,
        "feature_schema_version": feature_schema_version,
        "class_labels": class_labels,
        "window_length": int(dataset_metadata.get("window_length", 24)),
        "checksum": exported_checksum,
        "feature_dim": int(dataset_metadata.get("feature_dim", 126)),
        "window_step": int(dataset_metadata.get("window_step", 3)),
        "checkpoint_path": str(checkpoint_path),
        "exported_path": str(exported_path),
        "checkpoint_checksum": checkpoint_checksum,
        "created_at": datetime.now(UTC).isoformat(),
    }

    models = existing.get("models")
    if not isinstance(models, list):
        models = []
    models.append(entry)
    existing["models"] = models

    path.write_text(json.dumps(existing, indent=2), encoding="utf-8")
    return entry
