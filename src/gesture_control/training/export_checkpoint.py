from __future__ import annotations

import hashlib
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
    artifact_type = "two_stream_lstm_inference"
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
