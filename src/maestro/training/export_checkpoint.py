from __future__ import annotations

import hashlib
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


def export_inference_artifact(
    *,
    result: TrainingResult,
    config: TrainingPipelineConfig,
    export_dir: str | Path,
) -> Path:
    """Export production inference artifact with model + schema metadata.

    For ``two_stream_*`` models the artifact contains a ``feature_schema`` dict
    (backward-compatible).  For ``stgcn_*`` models it contains a ``graph_schema``
    dict with the adjacency matrix and node/coord counts.
    """
    path = Path(export_dir)
    path.mkdir(parents=True, exist_ok=True)

    artifact_path = path / f"{result.model_version}_inference.pt"

    is_stgcn = config.model.model_name.startswith("stgcn")
    artifact_type = "stgcn_lstm_inference" if is_stgcn else "two_stream_lstm_inference"

    model_config_dict: dict[str, Any]
    if is_stgcn:
        model_config_dict = {
            "model_name": config.model.model_name,
            "num_nodes": config.model.num_nodes,
            "num_coords": config.model.num_coords,
            "stgcn_channels": list(config.model.stgcn_channels),
            "num_stgcn_layers": config.model.num_stgcn_layers,
            "lstm_hidden": config.model.lstm_hidden,
            "lstm_layers": config.model.lstm_layers,
            "dropout": config.model.dropout,
            "num_classes": config.model.num_classes,
        }
    else:
        model_config_dict = {
            "model_name": config.model.model_name,
            "input_size": config.model.input_size,
            "hidden_size": config.model.hidden_size,
            "num_layers": config.model.num_layers,
            "dropout": config.model.dropout,
            "num_classes": config.model.num_classes,
        }

    payload: dict[str, Any] = {
        "artifact_type": artifact_type,
        "artifact_version": result.model_version,
        "created_at": datetime.now(UTC).isoformat(),
        "model_state_dict": result.model.state_dict(),
        "model_config": model_config_dict,
        "class_labels": result.class_labels,
    }

    if is_stgcn:
        # Include adjacency from model buffer for self-contained inference.
        adjacency_tensor = result.model.adjacency  # type: ignore[attr-defined]
        payload["graph_schema"] = {
            "graph_schema_version": config.export.graph_schema_version,
            "num_nodes": config.model.num_nodes,
            "num_coords": config.model.num_coords,
            "window_length": int(result.dataset_metadata.get("window_length", 24)),
            "window_step": int(result.dataset_metadata.get("window_step", 3)),
            "adjacency": adjacency_tensor,
        }
    else:
        payload["feature_schema"] = {
            "feature_schema_version": config.export.feature_schema_version,
            "feature_dim": int(result.dataset_metadata.get("feature_dim", 126)),
            "window_length": int(result.dataset_metadata.get("window_length", 24)),
            "window_step": int(result.dataset_metadata.get("window_step", 3)),
            "orientation_normalization": bool(
                result.dataset_metadata.get("enable_orientation_normalization", False)
            ),
        }

    torch.save(payload, artifact_path)
    return artifact_path
