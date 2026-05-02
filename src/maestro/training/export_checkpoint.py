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


def export_inference_artifact(
    *,
    result: TrainingResult,
    config: TrainingPipelineConfig,
    export_dir: str | Path,
) -> Path:
    """Export production inference artifact with model + schema metadata.

    For ``two_stream_*`` models the artifact contains a ``feature_schema`` dict
    (backward-compatible).  For ``dual_stgcn_*`` models it contains a ``graph_schema``
    dict with the adjacency matrix and node/coord counts.
    """
    path = Path(export_dir)
    path.mkdir(parents=True, exist_ok=True)

    artifact_path = path / f"{result.model_version}_inference.pt"

    is_dual_stgcn = config.model.model_name.startswith("dual_stgcn")
    is_graph_model = is_dual_stgcn

    if is_dual_stgcn:
        artifact_type = "dual_stgcn_lstm_inference"
    else:
        artifact_type = "two_stream_lstm_inference"

    model_config_dict: dict[str, Any]
    if is_dual_stgcn:
        model_config_dict = {
            "model_name": config.model.model_name,
            "num_nodes": config.model.num_nodes,
            "num_coords": config.model.num_coords,
            "phono_conv_channels": config.model.phono_conv_channels,
            "lstm_hidden": config.model.lstm_hidden,
            "lstm_layers": config.model.lstm_layers,
            "stgcn_channels": list(config.model.stgcn_channels),
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

    if is_graph_model:
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


def update_model_registry(
    *,
    registry_path: str | Path,
    model_version: str,
    checkpoint_path: str | Path,
    exported_path: str | Path,
    class_labels: list[str],
    dataset_metadata: dict[str, Any],
    feature_schema_version: str = "",
    graph_schema_version: str = "",
    training_data_version: str = "unknown",
) -> None:
    """Append a new model entry to the JSON registry file.

    Supports both flat-feature (two_stream) and graph-schema (stgcn) models.
    Pass *graph_schema_version* for STGCN models, *feature_schema_version* for
    two-stream models.
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

    if graph_schema_version:
        entry["graph_schema_version"] = graph_schema_version
        entry["num_nodes"] = int(dataset_metadata.get("num_nodes", 67))
        entry["num_coords"] = int(dataset_metadata.get("num_coords", 3))
    else:
        entry["feature_schema_version"] = feature_schema_version
        entry["feature_dim"] = int(dataset_metadata.get("feature_dim", 0))

    registry.setdefault("models", []).append(entry)
    path.write_text(json.dumps(registry, indent=2, default=str), encoding="utf-8")
