from __future__ import annotations

import json
from pathlib import Path

import torch

from gesture_control.infrastructure.model.checkpoint_loader import (
    load_inference_artifact,
    load_latest_inference_artifact,
    validate_runtime_compatibility,
)
from gesture_control.training.train_lstm import GestureLSTMClassifier
from gesture_control.utils.schema_validation import RuntimeConfig


def _write_artifact(path: Path) -> None:
    model = GestureLSTMClassifier(
        input_size=126,
        hidden_size=32,
        num_layers=2,
        dropout=0.1,
        num_classes=2,
    )
    payload = {
        "artifact_type": "lstm_inference",
        "artifact_version": "lstm_test_v1",
        "model_state_dict": model.state_dict(),
        "model_config": {
            "input_size": 126,
            "hidden_size": 32,
            "num_layers": 2,
            "dropout": 0.1,
            "num_classes": 2,
        },
        "class_labels": ["open_palm_hold", "unknown"],
        "feature_schema": {
            "feature_schema_version": "feature-schema-v1",
            "feature_dim": 126,
            "window_length": 24,
            "window_step": 3,
            "orientation_normalization": True,
        },
    }
    torch.save(payload, path)


def test_load_inference_artifact(tmp_path: Path) -> None:
    artifact_path = tmp_path / "model.pt"
    _write_artifact(artifact_path)

    artifact = load_inference_artifact(
        artifact_path=artifact_path,
        device=torch.device("cpu"),
    )

    assert artifact.model_version == "lstm_test_v1"
    assert artifact.feature_dim == 126
    assert artifact.window_length == 24
    assert artifact.window_step == 3
    assert artifact.orientation_normalization is True
    assert artifact.class_labels == ["open_palm_hold", "unknown"]


def test_load_latest_inference_artifact(tmp_path: Path) -> None:
    artifact_path = tmp_path / "model.pt"
    _write_artifact(artifact_path)

    registry_path = tmp_path / "registry.json"
    registry_path.write_text(
        json.dumps(
            {
                "models": [
                    {
                        "model_version": "lstm_test_v1",
                        "exported_path": str(artifact_path),
                    }
                ]
            }
        ),
        encoding="utf-8",
    )

    artifact = load_latest_inference_artifact(
        registry_path=registry_path,
        device=torch.device("cpu"),
    )
    assert artifact.artifact_path == artifact_path


def test_validate_runtime_compatibility_passes(config_dir: Path) -> None:
    runtime = RuntimeConfig.model_validate(
        {
            "app": {
                "app_name": "x",
                "camera_index": 0,
                "frame_width": 1280,
                "frame_height": 720,
                "target_fps": 30,
                "active_app": "powerpoint",
            },
            "inference": {
                "window_length": 24,
                "window_step": 3,
                "min_valid_ratio": 0.8,
                "idle_timeout_ms": 3000,
                "command_cooldown_ms": 500,
            },
            "feature_pipeline": {
                "wrist_relative_normalization": True,
                "scale_normalization": True,
                "orientation_normalization": True,
                "velocity_features": True,
            },
        }
    )

    artifact_path = (
        config_dir.parent
        / "models"
        / "exported"
        / "lstm_v1_20260328T100850Z_inference.pt"
    )
    artifact = load_inference_artifact(
        artifact_path=artifact_path,
        device=torch.device("cpu"),
    )
    validate_runtime_compatibility(runtime_config=runtime, artifact=artifact)
