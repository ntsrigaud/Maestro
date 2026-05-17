from __future__ import annotations

from pathlib import Path

import torch

from maestro.infrastructure.model.checkpoint_loader import (
    load_inference_artifact,
    validate_runtime_compatibility,
)
from maestro.infrastructure.model.two_stream_attn_lstm import EnhancedTwoStreamLSTM
from maestro.utils.schema_validation import RuntimeConfig


def _write_artifact(path: Path, *, window_length: int = 16, num_classes: int = 10) -> None:
    _labels_10 = [
        "unknown",
        "point_one",
        "point_two",
        "fist",
        "swiping_down",
        "swiping_left",
        "swiping_right",
        "swiping_up",
        "zooming_in_full_hand",
        "zooming_out_full_hand",
    ]
    _labels_8 = [
        "open_palm_hold",
        "swipe_right",
        "swipe_left",
        "swipe_up",
        "swipe_down",
        "zoom_in",
        "zoom_out",
        "unknown",
    ]
    class_labels = _labels_10 if num_classes == 10 else _labels_8
    model = EnhancedTwoStreamLSTM(
        input_size=147,
        hidden_size=64,
        num_layers=2,
        dropout=0.1,
        num_classes=num_classes,
        num_heads=4,
        proj_dim=64,
    )
    payload = {
        "artifact_type": "two_stream_attn_inference",
        "artifact_version": "two_stream_attn_test_v1",
        "model_state_dict": model.state_dict(),
        "model_config": {
            "model_name": "two_stream_attn_v1",
            "input_size": 147,
            "hidden_size": 64,
            "num_layers": 2,
            "dropout": 0.1,
            "num_classes": num_classes,
            "num_heads": 4,
            "proj_dim": 64,
        },
        "class_labels": class_labels,
        "feature_schema": {
            "feature_schema_version": "feature-schema-v5",
            "feature_dim": 147,
            "window_length": window_length,
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

    assert artifact.model_version == "two_stream_attn_test_v1"
    assert artifact.feature_dim == 147
    assert artifact.window_length == 16
    assert artifact.window_step == 3
    assert artifact.orientation_normalization is True
    assert len(artifact.class_labels) == 10
    assert artifact.class_labels[0] == "unknown"


def test_load_inference_artifact_rejects_legacy_two_stream_type(tmp_path: Path) -> None:
    """Legacy baseline runtime artifacts must raise ValueError."""
    artifact_path = tmp_path / "bad.pt"
    payload = {
        "artifact_type": "two_stream_lstm_inference",
        "artifact_version": "v0",
        "model_state_dict": {},
        "model_config": {
            "model_name": "two_stream_lstm_v1",
            "input_size": 147,
            "hidden_size": 64,
            "num_layers": 2,
            "dropout": 0.1,
            "num_classes": 8,
        },
        "class_labels": ["a", "b"],
        "feature_schema": {
            "feature_schema_version": "v1",
            "feature_dim": 147,
            "window_length": 32,
            "window_step": 3,
            "orientation_normalization": False,
        },
    }
    torch.save(payload, artifact_path)
    import pytest

    with pytest.raises(ValueError, match="Unsupported artifact type"):
        load_inference_artifact(artifact_path=artifact_path, device=torch.device("cpu"))


def test_validate_runtime_compatibility_passes(tmp_path: Path) -> None:
    runtime = RuntimeConfig.model_validate(
        {
            "app": {
                "app_name": "x",
                "camera_index": 0,
                "frame_width": 1280,
                "frame_height": 720,
                "target_fps": 30,
            },
            "inference": {
                "window_length": 16,
                "window_step": 3,
                "min_valid_ratio": 0.8,
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

    artifact_path = tmp_path / "model.pt"
    _write_artifact(artifact_path)
    artifact = load_inference_artifact(
        artifact_path=artifact_path,
        device=torch.device("cpu"),
    )
    validate_runtime_compatibility(runtime_config=runtime, artifact=artifact)


def test_validate_runtime_compatibility_rejects_window_mismatch(tmp_path: Path) -> None:
    """A legacy 24-frame artifact must fail validation against a 16-frame runtime."""
    import pytest

    runtime = RuntimeConfig.model_validate(
        {
            "app": {
                "app_name": "x",
                "camera_index": 0,
                "frame_width": 1280,
                "frame_height": 720,
                "target_fps": 30,
            },
            "inference": {
                "window_length": 16,
                "window_step": 3,
                "min_valid_ratio": 0.8,
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

    artifact_path = tmp_path / "legacy.pt"
    _write_artifact(artifact_path, window_length=24, num_classes=10)
    artifact = load_inference_artifact(artifact_path=artifact_path, device=torch.device("cpu"))

    with pytest.raises(ValueError, match="window_length"):
        validate_runtime_compatibility(runtime_config=runtime, artifact=artifact)
