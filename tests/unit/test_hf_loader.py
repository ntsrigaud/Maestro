"""Unit tests for hf_loader.py.

``hf_hub_download`` is mocked so the tests run fully offline without requiring
a Hugging Face account or network access.
"""

from __future__ import annotations

from pathlib import Path
from unittest.mock import patch

import pytest
import torch

from gesture_control.infrastructure.model.two_stream_lstm import TwoStreamGestureLSTM

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _write_artifact(path: Path) -> None:
    model = TwoStreamGestureLSTM(
        input_size=147,
        hidden_size=64,
        num_layers=2,
        dropout=0.1,
        num_classes=8,
    )
    payload = {
        "artifact_type": "two_stream_lstm_inference",
        "artifact_version": "two_stream_test_v1",
        "model_state_dict": model.state_dict(),
        "model_config": {
            "model_name": "two_stream_lstm_v1",
            "input_size": 147,
            "hidden_size": 64,
            "num_layers": 2,
            "dropout": 0.1,
            "num_classes": 8,
        },
        "class_labels": [
            "open_palm_hold",
            "swipe_right",
            "swipe_left",
            "swipe_up",
            "swipe_down",
            "zoom_in",
            "zoom_out",
            "unknown",
        ],
        "feature_schema": {
            "feature_schema_version": "feature-schema-v5",
            "feature_dim": 147,
            "window_length": 24,
            "window_step": 3,
            "orientation_normalization": False,
        },
    }
    torch.save(payload, path)


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


def test_load_artifact_from_hub_calls_hf_hub_download(tmp_path: Path) -> None:
    """load_artifact_from_hub should delegate to hf_hub_download and return
    a correctly populated InferenceArtifact."""
    artifact_path = tmp_path / "lstm_v1_test_inference.pt"
    _write_artifact(artifact_path)

    with patch(
        "gesture_control.infrastructure.model.hf_loader.hf_hub_download"
    ) as mock_dl:
        mock_dl.return_value = str(artifact_path)

        from gesture_control.infrastructure.model.hf_loader import (
            load_artifact_from_hub,
        )

        artifact = load_artifact_from_hub(
            repo_id="ntsrigaud/gesture-control-lstm",
            filename="lstm_v1_test_inference.pt",
            revision="main",
            device=torch.device("cpu"),
        )

    mock_dl.assert_called_once_with(
        repo_id="ntsrigaud/gesture-control-lstm",
        filename="lstm_v1_test_inference.pt",
        revision="main",
        cache_dir=None,
    )
    assert artifact.feature_schema_version == "feature-schema-v5"
    assert artifact.feature_dim == 147
    assert artifact.model_version == "two_stream_test_v1"
    assert len(artifact.class_labels) == 8


def test_load_artifact_from_hub_passes_cache_dir(tmp_path: Path) -> None:
    """cache_dir should be forwarded to hf_hub_download as a string."""
    artifact_path = tmp_path / "model.pt"
    _write_artifact(artifact_path)

    with patch(
        "gesture_control.infrastructure.model.hf_loader.hf_hub_download"
    ) as mock_dl:
        mock_dl.return_value = str(artifact_path)

        from gesture_control.infrastructure.model.hf_loader import (
            load_artifact_from_hub,
        )

        load_artifact_from_hub(
            repo_id="ntsrigaud/gesture-control-lstm",
            filename="model.pt",
            revision="v1.0.0",
            cache_dir=tmp_path / "hf_cache",
            device=torch.device("cpu"),
        )

    _call_kwargs = mock_dl.call_args.kwargs
    assert _call_kwargs["cache_dir"] == str(tmp_path / "hf_cache")
    assert _call_kwargs["revision"] == "v1.0.0"


def test_load_artifact_from_hub_raises_on_missing_huggingface_hub(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """An informative ImportError should be raised when huggingface_hub is absent."""
    import gesture_control.infrastructure.model.hf_loader as hf_loader_module

    # Simulate huggingface_hub being absent by patching the module-level flag
    monkeypatch.setattr(hf_loader_module, "_HF_AVAILABLE", False)
    monkeypatch.setattr(hf_loader_module, "hf_hub_download", None)

    with pytest.raises(ImportError, match="huggingface_hub is required"):
        hf_loader_module.load_artifact_from_hub(
            repo_id="ntsrigaud/gesture-control-lstm",
            filename="model.pt",
            device=torch.device("cpu"),
        )
