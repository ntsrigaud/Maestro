from __future__ import annotations

import json
from pathlib import Path

from gesture_control.training.export_checkpoint import (
    compute_sha256,
    update_model_registry,
)


def test_compute_sha256(tmp_path: Path) -> None:
    path = tmp_path / "artifact.bin"
    path.write_bytes(b"abc")

    digest = compute_sha256(path)
    assert len(digest) == 64


def test_update_model_registry_appends_entry(tmp_path: Path) -> None:
    registry_path = tmp_path / "registry.json"
    registry_path.write_text(json.dumps({"models": []}), encoding="utf-8")

    checkpoint_path = tmp_path / "ckpt.pt"
    export_path = tmp_path / "artifact.pt"
    checkpoint_path.write_bytes(b"checkpoint-bytes")
    export_path.write_bytes(b"export-bytes")

    entry = update_model_registry(
        registry_path=registry_path,
        model_version="lstm_v1_20260323",
        checkpoint_path=checkpoint_path,
        exported_path=export_path,
        class_labels=["a", "b"],
        dataset_metadata={"window_length": 24, "feature_dim": 126, "window_step": 3},
        feature_schema_version="feature-schema-v1",
        training_data_version="gesture_dataset_v1",
    )

    payload = json.loads(registry_path.read_text(encoding="utf-8"))
    assert isinstance(payload["models"], list)
    assert len(payload["models"]) == 1
    assert payload["models"][0]["model_version"] == "lstm_v1_20260323"
    assert payload["models"][0]["checksum"] == entry["checksum"]
    assert payload["models"][0]["window_length"] == 24
