from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

from gesture_control.training.dataset_builder import SequenceDatasetBuilder


def _record(index: int, *, label: str) -> dict[str, object]:
    landmarks = []
    for i in range(21):
        landmarks.append(
            [
                0.2 + (i * 0.01) + (index * 0.0005),
                0.3 + (i * 0.02) + (index * 0.0003),
                i * 0.005,
            ]
        )
    return {
        "recording_id": "rec-cli",
        "timestamp_ms": index * 33,
        "label": label,
        "landmarks": landmarks,
        "valid": True,
        "handedness": "Right",
        "detect_conf": 0.95,
        "track_conf": 1.0,
    }


def test_build_dataset_script_exports_training_ready_sequences(tmp_path: Path) -> None:
    repo_root = Path(__file__).resolve().parents[2]
    input_jsonl = tmp_path / "labeled_landmarks.jsonl"
    output_npz = tmp_path / "dataset.npz"

    labels = ["swipe_right"] * 30
    with input_jsonl.open("w", encoding="utf-8") as file:
        for i, label in enumerate(labels):
            file.write(json.dumps(_record(i, label=label)) + "\n")

    result = subprocess.run(
        [
            sys.executable,
            str(repo_root / "scripts" / "build_dataset.py"),
            "--input-jsonl",
            str(input_jsonl),
            "--output-npz",
            str(output_npz),
            "--config-dir",
            str(repo_root / "configs"),
        ],
        cwd=repo_root,
        check=False,
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, result.stderr

    payload = SequenceDatasetBuilder.load_npz(output_npz)
    assert payload["sequences"].ndim == 3
    assert payload["sequences"].shape[2] == 126
    assert payload["labels"].shape[0] > 0
    assert payload["metadata"]["window_length"] == 24
