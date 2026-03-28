from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import torch

from gesture_control.training.train_lstm import load_training_config, train_lstm


def _create_dataset(path: Path) -> None:
    rng = np.random.default_rng(123)
    n, t, f = 128, 24, 126

    sequences = rng.normal(size=(n, t, f)).astype(np.float32)
    labels = np.array(["open_palm_hold"] * 64 + ["swipe_right"] * 64, dtype=object)
    label_ids = np.array([0] * 64 + [1] * 64, dtype=np.int64)

    metadata = {
        "feature_schema_version": "feature-schema-v1",
        "feature_dim": f,
        "window_length": t,
        "window_step": 3,
        "vocabulary": ["open_palm_hold", "swipe_right"],
        "enable_orientation_normalization": True,
    }

    np.savez_compressed(
        path,
        sequences=sequences,
        labels=labels,
        label_ids=label_ids,
        recording_ids=np.array(["rec"] * n, dtype=object),
        sample_ids=np.array([f"sample:{i}" for i in range(n)], dtype=object),
        start_timestamp_ms=np.arange(n, dtype=np.int64),
        end_timestamp_ms=np.arange(n, dtype=np.int64) + 23,
        valid_ratio=np.ones(n, dtype=np.float32),
        metadata_json=np.asarray(json.dumps(metadata), dtype=object),
    )


def test_training_pipeline_smoke(tmp_path: Path) -> None:
    dataset_path = tmp_path / "train.npz"
    _create_dataset(dataset_path)

    cfg_path = tmp_path / "lstm.yaml"
    cfg_path.write_text(
        """
model:
  model_name: lstm_smoke
  input_size: 126
  hidden_size: 32
  num_layers: 2
  dropout: 0.1
  num_classes: 2
training:
  epochs: 2
  batch_size: 16
  learning_rate: 0.001
  weight_decay: 0.0
  grad_clip_norm: 1.0
  early_stopping_patience: 2
  early_stopping_min_delta: 0.0
  seed: 7
  num_workers: 0
  class_weighting: false
splits:
  train_ratio: 0.7
  val_ratio: 0.15
  test_ratio: 0.15
  stratified: true
scheduler:
  enabled: false
  factor: 0.5
  patience: 2
  min_lr: 0.000001
export:
  checkpoint_dir: models/checkpoints
  exported_dir: models/exported
  registry_path: models/registry.json
  feature_schema_version: feature-schema-v1
""".strip(),
        encoding="utf-8",
    )

    cfg = load_training_config(cfg_path)
    result = train_lstm(
        dataset_path=dataset_path,
        config=cfg,
        output_dir=tmp_path / "checkpoints",
        run_root_dir=tmp_path / "runs",
        device=torch.device("cpu"),
    )

    assert result.checkpoint_path.exists()
    assert result.run_dir.exists()
    assert (result.run_dir / "training_metrics.json").exists()
    assert (result.run_dir / "config_snapshot.yaml").exists()
    assert result.test_report.accuracy >= 0.0
