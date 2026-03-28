from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from gesture_control.training.datamodule import DataSplitConfig, GestureDataModule


def _create_dataset(path: Path) -> None:
    rng = np.random.default_rng(42)
    n, t, f = 120, 24, 126

    sequences = rng.normal(size=(n, t, f)).astype(np.float32)
    labels = np.array(["a"] * 60 + ["b"] * 60, dtype=object)
    label_ids = np.array([0] * 60 + [1] * 60, dtype=np.int64)

    metadata = {
        "feature_schema_version": "feature-schema-v1",
        "feature_dim": f,
        "window_length": t,
        "window_step": 3,
        "vocabulary": ["a", "b"],
        "enable_orientation_normalization": True,
    }

    np.savez_compressed(
        path,
        sequences=sequences,
        labels=labels,
        label_ids=label_ids,
        recording_ids=np.array(["r"] * n, dtype=object),
        sample_ids=np.array([f"s:{i}" for i in range(n)], dtype=object),
        start_timestamp_ms=np.arange(n, dtype=np.int64),
        end_timestamp_ms=np.arange(n, dtype=np.int64) + 23,
        valid_ratio=np.ones(n, dtype=np.float32),
        metadata_json=np.asarray(json.dumps(metadata), dtype=object),
    )


def test_datamodule_setup_and_shapes(tmp_path: Path) -> None:
    dataset_path = tmp_path / "toy.npz"
    _create_dataset(dataset_path)

    dm = GestureDataModule(
        dataset_path=dataset_path,
        batch_size=16,
        split=DataSplitConfig(train_ratio=0.7, val_ratio=0.15, test_ratio=0.15),
        seed=123,
    )
    dm.setup()

    assert dm.feature_dim == 126
    assert dm.sequence_length == 24
    assert dm.num_classes == 2

    train_batch_x, train_batch_y = next(iter(dm.train_dataloader()))
    assert train_batch_x.ndim == 3
    assert train_batch_x.shape[-1] == 126
    assert train_batch_y.ndim == 1


def test_datamodule_split_is_deterministic(tmp_path: Path) -> None:
    dataset_path = tmp_path / "toy.npz"
    _create_dataset(dataset_path)

    split = DataSplitConfig(train_ratio=0.7, val_ratio=0.15, test_ratio=0.15)
    dm1 = GestureDataModule(dataset_path=dataset_path, batch_size=16, split=split, seed=99)
    dm2 = GestureDataModule(dataset_path=dataset_path, batch_size=16, split=split, seed=99)

    dm1.setup()
    dm2.setup()

    b1_x, b1_y = next(iter(dm1.train_dataloader()))
    b2_x, b2_y = next(iter(dm2.train_dataloader()))

    assert np.allclose(b1_x.numpy(), b2_x.numpy())
    assert np.array_equal(b1_y.numpy(), b2_y.numpy())
