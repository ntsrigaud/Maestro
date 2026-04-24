from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from maestro.training.augmentation import AugmentationConfig
from maestro.training.datamodule import DataSplitConfig, GestureDataModule

pytestmark = pytest.mark.training


def _create_dataset(path: Path) -> None:
    rng = np.random.default_rng(42)
    n, t, f = 120, 24, 147

    sequences = rng.normal(size=(n, t, f)).astype(np.float32)
    labels = np.array(["a"] * 60 + ["b"] * 60, dtype=object)
    label_ids = np.array([0] * 60 + [1] * 60, dtype=np.int64)

    metadata = {
        "feature_schema_version": "feature-schema-v3",
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
        use_group_split=False,  # single-recording toy dataset; no group split needed
    )
    dm.setup()

    assert dm.feature_dim == 147
    assert dm.sequence_length == 24
    assert dm.num_classes == 2

    train_batch_x, train_batch_y = next(iter(dm.train_dataloader()))
    assert train_batch_x.ndim == 3
    assert train_batch_x.shape[-1] == 147
    assert train_batch_y.ndim == 1


def test_datamodule_split_is_deterministic(tmp_path: Path) -> None:
    dataset_path = tmp_path / "toy.npz"
    _create_dataset(dataset_path)

    split = DataSplitConfig(train_ratio=0.7, val_ratio=0.15, test_ratio=0.15)
    dm1 = GestureDataModule(
        dataset_path=dataset_path,
        batch_size=16,
        split=split,
        seed=99,
        use_group_split=False,
    )
    dm2 = GestureDataModule(
        dataset_path=dataset_path,
        batch_size=16,
        split=split,
        seed=99,
        use_group_split=False,
    )

    dm1.setup()
    dm2.setup()

    b1_x, b1_y = next(iter(dm1.train_dataloader()))
    b2_x, b2_y = next(iter(dm2.train_dataloader()))

    assert np.allclose(b1_x.numpy(), b2_x.numpy())
    assert np.array_equal(b1_y.numpy(), b2_y.numpy())


# ---------------------------------------------------------------------------
# Helpers for multi-recording dataset
# ---------------------------------------------------------------------------


def _create_multi_recording_dataset(
    path: Path,
    *,
    recordings_per_class: int = 5,
    samples_per_recording: int = 15,
    n_classes: int = 2,
) -> None:
    """NPZ with *recordings_per_class* unique recording IDs per class."""
    rng = np.random.default_rng(7)
    T, F = 24, 147

    sequences_list, labels_list, label_ids_list, rec_ids_list = [], [], [], []
    for cls in range(n_classes):
        for rec in range(recordings_per_class):
            rec_id = f"rec-class{cls}-{rec:03d}"
            for _ in range(samples_per_recording):
                sequences_list.append(rng.normal(size=(T, F)).astype(np.float32))
                labels_list.append(f"class{cls}")
                label_ids_list.append(cls)
                rec_ids_list.append(rec_id)

    n = len(labels_list)
    sequences = np.stack(sequences_list)
    labels = np.array(labels_list, dtype=object)
    label_ids = np.array(label_ids_list, dtype=np.int64)
    recording_ids = np.array(rec_ids_list, dtype=object)

    metadata = {
        "feature_schema_version": "feature-schema-v3",
        "feature_dim": F,
        "window_length": T,
        "window_step": 3,
        "vocabulary": [f"class{i}" for i in range(n_classes)],
        "enable_orientation_normalization": True,
    }
    np.savez_compressed(
        path,
        sequences=sequences,
        labels=labels,
        label_ids=label_ids,
        recording_ids=recording_ids,
        sample_ids=np.array([f"s:{i}" for i in range(n)], dtype=object),
        start_timestamp_ms=np.arange(n, dtype=np.int64),
        end_timestamp_ms=np.arange(n, dtype=np.int64) + 23,
        valid_ratio=np.ones(n, dtype=np.float32),
        metadata_json=np.asarray(json.dumps(metadata), dtype=object),
    )


# ---------------------------------------------------------------------------
# Group split — leakage prevention
# ---------------------------------------------------------------------------


def test_group_split_no_recording_leakage(tmp_path: Path) -> None:
    """No recording ID should appear in more than one split partition."""
    dataset_path = tmp_path / "multi_rec.npz"
    _create_multi_recording_dataset(dataset_path, recordings_per_class=5)

    dm = GestureDataModule(
        dataset_path=dataset_path,
        batch_size=16,
        split=DataSplitConfig(train_ratio=0.7, val_ratio=0.15, test_ratio=0.15),
        seed=42,
        use_group_split=True,
    )
    dm.setup()

    info = dm.split_info()
    assert info["train"].isdisjoint(info["val"]), "Train/val recording overlap"
    assert info["train"].isdisjoint(info["test"]), "Train/test recording overlap"
    assert info["val"].isdisjoint(info["test"]), "Val/test recording overlap"


def test_group_split_all_recordings_assigned(tmp_path: Path) -> None:
    """Every recording must appear in exactly one split."""
    dataset_path = tmp_path / "multi_rec.npz"
    _create_multi_recording_dataset(dataset_path, recordings_per_class=6)

    dm = GestureDataModule(
        dataset_path=dataset_path,
        batch_size=16,
        split=DataSplitConfig(train_ratio=0.7, val_ratio=0.15, test_ratio=0.15),
        seed=0,
        use_group_split=True,
    )
    dm.setup()

    info = dm.split_info()
    all_assigned = info["train"] | info["val"] | info["test"]
    # 6 recordings × 2 classes = 12 unique recording IDs
    assert len(all_assigned) == 12


def test_group_split_splits_are_non_empty(tmp_path: Path) -> None:
    dataset_path = tmp_path / "multi_rec.npz"
    _create_multi_recording_dataset(dataset_path, recordings_per_class=5)

    dm = GestureDataModule(
        dataset_path=dataset_path,
        batch_size=16,
        split=DataSplitConfig(train_ratio=0.7, val_ratio=0.15, test_ratio=0.15),
        seed=1,
        use_group_split=True,
    )
    dm.setup()

    assert len(dm.train_dataloader().dataset) > 0
    assert len(dm.val_dataloader().dataset) > 0
    assert len(dm.test_dataloader().dataset) > 0


def test_group_split_fallback_to_random_with_single_recording(
    tmp_path: Path,
) -> None:
    """A dataset with one unique recording ID falls back to random split with a warning."""
    dataset_path = tmp_path / "single_rec.npz"
    _create_dataset(dataset_path)  # all recording_ids == "r"

    dm = GestureDataModule(
        dataset_path=dataset_path,
        batch_size=16,
        split=DataSplitConfig(train_ratio=0.7, val_ratio=0.15, test_ratio=0.15),
        seed=5,
        use_group_split=True,
    )
    with pytest.warns(UserWarning, match="falling back to random split"):
        dm.setup()

    # Must still produce valid splits despite the fallback.
    assert len(dm.train_dataloader().dataset) > 0


# ---------------------------------------------------------------------------
# Augmentation
# ---------------------------------------------------------------------------


def test_augmented_training_dataset_shape(tmp_path: Path) -> None:
    """AugmentedSequenceTensorDataset must yield tensors with the expected shape."""
    dataset_path = tmp_path / "toy.npz"
    _create_dataset(dataset_path)

    aug_config = AugmentationConfig(apply_prob=1.0, seed=0)
    dm = GestureDataModule(
        dataset_path=dataset_path,
        batch_size=16,
        split=DataSplitConfig(train_ratio=0.7, val_ratio=0.15, test_ratio=0.15),
        seed=42,
        augmentation=aug_config,
        use_group_split=False,
    )
    dm.setup()

    x, y = next(iter(dm.train_dataloader()))
    assert x.shape[-1] == 147
    assert x.ndim == 3
    assert y.ndim == 1


def test_augmentation_not_applied_to_val_test(tmp_path: Path) -> None:
    """Val and test loaders return the original (unaugmented) sequences."""
    dataset_path = tmp_path / "toy.npz"
    _create_dataset(dataset_path)

    aug_config = AugmentationConfig(apply_prob=1.0, seed=0)
    dm = GestureDataModule(
        dataset_path=dataset_path,
        batch_size=128,
        split=DataSplitConfig(train_ratio=0.7, val_ratio=0.15, test_ratio=0.15),
        seed=42,
        augmentation=aug_config,
        use_group_split=False,
    )
    dm.setup()

    # Calling the val dataloader twice with the same seed must yield identical batches,
    # which confirms no stochastic augmentation is applied.
    x1, _ = next(iter(dm.val_dataloader()))
    x2, _ = next(iter(dm.val_dataloader()))
    assert np.array_equal(x1.numpy(), x2.numpy())


@pytest.mark.parametrize("apply_prob", [0.0, 0.5, 1.0])
def test_augmentation_apply_prob_round_trip(tmp_path: Path, apply_prob: float) -> None:
    """DataModule with various apply_prob values must set up without error."""
    dataset_path = tmp_path / "toy.npz"
    _create_dataset(dataset_path)

    aug_config = AugmentationConfig(apply_prob=apply_prob, seed=0)
    dm = GestureDataModule(
        dataset_path=dataset_path,
        batch_size=16,
        split=DataSplitConfig(train_ratio=0.7, val_ratio=0.15, test_ratio=0.15),
        seed=42,
        augmentation=aug_config,
        use_group_split=False,
    )
    dm.setup()
    assert len(dm.train_dataloader().dataset) > 0
