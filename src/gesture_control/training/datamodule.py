from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, Dataset

from gesture_control.training.dataset_builder import SequenceDatasetBuilder


class SequenceTensorDataset(Dataset[tuple[torch.Tensor, torch.Tensor]]):
    """Torch dataset wrapper for sequence tensors and label ids."""

    def __init__(self, sequences: np.ndarray, label_ids: np.ndarray) -> None:
        if sequences.ndim != 3:
            raise ValueError("sequences must have shape (N, T, F)")
        if label_ids.ndim != 1:
            raise ValueError("label_ids must have shape (N,)")
        if len(sequences) != len(label_ids):
            raise ValueError("sequences and label_ids must have the same length")

        self._sequences = torch.from_numpy(sequences.astype(np.float32))
        self._label_ids = torch.from_numpy(label_ids.astype(np.int64))

    def __len__(self) -> int:
        return int(self._label_ids.shape[0])

    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor]:
        return self._sequences[index], self._label_ids[index]


@dataclass(slots=True)
class DataSplitConfig:
    train_ratio: float = 0.7
    val_ratio: float = 0.15
    test_ratio: float = 0.15
    stratified: bool = True

    def validate(self) -> None:
        for name, value in (
            ("train_ratio", self.train_ratio),
            ("val_ratio", self.val_ratio),
            ("test_ratio", self.test_ratio),
        ):
            if value <= 0.0 or value >= 1.0:
                raise ValueError(f"{name} must be between 0 and 1")

        total = self.train_ratio + self.val_ratio + self.test_ratio
        if not np.isclose(total, 1.0, atol=1e-6):
            raise ValueError("train_ratio + val_ratio + test_ratio must equal 1.0")


@dataclass(slots=True)
class DatasetBundle:
    sequences: np.ndarray
    labels: np.ndarray
    label_ids: np.ndarray
    metadata: dict[str, Any]


class GestureDataModule:
    """Loads NPZ dataset artifacts and prepares deterministic PyTorch dataloaders."""

    def __init__(
        self,
        *,
        dataset_path: str | Path,
        batch_size: int,
        split: DataSplitConfig,
        seed: int,
        num_workers: int = 0,
        pin_memory: bool = False,
    ) -> None:
        self._dataset_path = Path(dataset_path)
        self._batch_size = batch_size
        self._split = split
        self._seed = seed
        self._num_workers = num_workers
        self._pin_memory = pin_memory

        self._bundle: DatasetBundle | None = None
        self._train_dataset: SequenceTensorDataset | None = None
        self._val_dataset: SequenceTensorDataset | None = None
        self._test_dataset: SequenceTensorDataset | None = None

    @property
    def metadata(self) -> dict[str, Any]:
        if self._bundle is None:
            raise RuntimeError("Dataset is not loaded. Call setup() first.")
        return dict(self._bundle.metadata)

    @property
    def class_labels(self) -> list[str]:
        metadata = self.metadata
        labels = metadata.get("vocabulary")
        if isinstance(labels, list) and labels:
            return [str(item) for item in labels]

        if self._bundle is None:
            raise RuntimeError("Dataset is not loaded. Call setup() first.")
        return sorted(set(str(x) for x in self._bundle.labels.tolist()))

    @property
    def feature_dim(self) -> int:
        metadata = self.metadata
        if "feature_dim" in metadata:
            return int(metadata["feature_dim"])
        if self._bundle is None:
            raise RuntimeError("Dataset is not loaded. Call setup() first.")
        return int(self._bundle.sequences.shape[2])

    @property
    def sequence_length(self) -> int:
        metadata = self.metadata
        if "window_length" in metadata:
            return int(metadata["window_length"])
        if self._bundle is None:
            raise RuntimeError("Dataset is not loaded. Call setup() first.")
        return int(self._bundle.sequences.shape[1])

    @property
    def num_classes(self) -> int:
        labels = self.class_labels
        return len(labels)

    def setup(self) -> None:
        self._split.validate()

        if not self._dataset_path.exists():
            raise FileNotFoundError(f"Dataset not found: {self._dataset_path}")

        payload = SequenceDatasetBuilder.load_npz(self._dataset_path)
        sequences = np.asarray(payload["sequences"], dtype=np.float32)
        labels = np.asarray(payload["labels"], dtype=object)
        label_ids = np.asarray(payload["label_ids"], dtype=np.int64)
        metadata = dict(payload["metadata"])

        if sequences.ndim != 3:
            raise ValueError("Dataset sequences must have shape (N, T, F)")
        if sequences.shape[0] <= 0:
            raise ValueError("Dataset must contain at least one sequence")

        self._bundle = DatasetBundle(
            sequences=sequences,
            labels=labels,
            label_ids=label_ids,
            metadata=metadata,
        )

        train_idx, val_idx, test_idx = self._split_indices(label_ids)
        self._train_dataset = SequenceTensorDataset(sequences[train_idx], label_ids[train_idx])
        self._val_dataset = SequenceTensorDataset(sequences[val_idx], label_ids[val_idx])
        self._test_dataset = SequenceTensorDataset(sequences[test_idx], label_ids[test_idx])

    def train_dataloader(self) -> DataLoader[tuple[torch.Tensor, torch.Tensor]]:
        if self._train_dataset is None:
            raise RuntimeError("DataModule not initialized. Call setup() first.")
        generator = torch.Generator()
        generator.manual_seed(self._seed)
        return DataLoader(
            self._train_dataset,
            batch_size=self._batch_size,
            shuffle=True,
            num_workers=self._num_workers,
            pin_memory=self._pin_memory,
            generator=generator,
        )

    def val_dataloader(self) -> DataLoader[tuple[torch.Tensor, torch.Tensor]]:
        if self._val_dataset is None:
            raise RuntimeError("DataModule not initialized. Call setup() first.")
        return DataLoader(
            self._val_dataset,
            batch_size=self._batch_size,
            shuffle=False,
            num_workers=self._num_workers,
            pin_memory=self._pin_memory,
        )

    def test_dataloader(self) -> DataLoader[tuple[torch.Tensor, torch.Tensor]]:
        if self._test_dataset is None:
            raise RuntimeError("DataModule not initialized. Call setup() first.")
        return DataLoader(
            self._test_dataset,
            batch_size=self._batch_size,
            shuffle=False,
            num_workers=self._num_workers,
            pin_memory=self._pin_memory,
        )

    def label_distribution(self) -> dict[str, int]:
        if self._bundle is None:
            raise RuntimeError("DataModule not initialized. Call setup() first.")

        labels = self.class_labels
        counts = np.bincount(self._bundle.label_ids, minlength=len(labels))
        return {labels[i]: int(counts[i]) for i in range(len(labels))}

    def _split_indices(self, label_ids: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        n = label_ids.shape[0]
        all_idx = np.arange(n)

        temp_ratio = self._split.val_ratio + self._split.test_ratio
        stratify_primary = label_ids if self._split.stratified else None

        try:
            train_idx, temp_idx = train_test_split(
                all_idx,
                test_size=temp_ratio,
                random_state=self._seed,
                shuffle=True,
                stratify=stratify_primary,
            )
        except ValueError:
            train_idx, temp_idx = train_test_split(
                all_idx,
                test_size=temp_ratio,
                random_state=self._seed,
                shuffle=True,
                stratify=None,
            )

        val_share_within_temp = self._split.val_ratio / temp_ratio
        temp_labels = label_ids[temp_idx]
        stratify_secondary = temp_labels if self._split.stratified else None

        try:
            val_idx, test_idx = train_test_split(
                temp_idx,
                test_size=(1.0 - val_share_within_temp),
                random_state=self._seed,
                shuffle=True,
                stratify=stratify_secondary,
            )
        except ValueError:
            val_idx, test_idx = train_test_split(
                temp_idx,
                test_size=(1.0 - val_share_within_temp),
                random_state=self._seed,
                shuffle=True,
                stratify=None,
            )

        return np.asarray(train_idx), np.asarray(val_idx), np.asarray(test_idx)
