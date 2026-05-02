from __future__ import annotations

import warnings
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch
from sklearn.model_selection import GroupShuffleSplit, train_test_split
from torch.utils.data import DataLoader, Dataset

from maestro.training.augmentation import AugmentationConfig, GestureAugmenter
from maestro.training.dataset_builder import SequenceDatasetBuilder


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


class AugmentedSequenceTensorDataset(Dataset[tuple[torch.Tensor, torch.Tensor]]):
    """Training dataset wrapper that applies on-the-fly gesture augmentation.

    Each call to ``__getitem__`` augments the sequence independently, so the
    model sees a different perturbed sample on every epoch.
    """

    def __init__(
        self,
        sequences: np.ndarray,
        label_ids: np.ndarray,
        augmenter: GestureAugmenter,
    ) -> None:
        if sequences.ndim != 3:
            raise ValueError("sequences must have shape (N, T, F)")
        if label_ids.ndim != 1:
            raise ValueError("label_ids must have shape (N,)")
        if len(sequences) != len(label_ids):
            raise ValueError("sequences and label_ids must have the same length")

        # Keep as numpy so the augmenter can work without tensor round-trips.
        self._sequences_np = sequences.astype(np.float32)
        self._label_ids = torch.from_numpy(label_ids.astype(np.int64))
        self._augmenter = augmenter

    def __len__(self) -> int:
        return int(self._label_ids.shape[0])

    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor]:
        seq_aug = self._augmenter.augment(self._sequences_np[index])
        return torch.from_numpy(seq_aug), self._label_ids[index]


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
    recording_ids: np.ndarray
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
        use_group_split: bool = True,
        augmentation: AugmentationConfig | None = None,
    ) -> None:
        self._dataset_path = Path(dataset_path)
        self._batch_size = batch_size
        self._split = split
        self._seed = seed
        self._num_workers = num_workers
        self._pin_memory = pin_memory
        self._use_group_split = use_group_split
        self._augmentation = augmentation

        self._bundle: DatasetBundle | None = None
        self._train_dataset: SequenceTensorDataset | AugmentedSequenceTensorDataset | None = None
        self._val_dataset: SequenceTensorDataset | None = None
        self._test_dataset: SequenceTensorDataset | None = None
        self._train_idx: np.ndarray | None = None
        self._val_idx: np.ndarray | None = None
        self._test_idx: np.ndarray | None = None

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

    @property
    def train_indices(self) -> np.ndarray:
        if self._train_idx is None:
            raise RuntimeError("DataModule not initialized. Call setup() first.")
        return self._train_idx

    @property
    def val_indices(self) -> np.ndarray:
        if self._val_idx is None:
            raise RuntimeError("DataModule not initialized. Call setup() first.")
        return self._val_idx

    @property
    def test_indices(self) -> np.ndarray:
        if self._test_idx is None:
            raise RuntimeError("DataModule not initialized. Call setup() first.")
        return self._test_idx

    def setup(self) -> None:
        self._split.validate()

        if not self._dataset_path.exists():
            raise FileNotFoundError(f"Dataset not found: {self._dataset_path}")

        payload = SequenceDatasetBuilder.load_npz(self._dataset_path)
        sequences = np.asarray(payload["sequences"], dtype=np.float32)
        labels = np.asarray(payload["labels"], dtype=object)
        label_ids = np.asarray(payload["label_ids"], dtype=np.int64)
        recording_ids = np.asarray(payload["recording_ids"], dtype=object)
        metadata = dict(payload["metadata"])

        if sequences.ndim != 3:
            raise ValueError("Dataset sequences must have shape (N, T, F)")
        if sequences.shape[0] <= 0:
            raise ValueError("Dataset must contain at least one sequence")

        self._bundle = DatasetBundle(
            sequences=sequences,
            labels=labels,
            label_ids=label_ids,
            recording_ids=recording_ids,
            metadata=metadata,
        )

        # Choose group-aware or random split.
        unique_recs = np.unique(recording_ids)
        if self._use_group_split and len(unique_recs) >= 3:
            train_idx, val_idx, test_idx = self._group_split_indices(label_ids, recording_ids)
        else:
            if self._use_group_split and len(unique_recs) < 3:
                warnings.warn(
                    f"Group split requested but only {len(unique_recs)} unique "
                    "recording ID(s) found — falling back to random split. "
                    "Collect at least 3 recordings per class for proper group splitting.",
                    UserWarning,
                    stacklevel=2,
                )
            train_idx, val_idx, test_idx = self._split_indices(label_ids)

        self._train_idx = np.asarray(train_idx)
        self._val_idx = np.asarray(val_idx)
        self._test_idx = np.asarray(test_idx)

        # Build datasets; apply augmentation to training split only.
        if self._augmentation is not None:
            augmenter = GestureAugmenter(self._augmentation)
            self._train_dataset = AugmentedSequenceTensorDataset(
                sequences[self._train_idx],
                label_ids[self._train_idx],
                augmenter,
            )
        else:
            self._train_dataset = SequenceTensorDataset(
                sequences[self._train_idx], label_ids[self._train_idx]
            )
        self._val_dataset = SequenceTensorDataset(
            sequences[self._val_idx], label_ids[self._val_idx]
        )
        self._test_dataset = SequenceTensorDataset(
            sequences[self._test_idx], label_ids[self._test_idx]
        )

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

    def split_info(self) -> dict[str, set[str]]:
        """Return the recording IDs assigned to each split.

        Useful for verifying that no recording appears in more than one partition
        (i.e. no data-leakage between train / val / test).
        Requires :meth:`setup` to have been called first.
        """
        if self._bundle is None or self._train_idx is None:
            raise RuntimeError("DataModule not initialized. Call setup() first.")
        return {
            "train": set(self._bundle.recording_ids[self._train_idx].tolist()),
            "val": set(self._bundle.recording_ids[self._val_idx].tolist()),
            "test": set(self._bundle.recording_ids[self._test_idx].tolist()),
        }

    def _group_split_indices(
        self,
        label_ids: np.ndarray,
        recording_ids: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Split samples by recording group so no recording leaks across partitions.

        Requires at least 3 unique recording IDs (one per split).  If the temp
        partition ends up with fewer than 2 unique recordings, the val/test sub-
        split falls back to random ``train_test_split``.
        """
        n = len(label_ids)
        all_idx = np.arange(n)
        temp_ratio = self._split.val_ratio + self._split.test_ratio

        gss = GroupShuffleSplit(n_splits=1, test_size=temp_ratio, random_state=self._seed)
        train_idx, temp_idx = next(gss.split(all_idx, label_ids, groups=recording_ids))

        temp_rec_ids = recording_ids[temp_idx]
        unique_temp = np.unique(temp_rec_ids)

        if len(unique_temp) >= 2:
            val_share = self._split.val_ratio / temp_ratio
            gss2 = GroupShuffleSplit(
                n_splits=1,
                test_size=1.0 - val_share,
                random_state=self._seed,
            )
            rel_val, rel_test = next(gss2.split(temp_idx, label_ids[temp_idx], groups=temp_rec_ids))
            val_idx = temp_idx[rel_val]
            test_idx = temp_idx[rel_test]
        else:
            # Only one recording in the temp pool — random sub-split is safe here.
            val_share = self._split.val_ratio / temp_ratio
            val_idx, test_idx = train_test_split(
                temp_idx,
                test_size=1.0 - val_share,
                random_state=self._seed,
                shuffle=True,
            )

        return np.asarray(train_idx), np.asarray(val_idx), np.asarray(test_idx)

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


# ──────────────────────────────────────────────────────────────────────────────
# Graph sequence datasets (graph-schema-v1, shape (T, N, C))
# ──────────────────────────────────────────────────────────────────────────────


class GraphSequenceTensorDataset(Dataset[tuple[torch.Tensor, torch.Tensor]]):
    """Torch dataset wrapper for 4-D graph sequence tensors.

    Accepts NPZ sequences of shape ``(M, T, N, C)`` where M is the sample count.
    Returns ``(T, N, C)`` float32 tensors paired with int64 label ids.
    """

    def __init__(self, sequences: np.ndarray, label_ids: np.ndarray) -> None:
        if sequences.ndim != 4:
            raise ValueError(f"sequences must have shape (M, T, N, C), got {sequences.shape}")
        if label_ids.ndim != 1:
            raise ValueError("label_ids must have shape (M,)")
        if len(sequences) != len(label_ids):
            raise ValueError("sequences and label_ids must have the same length")
        self._sequences = torch.from_numpy(sequences.astype(np.float32))
        self._label_ids = torch.from_numpy(label_ids.astype(np.int64))

    def __len__(self) -> int:
        return int(self._label_ids.shape[0])

    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor]:
        return self._sequences[index], self._label_ids[index]


class AugmentedGraphSequenceTensorDataset(Dataset[tuple[torch.Tensor, torch.Tensor]]):
    """Training dataset wrapper that applies on-the-fly graph augmentation."""

    def __init__(
        self,
        sequences: np.ndarray,
        label_ids: np.ndarray,
        augmenter: Any,
    ) -> None:
        if sequences.ndim != 4:
            raise ValueError(f"sequences must have shape (M, T, N, C), got {sequences.shape}")
        if label_ids.ndim != 1:
            raise ValueError("label_ids must have shape (M,)")
        if len(sequences) != len(label_ids):
            raise ValueError("sequences and label_ids must have the same length")
        self._sequences_np = sequences.astype(np.float32)
        self._label_ids = torch.from_numpy(label_ids.astype(np.int64))
        self._augmenter = augmenter

    def __len__(self) -> int:
        return int(self._label_ids.shape[0])

    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor]:
        seq_aug = self._augmenter.augment(self._sequences_np[index])
        return torch.from_numpy(seq_aug), self._label_ids[index]


@dataclass(slots=True)
class GraphDatasetBundle:
    sequences: np.ndarray  # (M, T, N, C)
    labels: np.ndarray
    label_ids: np.ndarray
    recording_ids: np.ndarray
    adjacency: np.ndarray  # (N, N)
    valid_masks: np.ndarray  # (M, T, N)
    metadata: dict[str, Any]


class GraphGestureDataModule:
    """Loads graph-schema NPZ artifacts and prepares PyTorch dataloaders.

    The NPZ must have been produced by :class:`GraphSequenceDatasetBuilder`
    and contain an ``adjacency`` key (shape ``(N, N)``).  The resulting
    dataloaders yield ``(B, T, N, C)`` float32 tensors.
    """

    def __init__(
        self,
        *,
        dataset_path: str | Path,
        batch_size: int,
        split: DataSplitConfig,
        seed: int,
        num_workers: int = 0,
        pin_memory: bool = False,
        use_group_split: bool = True,
        graph_augmentation: Any = None,
    ) -> None:
        self._dataset_path = Path(dataset_path)
        self._batch_size = batch_size
        self._split = split
        self._seed = seed
        self._num_workers = num_workers
        self._pin_memory = pin_memory
        self._use_group_split = use_group_split
        self._graph_augmentation = graph_augmentation

        self._bundle: GraphDatasetBundle | None = None
        self._train_dataset: (
            GraphSequenceTensorDataset | AugmentedGraphSequenceTensorDataset | None
        ) = None
        self._val_dataset: GraphSequenceTensorDataset | None = None
        self._test_dataset: GraphSequenceTensorDataset | None = None
        self._train_idx: np.ndarray | None = None
        self._val_idx: np.ndarray | None = None
        self._test_idx: np.ndarray | None = None

    # ── Properties ────────────────────────────────────────────────────

    @property
    def metadata(self) -> dict[str, Any]:
        if self._bundle is None:
            raise RuntimeError("Dataset is not loaded. Call setup() first.")
        return dict(self._bundle.metadata)

    @property
    def adjacency(self) -> np.ndarray:
        if self._bundle is None:
            raise RuntimeError("Dataset is not loaded. Call setup() first.")
        return self._bundle.adjacency

    @property
    def num_nodes(self) -> int:
        if self._bundle is None:
            raise RuntimeError("Dataset is not loaded. Call setup() first.")
        return int(self._bundle.sequences.shape[2])

    @property
    def num_coords(self) -> int:
        if self._bundle is None:
            raise RuntimeError("Dataset is not loaded. Call setup() first.")
        return int(self._bundle.sequences.shape[3])

    @property
    def sequence_length(self) -> int:
        if self._bundle is None:
            raise RuntimeError("Dataset is not loaded. Call setup() first.")
        return int(self._bundle.sequences.shape[1])

    @property
    def class_labels(self) -> list[str]:
        meta = self.metadata
        labels = meta.get("vocabulary")
        if isinstance(labels, list) and labels:
            return [str(x) for x in labels]
        if self._bundle is None:
            raise RuntimeError("Dataset is not loaded. Call setup() first.")
        return sorted(set(str(x) for x in self._bundle.labels.tolist()))

    @property
    def num_classes(self) -> int:
        return len(self.class_labels)

    @property
    def graph_schema_version(self) -> str:
        return str(self.metadata.get("graph_schema_version", "unknown"))

    @property
    def train_indices(self) -> np.ndarray:
        if self._train_idx is None:
            raise RuntimeError("DataModule not initialized. Call setup() first.")
        return self._train_idx

    @property
    def val_indices(self) -> np.ndarray:
        if self._val_idx is None:
            raise RuntimeError("DataModule not initialized. Call setup() first.")
        return self._val_idx

    @property
    def test_indices(self) -> np.ndarray:
        if self._test_idx is None:
            raise RuntimeError("DataModule not initialized. Call setup() first.")
        return self._test_idx

    # ── Setup ─────────────────────────────────────────────────────────

    def setup(self) -> None:
        self._split.validate()

        if not self._dataset_path.exists():
            raise FileNotFoundError(f"Dataset not found: {self._dataset_path}")

        payload = SequenceDatasetBuilder.load_npz(self._dataset_path)
        sequences = np.asarray(payload["sequences"], dtype=np.float32)
        labels = np.asarray(payload["labels"], dtype=object)
        label_ids = np.asarray(payload["label_ids"], dtype=np.int64)
        recording_ids = np.asarray(payload["recording_ids"], dtype=object)
        metadata = dict(payload["metadata"])

        if "adjacency" not in payload:
            raise ValueError(
                "NPZ file does not contain an 'adjacency' key — "
                "this does not appear to be a graph-schema dataset. "
                "Use GestureDataModule for flat-feature datasets."
            )
        adjacency = np.asarray(payload["adjacency"], dtype=np.float32)

        valid_masks_raw = payload.get("valid_masks")
        valid_masks = (
            np.asarray(valid_masks_raw, dtype=bool)
            if valid_masks_raw is not None
            else np.ones(sequences.shape[:3], dtype=bool)
        )

        if sequences.ndim != 4:
            raise ValueError(
                f"Graph dataset sequences must have shape (M, T, N, C), got {sequences.shape}"
            )
        if sequences.shape[0] <= 0:
            raise ValueError("Dataset must contain at least one sequence")

        self._bundle = GraphDatasetBundle(
            sequences=sequences,
            labels=labels,
            label_ids=label_ids,
            recording_ids=recording_ids,
            adjacency=adjacency,
            valid_masks=valid_masks,
            metadata=metadata,
        )

        unique_recs = np.unique(recording_ids)
        if self._use_group_split and len(unique_recs) >= 3:
            train_idx, val_idx, test_idx = self._group_split_indices(label_ids, recording_ids)
        else:
            if self._use_group_split and len(unique_recs) < 3:
                warnings.warn(
                    f"Group split requested but only {len(unique_recs)} unique "
                    "recording ID(s) found — falling back to random split.",
                    UserWarning,
                    stacklevel=2,
                )
            train_idx, val_idx, test_idx = self._split_indices(label_ids)

        self._train_idx = np.asarray(train_idx)
        self._val_idx = np.asarray(val_idx)
        self._test_idx = np.asarray(test_idx)

        if self._graph_augmentation is not None:
            from maestro.training.augmentation import GraphGestureAugmenter

            augmenter = GraphGestureAugmenter(self._graph_augmentation)
            self._train_dataset = AugmentedGraphSequenceTensorDataset(
                sequences[self._train_idx],
                label_ids[self._train_idx],
                augmenter,
            )
        else:
            self._train_dataset = GraphSequenceTensorDataset(
                sequences[self._train_idx], label_ids[self._train_idx]
            )
        self._val_dataset = GraphSequenceTensorDataset(
            sequences[self._val_idx], label_ids[self._val_idx]
        )
        self._test_dataset = GraphSequenceTensorDataset(
            sequences[self._test_idx], label_ids[self._test_idx]
        )

    # ── Dataloaders ───────────────────────────────────────────────────

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

    # ── Split helpers ─────────────────────────────────────────────────

    def _group_split_indices(
        self,
        label_ids: np.ndarray,
        recording_ids: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        n = len(label_ids)
        all_idx = np.arange(n)
        temp_ratio = self._split.val_ratio + self._split.test_ratio
        gss = GroupShuffleSplit(n_splits=1, test_size=temp_ratio, random_state=self._seed)
        train_idx, temp_idx = next(gss.split(all_idx, label_ids, groups=recording_ids))
        temp_rec_ids = recording_ids[temp_idx]
        unique_temp = np.unique(temp_rec_ids)
        if len(unique_temp) >= 2:
            val_share = self._split.val_ratio / temp_ratio
            gss2 = GroupShuffleSplit(n_splits=1, test_size=1.0 - val_share, random_state=self._seed)
            rel_val, rel_test = next(gss2.split(temp_idx, label_ids[temp_idx], groups=temp_rec_ids))
            val_idx = temp_idx[rel_val]
            test_idx = temp_idx[rel_test]
        else:
            val_share = self._split.val_ratio / temp_ratio
            val_idx, test_idx = train_test_split(
                temp_idx, test_size=1.0 - val_share, random_state=self._seed, shuffle=True
            )
        return np.asarray(train_idx), np.asarray(val_idx), np.asarray(test_idx)

    def _split_indices(self, label_ids: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        n = label_ids.shape[0]
        all_idx = np.arange(n)
        temp_ratio = self._split.val_ratio + self._split.test_ratio
        stratify = label_ids if self._split.stratified else None
        try:
            train_idx, temp_idx = train_test_split(
                all_idx,
                test_size=temp_ratio,
                random_state=self._seed,
                shuffle=True,
                stratify=stratify,
            )
        except ValueError:
            train_idx, temp_idx = train_test_split(
                all_idx, test_size=temp_ratio, random_state=self._seed, shuffle=True, stratify=None
            )
        val_share = self._split.val_ratio / temp_ratio
        temp_labels = label_ids[temp_idx]
        stratify2 = temp_labels if self._split.stratified else None
        try:
            val_idx, test_idx = train_test_split(
                temp_idx,
                test_size=1.0 - val_share,
                random_state=self._seed,
                shuffle=True,
                stratify=stratify2,
            )
        except ValueError:
            val_idx, test_idx = train_test_split(
                temp_idx,
                test_size=1.0 - val_share,
                random_state=self._seed,
                shuffle=True,
                stratify=None,
            )
        return np.asarray(train_idx), np.asarray(val_idx), np.asarray(test_idx)
