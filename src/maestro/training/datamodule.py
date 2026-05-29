from __future__ import annotations

import ast as _ast
import struct as _struct
import warnings
import zipfile as _zipfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset

from maestro.training.augmentation import AugmentationConfig, GestureAugmenter
from maestro.training.dataset_builder import SequenceDatasetBuilder


class SequenceTensorDataset(Dataset[tuple[torch.Tensor, torch.Tensor]]):
    """Torch dataset wrapper for sequence tensors and label ids.

    When *indices* is provided the dataset holds a shared reference to the full
    *sequences* array and slices one window per ``__getitem__`` call, avoiding
    the per-split upfront copy that can cost several GiB of RAM.
    """

    def __init__(
        self,
        sequences: np.ndarray,
        label_ids: np.ndarray,
        indices: np.ndarray | None = None,
    ) -> None:
        if sequences.ndim != 3:
            raise ValueError("sequences must have shape (N, T, F)")
        if label_ids.ndim != 1:
            raise ValueError("label_ids must have shape (N,)")
        if indices is not None:
            if len(label_ids) != len(indices):
                raise ValueError(
                    "label_ids and indices must have the same length when indices is provided"
                )
        elif len(sequences) != len(label_ids):
            raise ValueError("sequences and label_ids must have the same length")

        # Store as numpy — per-item slices are cheap (T, F) views into the full array.
        self._sequences = sequences.astype(np.float32, copy=False)
        self._indices = indices
        self._label_ids = torch.from_numpy(label_ids.astype(np.int64, copy=False))

    def __len__(self) -> int:
        return int(self._label_ids.shape[0])

    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor]:
        seq = self._sequences[int(self._indices[index]) if self._indices is not None else index]
        return torch.from_numpy(np.asarray(seq, dtype=np.float32)), self._label_ids[index]


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
        indices: np.ndarray | None = None,
    ) -> None:
        if sequences.ndim != 3:
            raise ValueError("sequences must have shape (N, T, F)")
        if label_ids.ndim != 1:
            raise ValueError("label_ids must have shape (N,)")
        if indices is not None:
            if len(label_ids) != len(indices):
                raise ValueError(
                    "label_ids and indices must have the same length when indices is provided"
                )
        elif len(sequences) != len(label_ids):
            raise ValueError("sequences and label_ids must have the same length")

        # Keep as numpy; when indices is provided sequences is the full array —
        # no per-split copy is made.  Augmentation creates a new array per item.
        self._sequences_np = sequences.astype(np.float32, copy=False)
        self._indices = indices
        self._label_ids = torch.from_numpy(label_ids.astype(np.int64, copy=False))
        self._augmenter = augmenter

    def __len__(self) -> int:
        return int(self._label_ids.shape[0])

    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor]:
        seq = self._sequences_np[int(self._indices[index]) if self._indices is not None else index]
        seq_aug = self._augmenter.augment(seq)
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


# ── Streaming NPZ reader ──────────────────────────────────────────────────────
# numpy compressed NPZ files store each array as a separate .npy entry inside a
# ZIP archive.  np.load() decompresses the full array into memory on first
# access — for the sequences array this is 6–7 GB, which exceeds available RAM
# on development machines.  The helpers below stream the array row-by-row
# through the ZIP decompressor so that only one chunk (~90 MB) is live at once.


def _parse_npy_header(stream: Any) -> tuple[np.dtype, tuple[int, ...]]:
    """Read the numpy .npy header from *stream* and return (dtype, shape)."""
    prefix = stream.read(8)
    if prefix[:6] != b"\x93NUMPY":
        raise ValueError("Not a valid .npy file")
    major = prefix[6]
    hdr_len = _struct.unpack("<H" if major == 1 else "<I", stream.read(2 if major == 1 else 4))[0]
    hdr = _ast.literal_eval(stream.read(hdr_len).decode("latin1").strip())
    if hdr.get("fortran_order", False):
        raise ValueError("Fortran-order arrays are not supported by the streaming reader")
    return np.dtype(hdr["descr"]), tuple(hdr["shape"])


def _iter_sequences_chunked(
    npz_path: Path,
    chunk_size: int = 4_000,
    key: str = "sequences",
) -> Any:
    """Yield (start_index, chunk) pairs streaming from a compressed NPZ.

    The sequences array is never fully decompressed; each call to
    ``zipfile.ZipExtFile.read(n_bytes)`` decompresses incrementally.  Peak RAM
    per call is one chunk (``chunk_size × T × F × 4`` bytes ≈ 90 MB for the
    default chunk_size with T=32, F=147).

    Yields
    ------
    start : int
        Index of the first row in this chunk relative to the full array.
    chunk : np.ndarray, shape (n, T, F), dtype float32
        A writable copy of the current chunk cast to float32.
    """
    with _zipfile.ZipFile(str(npz_path), "r") as zf:
        with zf.open(f"{key}.npy", "r") as f:
            dtype, shape = _parse_npy_header(f)
            N_total, *rest = shape
            row_bytes = int(np.prod(rest)) * dtype.itemsize
            start = 0
            while start < N_total:
                n = min(chunk_size, N_total - start)
                raw = f.read(n * row_bytes)
                actual_n = len(raw) // row_bytes
                if actual_n == 0:
                    break
                raw = raw[: actual_n * row_bytes]
                chunk = np.frombuffer(raw, dtype=dtype).reshape(actual_n, *rest)
                yield start, chunk.astype(np.float32)
                start += actual_n


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
        max_samples_per_class: int | None = None,
        exclude_classes: list[str] | None = None,
    ) -> None:
        self._dataset_path = Path(dataset_path)
        self._batch_size = batch_size
        self._split = split
        self._seed = seed
        self._num_workers = num_workers
        self._pin_memory = pin_memory
        self._use_group_split = use_group_split
        self._augmentation = augmentation
        self._max_samples_per_class = max_samples_per_class
        self._exclude_classes: list[str] | None = exclude_classes

        self._bundle: DatasetBundle | None = None
        self._class_labels_override: list[str] | None = None
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
        # When exclude_classes was applied without a full setup() (e.g. after
        # setup_indices_only()), the filtered vocabulary is stored here.
        if self._class_labels_override is not None:
            return list(self._class_labels_override)

        # Fast path: bundle is loaded (after setup()).
        if self._bundle is not None:
            metadata = dict(self._bundle.metadata)
            labels = metadata.get("vocabulary")
            if isinstance(labels, list) and labels:
                return [str(item) for item in labels]
            return sorted(set(str(x) for x in self._bundle.labels.tolist()))

        # Fallback: setup_indices_only() was called without vocab, or vocab was
        # not stored (e.g. older kernel with cached module).  Load vocabulary
        # from the dataset file — metadata-only, no sequences are decompressed.
        try:
            payload = SequenceDatasetBuilder.load_npz(self._dataset_path)
            vocab = payload["metadata"].get("vocabulary")
            if isinstance(vocab, list) and vocab:
                self._class_labels_override = [str(v) for v in vocab]
                return list(self._class_labels_override)
        except Exception:
            pass

        raise RuntimeError(
            "class_labels: vocabulary could not be determined. "
            "Call setup() or setup_indices_only(vocab=...) first."
        )

    def _load_metadata_from_file(self) -> dict:
        """Load dataset metadata without decompressing sequences."""
        payload = SequenceDatasetBuilder.load_npz(self._dataset_path)
        return dict(payload["metadata"])

    @property
    def feature_dim(self) -> int:
        if self._bundle is not None:
            metadata = dict(self._bundle.metadata)
        else:
            metadata = self._load_metadata_from_file()
        if "feature_dim" in metadata:
            return int(metadata["feature_dim"])
        if self._bundle is None:
            raise RuntimeError("feature_dim not in metadata. Call setup() first.")
        return int(self._bundle.sequences.shape[2])

    @property
    def sequence_length(self) -> int:
        if self._bundle is not None:
            metadata = dict(self._bundle.metadata)
        else:
            metadata = self._load_metadata_from_file()
        if "window_length" in metadata:
            return int(metadata["window_length"])
        if self._bundle is None:
            raise RuntimeError("window_length not in metadata. Call setup() first.")
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

        # ── Load small arrays cheaply from the NPZ ────────────────────────────
        # np.load on a compressed NPZ is lazy — arrays are decompressed only on
        # first item access.  We access only the small side-car arrays here and
        # deliberately avoid touching "sequences" (6–7 GB uncompressed).
        payload = SequenceDatasetBuilder.load_npz(self._dataset_path)
        labels = np.asarray(payload["labels"], dtype=object)
        label_ids = np.asarray(payload["label_ids"], dtype=np.int64)
        recording_ids = np.asarray(payload["recording_ids"], dtype=object)
        metadata = dict(payload["metadata"])

        N = len(label_ids)
        if N == 0:
            raise ValueError("Dataset must contain at least one sequence")

        # ── Determine which samples to keep ───────────────────────────────────
        # When exclude_classes is active we compute the keep mask from label_ids
        # (cheap, ~3 MB) before any sequences memory is allocated.
        if self._exclude_classes:
            vocab = list(metadata.get("vocabulary") or [])
            filtered_vocab, keep_indices, remapped_ids = self._apply_exclude_classes(
                vocab, label_ids
            )
            labels = labels[keep_indices]
            label_ids = remapped_ids
            recording_ids = recording_ids[keep_indices]
            metadata = {**metadata, "vocabulary": filtered_vocab}
        else:
            keep_indices = None  # keep all

        N_kept = len(label_ids)

        # ── Apply per-class cap BEFORE splitting ─────────────────────────────
        # Capping the full dataset (not just the training split) ensures that
        # val and test are proportionally balanced across classes.  Without this
        # step, majority classes (e.g. point_one/point_two with ~56 k sequences)
        # produce ~8 400 val/test samples each while minority classes produce
        # only ~360, creating a 23× evaluation disparity that masks poor
        # per-class performance on minority gestures.
        if self._max_samples_per_class is not None:
            cap_sub_idx = self._downsample_dataset_indices(label_ids)
            labels = labels[cap_sub_idx]
            label_ids = label_ids[cap_sub_idx]
            recording_ids = recording_ids[cap_sub_idx]
            # Compose with any earlier exclude_classes filter so the streaming
            # step builds the sequences array from the right original rows.
            if keep_indices is not None:
                keep_indices = keep_indices[cap_sub_idx]
            else:
                keep_indices = cap_sub_idx
            N_kept = len(label_ids)

        # ── Compute splits on the (possibly filtered + capped) label_ids ──────
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

        # ── Stream sequences into a single compact allocation ─────────────────
        # We stream the full sequences array chunk-by-chunk through the ZIP
        # decompressor, applying the keep_indices filter on each chunk so that
        # only the kept rows land in the output array.  Peak RAM = one chunk
        # (~90 MB) + the final allocation (N_kept × T × F × 4 bytes).
        #
        # For the v6 dataset on a machine with 6.5 GB free:
        #   Old approach (np.load → 6.64 GB + fancy-index copy → 6.22 GB):
        #     peak ≈ 13 GB  → OOM kill
        #   New approach (streaming → compact 6.22 GB result):
        #     peak ≈ 6.3 GB  → fits in available RAM
        meta = metadata
        # Peek at the actual sequences shape from the NPZ ZIP header (reads only
        # a few hundred bytes — no decompression of the data array).  This guards
        # against legacy NPZ files that omit 'window_length' / 'feature_dim' in
        # their metadata_json, where a hardcoded fallback would silently
        # misallocate the sequences buffer and cause a broadcast ValueError.
        with _zipfile.ZipFile(str(self._dataset_path), "r") as _zf_peek:
            with _zf_peek.open("sequences.npy", "r") as _hdr:
                _, _actual_seq_shape = _parse_npy_header(_hdr)
        T = int(meta.get("window_length", _actual_seq_shape[1]))
        F = int(meta.get("feature_dim", _actual_seq_shape[2]))

        sequences = np.empty((N_kept, T, F), dtype=np.float32)
        write_pos = 0

        if keep_indices is not None:
            # keep_mask is a boolean array over the original N rows.
            # For each streaming chunk we slice just the kept rows — one cheap
            # boolean-index on a small chunk rather than a full-array fancy-index.
            keep_mask = np.zeros(N, dtype=bool)
            keep_mask[keep_indices] = True
            for _s, chunk in _iter_sequences_chunked(self._dataset_path):
                n = len(chunk)
                chunk_mask = keep_mask[_s : _s + n]
                kept = chunk[chunk_mask]
                nk = len(kept)
                if nk:
                    sequences[write_pos : write_pos + nk] = kept
                    write_pos += nk
                del chunk, kept
        else:
            for _s, chunk in _iter_sequences_chunked(self._dataset_path):
                n = len(chunk)
                sequences[write_pos : write_pos + n] = chunk
                write_pos += n
                del chunk

        assert write_pos == N_kept, f"Streaming read mismatch: wrote {write_pos}, expected {N_kept}"

        self._bundle = DatasetBundle(
            sequences=sequences,
            labels=labels,
            label_ids=label_ids,
            recording_ids=recording_ids,
            metadata=metadata,
        )

        # ── Build datasets; apply augmentation to training split only ─────────
        # Pass the compact sequences array + split indices so each __getitem__
        # slices one (T, F) window on demand — no per-split fancy-index copies.
        if self._augmentation is not None:
            augmenter = GestureAugmenter(self._augmentation)
            self._train_dataset = AugmentedSequenceTensorDataset(
                sequences,
                label_ids[self._train_idx],
                augmenter,
                indices=self._train_idx,
            )
        else:
            self._train_dataset = SequenceTensorDataset(
                sequences, label_ids[self._train_idx], indices=self._train_idx
            )
        self._val_dataset = SequenceTensorDataset(
            sequences, label_ids[self._val_idx], indices=self._val_idx
        )
        self._test_dataset = SequenceTensorDataset(
            sequences, label_ids[self._test_idx], indices=self._test_idx
        )

    def setup_indices_only(
        self,
        label_ids: np.ndarray,
        recording_ids: np.ndarray,
        vocab: list[str] | None = None,
    ) -> None:
        """Compute train/val/test split indices from pre-loaded arrays.

        Identical splitting logic to :meth:`setup` (same seed, stratification,
        and group-split flag), but avoids loading the dataset from disk and never
        creates ``SequenceTensorDataset`` instances.  This is the correct method
        to call from analysis notebooks that have already loaded the sequences
        array into memory — calling :meth:`setup` would reload and materialise
        the full sequences array a second time, exhausting available RAM.

        After calling this method :attr:`train_indices`, :attr:`val_indices`, and
        :attr:`test_indices` are fully operational.  Calls to the dataloader
        methods will raise ``RuntimeError`` (datasets are not created), which is
        the expected behaviour for index-only initialisation.

        Parameters
        ----------
        label_ids:
            Integer label array, shape ``(N,)``.
        recording_ids:
            String / object array of recording identifiers, shape ``(N,)``.
        vocab:
            Full vocabulary list (in label-id order) as stored in the dataset
            metadata.  Required when *exclude_classes* was passed to the
            constructor — the method uses it to identify which label IDs to
            remove before splitting.  Ignored when *exclude_classes* is None.
        """
        self._split.validate()

        # When exclude_classes is active, filter the input arrays and map split
        # results back to original indices so that callers can still index into
        # the full (N,) arrays with the returned indices.
        if self._exclude_classes and vocab:
            filtered_vocab, keep_indices, remapped_ids = self._apply_exclude_classes(
                vocab, label_ids
            )
            self._class_labels_override = filtered_vocab
            filtered_recording_ids = recording_ids[keep_indices]

            # Apply per-class cap BEFORE splitting (same reasoning as setup()).
            if self._max_samples_per_class is not None:
                cap_sub_idx = self._downsample_dataset_indices(remapped_ids)
                remapped_ids = remapped_ids[cap_sub_idx]
                filtered_recording_ids = filtered_recording_ids[cap_sub_idx]
                # Compose: split → cap → exclude-filter → original index space.
                keep_indices = keep_indices[cap_sub_idx]

            unique_recs = np.unique(filtered_recording_ids)
            if self._use_group_split and len(unique_recs) >= 3:
                train_idx, val_idx, test_idx = self._group_split_indices(
                    remapped_ids, filtered_recording_ids
                )
            else:
                train_idx, val_idx, test_idx = self._split_indices(remapped_ids)

            # Map split-relative indices back to original-array indices so
            # callers can do all_labels[_dm.train_indices] on the full array.
            self._train_idx = keep_indices[np.asarray(train_idx)]
            self._val_idx = keep_indices[np.asarray(val_idx)]
            self._test_idx = keep_indices[np.asarray(test_idx)]
            return

        # Apply per-class cap BEFORE splitting (no exclude_classes path).
        if self._max_samples_per_class is not None:
            cap_idx = self._downsample_dataset_indices(label_ids)
            label_ids_split = label_ids[cap_idx]
            rec_ids_split = recording_ids[cap_idx]
        else:
            cap_idx = None
            label_ids_split = label_ids
            rec_ids_split = recording_ids

        unique_recs = np.unique(rec_ids_split)
        if self._use_group_split and len(unique_recs) >= 3:
            train_idx, val_idx, test_idx = self._group_split_indices(label_ids_split, rec_ids_split)
        else:
            train_idx, val_idx, test_idx = self._split_indices(label_ids_split)

        if cap_idx is not None:
            self._train_idx = cap_idx[np.asarray(train_idx)]
            self._val_idx = cap_idx[np.asarray(val_idx)]
            self._test_idx = cap_idx[np.asarray(test_idx)]
        else:
            self._train_idx = np.asarray(train_idx)
            self._val_idx = np.asarray(val_idx)
            self._test_idx = np.asarray(test_idx)

        # Ensure class_labels / num_classes work without a full setup().
        # If vocab was provided use it directly; otherwise fall back to loading
        # the vocabulary from the dataset file (cheap — metadata only, no sequences).
        if not vocab:
            try:
                payload = SequenceDatasetBuilder.load_npz(self._dataset_path)
                vocab = [str(v) for v in (payload["metadata"].get("vocabulary") or [])]
            except Exception:
                pass
        if vocab and self._class_labels_override is None:
            self._class_labels_override = [str(v) for v in vocab]

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

    def _apply_exclude_classes(
        self,
        vocab: list[str],
        label_ids: np.ndarray,
    ) -> tuple[list[str], np.ndarray, np.ndarray]:
        """Remove excluded classes from *label_ids* and remap IDs to be contiguous.

        Parameters
        ----------
        vocab:
            Full class vocabulary in label-id order (index = class ID).
        label_ids:
            Original integer label array, shape ``(N,)``.

        Returns
        -------
        filtered_vocab:
            Vocabulary with excluded classes removed (preserves original order).
        keep_indices:
            Indices into the original (N,) arrays for samples that are kept.
        remapped_ids:
            New contiguous label IDs (0..M-1) for the kept samples, where M is
            ``len(filtered_vocab)``.
        """
        assert self._exclude_classes is not None
        exclude_set = set(self._exclude_classes)

        # IDs that belong to excluded classes.
        exclude_ids = {i for i, cls in enumerate(vocab) if cls in exclude_set}

        # Vocabulary after removal (order preserved).
        filtered_vocab = [cls for cls in vocab if cls not in exclude_set]

        # Build old-id → new-id remap (-1 for excluded classes).
        remap = np.full(len(vocab), -1, dtype=np.int64)
        for new_id, cls in enumerate(filtered_vocab):
            old_id = vocab.index(cls)
            remap[old_id] = new_id

        keep_mask = ~np.isin(label_ids, list(exclude_ids))
        keep_indices = np.where(keep_mask)[0]
        remapped_ids = remap[label_ids[keep_indices]]
        return filtered_vocab, keep_indices, remapped_ids

    def _downsample_dataset_indices(
        self,
        label_ids: np.ndarray,
    ) -> np.ndarray:
        """Return indices that cap each class at ``self._max_samples_per_class``.

        Applied to the **full dataset** before the train/val/test split so that
        the cap is respected in every partition, not only in the training split.
        This prevents majority classes (e.g. point_one/point_two) from
        dominating the validation and test sets while being downsampled in
        training, which would make evaluation metrics misleadingly optimistic for
        those classes.

        The subsampling is reproducible via a seeded RNG derived from
        ``self._seed``.
        """
        assert self._max_samples_per_class is not None
        rng = np.random.default_rng(self._seed)
        keep: list[np.ndarray] = []
        all_idx = np.arange(len(label_ids))
        for cls_id in np.unique(label_ids):
            cls_idx = all_idx[label_ids == cls_id]
            if len(cls_idx) > self._max_samples_per_class:
                cls_idx = rng.choice(cls_idx, size=self._max_samples_per_class, replace=False)
            keep.append(cls_idx)
        return np.sort(np.concatenate(keep))

    def label_distribution(self) -> dict[str, int]:
        if self._bundle is None:
            raise RuntimeError("DataModule not initialized. Call setup() first.")

        labels = self.class_labels
        counts = np.bincount(self._bundle.label_ids, minlength=len(labels))
        return {labels[i]: int(counts[i]) for i in range(len(labels))}

    def training_label_distribution(self) -> dict[str, int]:
        """Return label counts for the TRAINING split only.

        Unlike :meth:`label_distribution` (which counts every sample in the
        full dataset), this method counts only the indices in the training
        split, reflecting any per-class downsampling applied by
        ``max_samples_per_class``.  This is the correct distribution to use
        when computing inverse-frequency class weights for the loss function.
        """
        if self._bundle is None or self._train_idx is None:
            raise RuntimeError("training_label_distribution() requires setup() to be called first.")
        labels = self.class_labels
        train_label_ids = self._bundle.label_ids[self._train_idx]
        counts = np.bincount(train_label_ids, minlength=len(labels))
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
        """Split samples by video so no video leaks across partitions.

        Recording IDs have the format ``"videostem:segNNNN"`` — multiple
        annotated segments (and their sliding windows) per video.  We split on
        the *video stem* (everything before the first ``:``) so that all windows
        from the same video land in exactly one partition.

        Requires at least 3 unique video stems (one per split).  If the temp
        partition ends up with fewer than 2 unique video stems, the val/test
        sub-split falls back to random ``train_test_split``.
        """
        from sklearn.model_selection import GroupShuffleSplit, train_test_split

        # Extract video stems: "videostem:segNNNN" → "videostem".
        # IDs without a colon are used as-is (handles test data and future formats).
        group_ids = np.array(
            [str(r).split(":")[0] if ":" in str(r) else str(r) for r in recording_ids]
        )

        n = len(label_ids)
        all_idx = np.arange(n)
        temp_ratio = self._split.val_ratio + self._split.test_ratio

        gss = GroupShuffleSplit(n_splits=1, test_size=temp_ratio, random_state=self._seed)
        train_idx, temp_idx = next(gss.split(all_idx, label_ids, groups=group_ids))

        temp_group_ids = group_ids[temp_idx]
        unique_temp = np.unique(temp_group_ids)

        if len(unique_temp) >= 2:
            val_share = self._split.val_ratio / temp_ratio
            gss2 = GroupShuffleSplit(
                n_splits=1,
                test_size=1.0 - val_share,
                random_state=self._seed,
            )
            rel_val, rel_test = next(
                gss2.split(temp_idx, label_ids[temp_idx], groups=temp_group_ids)
            )
            val_idx = temp_idx[rel_val]
            test_idx = temp_idx[rel_test]
        else:
            # Only one video in the temp pool — random sub-split is safe here.
            val_share = self._split.val_ratio / temp_ratio
            val_idx, test_idx = train_test_split(
                temp_idx,
                test_size=1.0 - val_share,
                random_state=self._seed,
                shuffle=True,
            )

        return np.asarray(train_idx), np.asarray(val_idx), np.asarray(test_idx)

    def _split_indices(self, label_ids: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        from sklearn.model_selection import train_test_split

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
