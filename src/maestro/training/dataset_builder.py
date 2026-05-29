from __future__ import annotations

import json
import os
import tempfile
from collections import Counter, deque
from collections.abc import Iterable
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from maestro.domain.entities import LandmarkPacket
from maestro.pipeline.buffering import SlidingWindowSequenceBuffer
from maestro.pipeline.feature_engineering import LandmarkFeatureTransformer


@dataclass(slots=True)
class LabeledLandmarkFrame:
    recording_id: str
    timestamp_ms: int
    label: str
    landmarks: list[list[float]]
    valid: bool = True
    handedness: str | None = "Right"
    detect_conf: float = 1.0
    track_conf: float = 1.0


@dataclass(slots=True)
class SequenceSample:
    sample_id: str
    recording_id: str
    label: str
    label_id: int
    sequence: np.ndarray | None
    start_timestamp_ms: int
    end_timestamp_ms: int
    valid_ratio: float


class SequenceDatasetBuilder:
    """Builds labeled sequence datasets from landmark frames for LSTM training."""

    def __init__(
        self,
        *,
        vocabulary: list[str],
        window_length: int = 16,
        window_step: int = 3,
        min_valid_ratio: float = 0.8,
        label_strategy: str = "last",
        enable_orientation_normalization: bool = False,
        min_trailing_label_purity: float = 0.9,
        spill_sequences_to_disk: bool = False,
    ) -> None:
        if not vocabulary:
            raise ValueError("vocabulary must not be empty")
        if label_strategy not in {"last", "majority"}:
            raise ValueError("label_strategy must be one of: last, majority")
        if not (0.0 <= min_trailing_label_purity <= 1.0):
            raise ValueError("min_trailing_label_purity must be in [0.0, 1.0]")

        self._vocabulary = vocabulary
        self._label_to_id = {label: index for index, label in enumerate(vocabulary)}
        self._window_length = window_length
        self._window_step = window_step
        self._min_valid_ratio = min_valid_ratio
        self._label_strategy = label_strategy
        self._enable_orientation_normalization = enable_orientation_normalization
        self._min_trailing_label_purity = min_trailing_label_purity
        self._spill_sequences_to_disk = spill_sequences_to_disk

        self._sample_index = 0
        self._samples: list[SequenceSample] = []
        self._active_recording_id: str | None = None
        self._sequence_store_dir: tempfile.TemporaryDirectory[str] | None = None
        self._sequence_store: np.memmap | None = None
        self._sequence_store_path: Path | None = None
        self._sequence_store_capacity = 0
        self._sequence_shape: tuple[int, int] | None = None
        self._sequence_count = 0

        self._transformer = LandmarkFeatureTransformer(
            enable_orientation_normalization=enable_orientation_normalization
        )
        self._sequence_buffer = SlidingWindowSequenceBuffer(
            window_length=window_length,
            window_step=window_step,
            min_valid_ratio=min_valid_ratio,
        )
        self._labels: deque[str] = deque(maxlen=window_length)

    @property
    def samples(self) -> list[SequenceSample]:
        return list(self._samples)

    def add_frame(self, frame: LabeledLandmarkFrame) -> list[SequenceSample]:
        if frame.label not in self._label_to_id:
            raise ValueError(f"Unknown label '{frame.label}' not found in vocabulary")

        if self._active_recording_id is None:
            self._active_recording_id = frame.recording_id
        elif frame.recording_id != self._active_recording_id:
            self._reset_recording_state(frame.recording_id)

        packet = LandmarkPacket(
            landmarks=frame.landmarks,
            handedness=frame.handedness,
            detect_conf=frame.detect_conf,
            track_conf=frame.track_conf,
            timestamp_ms=frame.timestamp_ms,
            valid=frame.valid,
        )
        feature = self._transformer.transform(packet)
        self._sequence_buffer.push(feature)
        self._labels.append(frame.label)

        emitted: list[SequenceSample] = []
        if self._sequence_buffer.ready():
            window = self._sequence_buffer.window()
            labels = list(self._labels)
            if len(labels) != self._window_length:
                raise RuntimeError("Label/window alignment mismatch")

            selected_label = self._select_label(labels)

            # Trailing-purity guard: only meaningful for the "last" strategy where
            # the assigned label equals the final frame's label.  Reject windows
            # whose trailing frames are predominantly a different class (noisy
            # boundary samples).  For "majority" strategy the label is chosen by
            # vote, so trailing purity is not the right quality signal.
            if self._label_strategy == "last" and self._min_trailing_label_purity > 0.0:
                trailing_n = max(1, round(self._window_length * self._min_trailing_label_purity))
                trailing_labels = labels[-trailing_n:]
                trailing_purity = trailing_labels.count(selected_label) / trailing_n
                if trailing_purity < self._min_trailing_label_purity:
                    return emitted

            sample = SequenceSample(
                sample_id=f"{frame.recording_id}:{self._sample_index}",
                recording_id=frame.recording_id,
                label=selected_label,
                label_id=self._label_to_id[selected_label],
                sequence=None
                if self._spill_sequences_to_disk
                else np.asarray(window.features, dtype=np.float32).copy(),
                start_timestamp_ms=window.start_timestamp_ms,
                end_timestamp_ms=window.end_timestamp_ms,
                valid_ratio=window.valid_ratio,
            )
            if self._spill_sequences_to_disk:
                self._store_sequence(window.features)
            self._sample_index += 1
            self._samples.append(sample)
            emitted.append(sample)

        return emitted

    def build_from_frames(self, frames: Iterable[LabeledLandmarkFrame]) -> list[SequenceSample]:
        for frame in frames:
            self.add_frame(frame)
        return list(self._samples)

    def save_npz(
        self,
        output_path: str | Path,
        *,
        feature_schema_version: str = "feature-schema-v1",
    ) -> Path:
        path = Path(output_path)
        path.parent.mkdir(parents=True, exist_ok=True)

        if not self._samples:
            raise RuntimeError("No sequence samples have been generated")

        n_samples = len(self._samples)
        if self._spill_sequences_to_disk:
            if self._sequence_shape is None or self._sequence_store is None:
                raise RuntimeError("Sequence spill store was not initialised")
            window_length, feature_dim = self._sequence_shape
        else:
            first = self._samples[0]
            if first.sequence is None:
                raise RuntimeError("Sample sequence is missing")
            window_length = len(first.sequence)
            feature_dim = len(first.sequence[0]) if window_length > 0 else 0

        labels = np.empty(n_samples, dtype=object)
        label_ids = np.empty(n_samples, dtype=np.int64)
        recording_ids = np.empty(n_samples, dtype=object)
        sample_ids = np.empty(n_samples, dtype=object)
        start_ts = np.empty(n_samples, dtype=np.int64)
        end_ts = np.empty(n_samples, dtype=np.int64)
        valid_ratio = np.empty(n_samples, dtype=np.float32)

        # Use a disk-backed buffer to avoid a second full in-memory copy of
        # sequences during save for large datasets.
        with tempfile.TemporaryDirectory(prefix="maestro_npz_") as tmpdir:
            if self._spill_sequences_to_disk:
                sequences = self._sequence_store[:n_samples]
            else:
                memmap_path = Path(tmpdir) / "sequences.float32.dat"
                sequences = np.memmap(
                    memmap_path,
                    dtype=np.float32,
                    mode="w+",
                    shape=(n_samples, window_length, feature_dim),
                )

                for i, sample in enumerate(self._samples):
                    if sample.sequence is None:
                        raise RuntimeError("Sample sequence is missing")
                    sequences[i] = np.asarray(sample.sequence, dtype=np.float32)

            for i, sample in enumerate(self._samples):
                labels[i] = sample.label
                label_ids[i] = sample.label_id
                recording_ids[i] = sample.recording_id
                sample_ids[i] = sample.sample_id
                start_ts[i] = sample.start_timestamp_ms
                end_ts[i] = sample.end_timestamp_ms
                valid_ratio[i] = sample.valid_ratio

            sequences.flush()

            metadata = {
                "feature_schema_version": feature_schema_version,
                "feature_dim": int(feature_dim),
                "window_length": int(window_length),
                "window_step": self._window_step,
                "min_valid_ratio": self._min_valid_ratio,
                "label_strategy": self._label_strategy,
                "enable_orientation_normalization": self._enable_orientation_normalization,
                "vocabulary": self._vocabulary,
            }

            np.savez_compressed(
                path,
                sequences=sequences,
                labels=labels,
                label_ids=label_ids,
                recording_ids=recording_ids,
                sample_ids=sample_ids,
                start_timestamp_ms=start_ts,
                end_timestamp_ms=end_ts,
                valid_ratio=valid_ratio,
                metadata_json=np.asarray(json.dumps(metadata), dtype=object),
            )
        return path

    def _store_sequence(self, sequence: np.ndarray) -> None:
        if not self._spill_sequences_to_disk:
            return

        if sequence.ndim != 2:
            raise ValueError(f"Expected a 2D sequence array, got shape {sequence.shape}")

        if self._sequence_shape is None:
            self._sequence_shape = (int(sequence.shape[0]), int(sequence.shape[1]))

        if self._sequence_store_dir is None:
            self._sequence_store_dir = tempfile.TemporaryDirectory(prefix="maestro_seq_")

        if self._sequence_store is None:
            self._sequence_store_capacity = max(1024, self._sequence_count + 1)
            self._sequence_store_path = (
                Path(self._sequence_store_dir.name) / "sequences.float32.dat"
            )
            self._sequence_store = np.memmap(
                self._sequence_store_path,
                dtype=np.float32,
                mode="w+",
                shape=(self._sequence_store_capacity, *self._sequence_shape),
            )
        elif self._sequence_count >= self._sequence_store_capacity:
            old_store = self._sequence_store
            old_path = self._sequence_store_path
            old_capacity = self._sequence_store_capacity
            self._sequence_store_capacity = max(old_capacity * 2, self._sequence_count + 1)
            self._sequence_store_path = Path(self._sequence_store_dir.name) / (
                f"sequences_{self._sequence_store_capacity}.dat"
            )
            self._sequence_store = np.memmap(
                self._sequence_store_path,
                dtype=np.float32,
                mode="w+",
                shape=(self._sequence_store_capacity, *self._sequence_shape),
            )
            self._sequence_store[: self._sequence_count] = old_store[: self._sequence_count]
            old_mmap = getattr(old_store, "_mmap", None)
            if old_mmap is not None:
                old_mmap.close()
            if old_path is not None:
                try:
                    os.remove(old_path)
                except FileNotFoundError:
                    pass

        assert self._sequence_store is not None
        self._sequence_store[self._sequence_count] = np.asarray(sequence, dtype=np.float32)
        self._sequence_count += 1

    @staticmethod
    def load_npz(dataset_path: str | Path) -> dict[str, Any]:
        path = Path(dataset_path)
        with np.load(path, allow_pickle=True) as data:
            metadata = json.loads(str(data["metadata_json"].item()))
            files = set(data.files)
            # ── sequences is intentionally NOT loaded here ─────────────────────
            # The uncompressed sequences array for the v6 dataset is 6–7 GB.
            # Loading it via np.load would decompress the full array into RAM,
            # then GestureDataModule.setup() would make an additional filtered
            # copy — together exceeding available RAM and triggering an OOM kill.
            # setup() streams sequences directly from the ZIP using
            # _iter_sequences_chunked(); we return only the shape for callers
            # that need dimension info without touching the data.
            seq_shape = None
            if "sequences" in files:
                seq_shape = data["sequences"].shape  # numpy reads shape from header, not data
            n = seq_shape[0] if seq_shape is not None else 0
            sample_ids = data["sample_ids"] if "sample_ids" in files else np.empty(n, dtype=object)
            start_ts = (
                data["start_timestamp_ms"]
                if "start_timestamp_ms" in files
                else np.zeros(n, dtype=np.int64)
            )
            end_ts = (
                data["end_timestamp_ms"]
                if "end_timestamp_ms" in files
                else np.zeros(n, dtype=np.int64)
            )
            result: dict[str, Any] = {
                "sequences": None,  # intentionally omitted — stream via _iter_sequences_chunked
                "sequences_shape": seq_shape,
                "labels": data["labels"],
                "label_ids": data["label_ids"],
                "recording_ids": data["recording_ids"],
                "sample_ids": sample_ids,
                "start_timestamp_ms": start_ts,
                "end_timestamp_ms": end_ts,
                "valid_ratio": (
                    data["valid_ratio"] if "valid_ratio" in files else np.ones(n, dtype=np.float32)
                ),
                "metadata": metadata,
            }
            # Graph-schema NPZ extras (sequences.ndim == 4)
            if seq_shape is not None and len(seq_shape) == 4:
                result["adjacency"] = data["adjacency"] if "adjacency" in files else None
                result["valid_masks"] = data["valid_masks"] if "valid_masks" in files else None
            return result

    def _select_label(self, labels: list[str]) -> str:
        if self._label_strategy == "last":
            return labels[-1]

        counter = Counter(labels)
        return counter.most_common(1)[0][0]

    def _reset_recording_state(self, recording_id: str) -> None:
        self._active_recording_id = recording_id
        self._transformer = LandmarkFeatureTransformer(
            enable_orientation_normalization=self._enable_orientation_normalization
        )
        self._sequence_buffer = SlidingWindowSequenceBuffer(
            window_length=self._window_length,
            window_step=self._window_step,
            min_valid_ratio=self._min_valid_ratio,
        )
        self._labels.clear()
