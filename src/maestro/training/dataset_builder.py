from __future__ import annotations

import json
import tempfile
from collections import Counter, deque
from collections.abc import Iterable
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from maestro.domain.entities import LabeledGraphFrame, LandmarkPacket
from maestro.pipeline import graph_schema
from maestro.pipeline.buffering import SlidingWindowSequenceBuffer
from maestro.pipeline.feature_engineering import LandmarkFeatureTransformer
from maestro.pipeline.graph_buffering import GraphSequenceBuffer


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
    sequence: list[list[float]]
    start_timestamp_ms: int
    end_timestamp_ms: int
    valid_ratio: float


class SequenceDatasetBuilder:
    """Builds labeled sequence datasets from landmark frames for LSTM training."""

    def __init__(
        self,
        *,
        vocabulary: list[str],
        window_length: int = 32,
        window_step: int = 3,
        min_valid_ratio: float = 0.8,
        label_strategy: str = "last",
        enable_orientation_normalization: bool = False,
        min_trailing_label_purity: float = 0.9,
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

        self._sample_index = 0
        self._samples: list[SequenceSample] = []
        self._active_recording_id: str | None = None

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
                sequence=window.features,
                start_timestamp_ms=window.start_timestamp_ms,
                end_timestamp_ms=window.end_timestamp_ms,
                valid_ratio=window.valid_ratio,
            )
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
        first = self._samples[0]
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
            memmap_path = Path(tmpdir) / "sequences.float32.dat"
            sequences = np.memmap(
                memmap_path,
                dtype=np.float32,
                mode="w+",
                shape=(n_samples, window_length, feature_dim),
            )

            for i, sample in enumerate(self._samples):
                sequences[i] = np.asarray(sample.sequence, dtype=np.float32)
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


# ---------------------------------------------------------------------------
# Graph-schema dataset builder (graph-schema-v1)
# ---------------------------------------------------------------------------


@dataclass(slots=True)
class GraphSequenceSample:
    sample_id: str
    recording_id: str
    label: str
    label_id: int
    sequence: np.ndarray  # (T, N, C)
    valid_masks: np.ndarray  # (T, N)
    start_timestamp_ms: int
    end_timestamp_ms: int
    valid_ratio: float


class GraphSequenceDatasetBuilder:
    """Builds graph-schema-v1 NPZ datasets from ``LabeledGraphFrame`` objects.

    Input frames carry **raw** (pre-normalisation) MediaPipe keypoints.
    Normalisation is applied internally via
    ``graph_schema.normalise_graph_frame()`` so that the coordinate space
    used in the NPZ is identical to what ``HolisticLandmarkExtractor`` emits
    at inference time.

    Output NPZ layout:
      - ``sequences``    : float32 (N, T, num_nodes, num_coords)
      - ``valid_masks``  : bool    (N, T, num_nodes)
      - ``adjacency``    : float32 (num_nodes, num_nodes)
      - ``labels``       : object  (N,)
      - ``label_ids``    : int64   (N,)
      - ``recording_ids``: object  (N,)
      - ``sample_ids``   : object  (N,)
      - ``start_timestamp_ms`` / ``end_timestamp_ms``: int64 (N,)
      - ``valid_ratio``  : float32 (N,)
      - ``metadata_json``: JSON string with graph_schema_version, num_nodes,
                           num_coords, window_length, window_step, vocabulary.
    """

    def __init__(
        self,
        *,
        vocabulary: list[str],
        window_length: int = 32,
        window_step: int = 3,
        min_valid_ratio: float = 0.8,
        label_strategy: str = "last",
        min_trailing_label_purity: float = 0.9,
    ) -> None:
        if not vocabulary:
            raise ValueError("vocabulary must not be empty")
        if label_strategy not in {"last", "majority"}:
            raise ValueError("label_strategy must be one of: last, majority")

        self._vocabulary = vocabulary
        self._label_to_id = {label: idx for idx, label in enumerate(vocabulary)}
        self._window_length = window_length
        self._window_step = window_step
        self._min_valid_ratio = min_valid_ratio
        self._label_strategy = label_strategy
        self._min_trailing_label_purity = min_trailing_label_purity

        self._sample_index = 0
        self._samples: list[GraphSequenceSample] = []
        self._active_recording_id: str | None = None

        self._graph_buffer = GraphSequenceBuffer(
            window_length=window_length,
            window_step=window_step,
            min_valid_ratio=min_valid_ratio,
        )
        self._labels: deque[str] = deque(maxlen=window_length)
        self._timestamps: deque[int] = deque(maxlen=window_length)

    @property
    def samples(self) -> list[GraphSequenceSample]:
        return list(self._samples)

    def add_frame(self, frame: LabeledGraphFrame) -> list[GraphSequenceSample]:
        """Process one ``LabeledGraphFrame`` and emit any ready windows."""
        if frame.label not in self._label_to_id:
            raise ValueError(f"Unknown label '{frame.label}' not found in vocabulary")

        if self._active_recording_id is None:
            self._active_recording_id = frame.recording_id
        elif frame.recording_id != self._active_recording_id:
            self._reset_recording_state(frame.recording_id)

        # Normalise coordinates (same call used by HolisticLandmarkExtractor).
        normalised = graph_schema.normalise_graph_frame(frame.keypoints.astype(np.float32))

        # A frame is considered valid if at least one hand block has detections.
        # The per-node valid_mask provides finer granularity for model masking;
        # frame-level validity is used only for the buffer's valid-ratio gate.
        any_hand_valid = bool(frame.valid_mask[:42].any())

        from maestro.domain.entities import GraphFrame

        gf = GraphFrame(
            nodes=normalised,
            valid_mask=frame.valid_mask,
            timestamp_ms=frame.timestamp_ms,
            valid=any_hand_valid,
        )

        self._graph_buffer.push(gf)
        self._labels.append(frame.label)
        self._timestamps.append(frame.timestamp_ms)

        emitted: list[GraphSequenceSample] = []
        if self._graph_buffer.ready():
            sequences, valid_masks = self._graph_buffer.window()
            labels = list(self._labels)

            selected_label = self._select_label(labels)

            # Trailing-purity guard (last strategy only).
            if self._label_strategy == "last" and self._min_trailing_label_purity > 0.0:
                trailing_n = max(1, round(self._window_length * self._min_trailing_label_purity))
                trailing_labels = labels[-trailing_n:]
                trailing_purity = trailing_labels.count(selected_label) / trailing_n
                if trailing_purity < self._min_trailing_label_purity:
                    return emitted

            ts_list = list(self._timestamps)
            sample = GraphSequenceSample(
                sample_id=f"{frame.recording_id}:{self._sample_index}",
                recording_id=frame.recording_id,
                label=selected_label,
                label_id=self._label_to_id[selected_label],
                sequence=sequences,
                valid_masks=valid_masks,
                start_timestamp_ms=ts_list[0],
                end_timestamp_ms=ts_list[-1],
                valid_ratio=self._graph_buffer.current_valid_ratio(),
            )
            self._sample_index += 1
            self._samples.append(sample)
            emitted.append(sample)

        return emitted

    def build_from_frames(self, frames: Iterable[LabeledGraphFrame]) -> list[GraphSequenceSample]:
        for frame in frames:
            self.add_frame(frame)
        return list(self._samples)

    def save_npz(
        self,
        output_path: str | Path,
        *,
        graph_schema_version: str = graph_schema.GRAPH_SCHEMA_VERSION,
    ) -> Path:
        path = Path(output_path)
        path.parent.mkdir(parents=True, exist_ok=True)

        if not self._samples:
            raise RuntimeError("No sequence samples have been generated")

        sequences = np.stack([s.sequence for s in self._samples], axis=0)  # (N, T, NN, C)
        valid_masks = np.stack([s.valid_masks for s in self._samples], axis=0)  # (N, T, NN)
        labels = np.asarray([s.label for s in self._samples], dtype=object)
        label_ids = np.asarray([s.label_id for s in self._samples], dtype=np.int64)
        recording_ids = np.asarray([s.recording_id for s in self._samples], dtype=object)
        sample_ids = np.asarray([s.sample_id for s in self._samples], dtype=object)
        start_ts = np.asarray([s.start_timestamp_ms for s in self._samples], dtype=np.int64)
        end_ts = np.asarray([s.end_timestamp_ms for s in self._samples], dtype=np.int64)
        valid_ratio = np.asarray([s.valid_ratio for s in self._samples], dtype=np.float32)
        adjacency = graph_schema.build_adjacency().astype(np.float32)

        metadata = {
            "graph_schema_version": graph_schema_version,
            "num_nodes": graph_schema.NUM_NODES,
            "num_coords": graph_schema.NUM_COORDS,
            "window_length": self._window_length,
            "window_step": self._window_step,
            "min_valid_ratio": self._min_valid_ratio,
            "label_strategy": self._label_strategy,
            "vocabulary": self._vocabulary,
        }

        np.savez_compressed(
            path,
            sequences=sequences,
            valid_masks=valid_masks,
            adjacency=adjacency,
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

    def _select_label(self, labels: list[str]) -> str:
        if self._label_strategy == "last":
            return labels[-1]
        counter = Counter(labels)
        return counter.most_common(1)[0][0]

    def _reset_recording_state(self, recording_id: str) -> None:
        self._active_recording_id = recording_id
        self._graph_buffer = GraphSequenceBuffer(
            window_length=self._window_length,
            window_step=self._window_step,
            min_valid_ratio=self._min_valid_ratio,
        )
        self._labels.clear()
        self._timestamps.clear()
