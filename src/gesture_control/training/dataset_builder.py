from __future__ import annotations

import json
from collections import Counter, deque
from collections.abc import Iterable
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from gesture_control.domain.entities import LandmarkPacket
from gesture_control.pipeline.buffering import SlidingWindowSequenceBuffer
from gesture_control.pipeline.feature_engineering import LandmarkFeatureTransformer


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
        window_length: int = 24,
        window_step: int = 3,
        min_valid_ratio: float = 0.8,
        label_strategy: str = "last",
        enable_orientation_normalization: bool = False,
    ) -> None:
        if not vocabulary:
            raise ValueError("vocabulary must not be empty")
        if label_strategy not in {"last", "majority"}:
            raise ValueError("label_strategy must be one of: last, majority")

        self._vocabulary = vocabulary
        self._label_to_id = {label: index for index, label in enumerate(vocabulary)}
        self._window_length = window_length
        self._window_step = window_step
        self._min_valid_ratio = min_valid_ratio
        self._label_strategy = label_strategy
        self._enable_orientation_normalization = enable_orientation_normalization

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

    def save_npz(self, output_path: str | Path) -> Path:
        path = Path(output_path)
        path.parent.mkdir(parents=True, exist_ok=True)

        if not self._samples:
            raise RuntimeError("No sequence samples have been generated")

        sequences = np.asarray(
            [sample.sequence for sample in self._samples],
            dtype=np.float32,
        )
        labels = np.asarray([sample.label for sample in self._samples], dtype=object)
        label_ids = np.asarray([sample.label_id for sample in self._samples], dtype=np.int64)
        recording_ids = np.asarray(
            [sample.recording_id for sample in self._samples],
            dtype=object,
        )
        sample_ids = np.asarray([sample.sample_id for sample in self._samples], dtype=object)
        start_ts = np.asarray(
            [sample.start_timestamp_ms for sample in self._samples],
            dtype=np.int64,
        )
        end_ts = np.asarray(
            [sample.end_timestamp_ms for sample in self._samples],
            dtype=np.int64,
        )
        valid_ratio = np.asarray(
            [sample.valid_ratio for sample in self._samples],
            dtype=np.float32,
        )

        metadata = {
            "feature_dim": int(sequences.shape[2]),
            "window_length": int(sequences.shape[1]),
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
            return {
                "sequences": data["sequences"],
                "labels": data["labels"],
                "label_ids": data["label_ids"],
                "recording_ids": data["recording_ids"],
                "sample_ids": data["sample_ids"],
                "start_timestamp_ms": data["start_timestamp_ms"],
                "end_timestamp_ms": data["end_timestamp_ms"],
                "valid_ratio": data["valid_ratio"],
                "metadata": metadata,
            }

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
