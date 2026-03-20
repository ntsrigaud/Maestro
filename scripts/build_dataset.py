from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import yaml

from gesture_control.training.dataset_builder import (
    LabeledLandmarkFrame,
    SequenceDatasetBuilder,
)
from gesture_control.utils.config_loader import load_runtime_config


def _load_vocabulary(config_dir: Path) -> list[str]:
    gestures_path = config_dir / "gestures" / "vocabulary_v1.yaml"
    if not gestures_path.exists():
        raise FileNotFoundError(f"Vocabulary config not found: {gestures_path}")

    with gestures_path.open("r", encoding="utf-8") as file:
        payload = yaml.safe_load(file) or {}

    gestures = payload.get("gestures", [])
    if not isinstance(gestures, list) or not gestures:
        raise ValueError("Vocabulary config must include a non-empty 'gestures' list")
    return [str(item) for item in gestures]


def _parse_landmark_point(point: Any) -> list[float]:
    if not isinstance(point, list) or len(point) != 3:
        raise ValueError("Each landmark point must be a list with 3 float values")
    return [float(point[0]), float(point[1]), float(point[2])]


def _parse_frame_record(raw: dict[str, Any]) -> LabeledLandmarkFrame:
    landmarks = raw.get("landmarks", [])
    if not isinstance(landmarks, list):
        raise ValueError("landmarks must be a list")

    return LabeledLandmarkFrame(
        recording_id=str(raw["recording_id"]),
        timestamp_ms=int(raw["timestamp_ms"]),
        label=str(raw["label"]),
        landmarks=[_parse_landmark_point(point) for point in landmarks],
        valid=bool(raw.get("valid", True)),
        handedness=raw.get("handedness"),
        detect_conf=float(raw.get("detect_conf", 1.0)),
        track_conf=float(raw.get("track_conf", 1.0)),
    )


def _read_jsonl_frames(input_path: Path) -> list[LabeledLandmarkFrame]:
    frames: list[LabeledLandmarkFrame] = []
    with input_path.open("r", encoding="utf-8") as file:
        for line_no, line in enumerate(file, start=1):
            stripped = line.strip()
            if not stripped:
                continue
            record = json.loads(stripped)
            if not isinstance(record, dict):
                raise ValueError(f"Line {line_no}: JSON record must be an object")
            try:
                frames.append(_parse_frame_record(record))
            except Exception as exc:  # pragma: no cover - includes validation context
                raise ValueError(f"Line {line_no}: invalid record ({exc})") from exc
    return frames


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Build labeled sequence dataset for LSTM training",
    )
    parser.add_argument(
        "--input-jsonl",
        required=True,
        help="Path to labeled landmark frames in JSONL format",
    )
    parser.add_argument(
        "--output-npz",
        required=True,
        help="Output NPZ path for sequence dataset",
    )
    parser.add_argument(
        "--config-dir",
        default="configs",
        help="Configuration directory containing inference and gestures configs",
    )
    parser.add_argument(
        "--label-strategy",
        default="last",
        choices=["last", "majority"],
        help="How sequence labels are aligned from per-frame labels",
    )
    parser.add_argument(
        "--enable-orientation-normalization",
        action="store_true",
        help="Enable orientation normalization in feature transformer",
    )
    args = parser.parse_args()

    input_path = Path(args.input_jsonl)
    output_path = Path(args.output_npz)
    config_dir = Path(args.config_dir)

    if not input_path.exists():
        raise FileNotFoundError(f"Input JSONL not found: {input_path}")

    runtime_config = load_runtime_config(config_dir)
    vocabulary = _load_vocabulary(config_dir)
    frames = _read_jsonl_frames(input_path)

    builder = SequenceDatasetBuilder(
        vocabulary=vocabulary,
        window_length=runtime_config.inference.window_length,
        window_step=runtime_config.inference.window_step,
        min_valid_ratio=runtime_config.inference.min_valid_ratio,
        label_strategy=args.label_strategy,
        enable_orientation_normalization=args.enable_orientation_normalization,
    )
    samples = builder.build_from_frames(frames)
    output_file = builder.save_npz(output_path)

    print("DATASET_BUILD_START")
    print(f"input_frames={len(frames)}")
    print(f"output_samples={len(samples)}")
    print(f"window_length={runtime_config.inference.window_length}")
    print(f"window_step={runtime_config.inference.window_step}")
    print(f"output_npz={output_file}")
    print("DATASET_BUILD_END")


if __name__ == "__main__":
    main()
