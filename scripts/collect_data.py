from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Any

from gesture_control.app.bootstrap import build_container
from gesture_control.domain.entities import LandmarkPacket


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Collect labeled hand landmarks from webcam into JSONL",
    )
    parser.add_argument(
        "--config",
        default="configs",
        help="Path to config directory containing app.yaml and inference.yaml",
    )
    parser.add_argument(
        "--label",
        required=True,
        help="Gesture label to assign to captured frames (must exist in vocabulary)",
    )
    parser.add_argument(
        "--recording-id",
        required=True,
        help="Unique recording identifier (e.g., rec-swipe-right-001)",
    )
    parser.add_argument(
        "--output-jsonl",
        required=True,
        help="Output JSONL file path (append mode)",
    )
    parser.add_argument(
        "--max-frames",
        type=int,
        default=0,
        help="Stop after this many frames; 0 means unlimited",
    )
    parser.add_argument(
        "--max-seconds",
        type=float,
        default=0.0,
        help="Stop after this many seconds; 0 means unlimited",
    )
    parser.add_argument(
        "--include-invalid",
        action="store_true",
        help="Also store frames where hand extraction is invalid",
    )
    return parser.parse_args()


def _packet_to_json_record(
    packet: LandmarkPacket,
    *,
    recording_id: str,
    label: str,
    frame_index: int,
) -> dict[str, Any]:
    return {
        "recording_id": recording_id,
        "frame_index": frame_index,
        "timestamp_ms": int(packet.timestamp_ms),
        "label": label,
        "landmarks": [[float(x), float(y), float(z)] for x, y, z in packet.landmarks],
        "valid": bool(packet.valid),
        "handedness": packet.handedness,
        "detect_conf": float(packet.detect_conf),
        "track_conf": float(packet.track_conf),
    }


def main() -> None:
    args = _parse_args()
    container = build_container(args.config)
    output_path = Path(args.output_jsonl)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    print("DATA_COLLECTION_START")
    print(f"recording_id={args.recording_id}")
    print(f"label={args.label}")
    print(f"output_jsonl={output_path}")
    print("Press Ctrl+C to stop recording.")

    frame_index = 0
    written_records = 0
    started_at = time.monotonic()

    try:
        with output_path.open("a", encoding="utf-8") as file:
            while True:
                frame = container.frame_source.read()
                packet = container.landmark_extractor.extract(frame)

                should_write = packet.valid or args.include_invalid
                if should_write:
                    record = _packet_to_json_record(
                        packet,
                        recording_id=args.recording_id,
                        label=args.label,
                        frame_index=frame_index,
                    )
                    file.write(json.dumps(record) + "\n")
                    written_records += 1

                frame_index += 1

                if args.max_frames > 0 and frame_index >= args.max_frames:
                    break

                if (
                    args.max_seconds > 0
                    and (time.monotonic() - started_at) >= args.max_seconds
                ):
                    break
    except KeyboardInterrupt:
        print("Keyboard interrupt received; stopping collection.")
    finally:
        container.frame_source.close()
        container.landmark_extractor.close()

    print("DATA_COLLECTION_END")
    print(f"frames_seen={frame_index}")
    print(f"records_written={written_records}")
    print(f"elapsed_seconds={time.monotonic() - started_at:.2f}")


if __name__ == "__main__":
    main()
