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
    parser.add_argument(
        "--interactive",
        action="store_true",
        help=(
            "Enable interactive hold-to-record mode. "
            "Opens a live camera preview; press SPACE to start/stop each "
            "individual gesture recording; press Q or ESC to quit. "
            "In this mode --recording-id is used as a prefix and an "
            "auto-incrementing counter is appended for each new recording "
            "(e.g. rec-swipe-right-001, rec-swipe-right-002, ...). "
            "--max-frames caps the length of each individual recording."
        ),
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


def _finish_recording(recording_id: str, frame_count: int) -> None:
    """Print a summary line and warn when a recording is too short."""
    print(f"RECORDING_COMPLETE recording_id={recording_id} frames={frame_count}")
    if frame_count < 24:  # one full LSTM window requires 24 frames
        print(
            f"WARNING: Recording has only {frame_count} frames. "
            "At least 24 are needed to produce one sequence window. "
            "Consider discarding this recording."
        )


def _run_interactive_collection(
    args: argparse.Namespace,
    container: Any,
    output_path: Path,
) -> tuple[int, int]:
    """Interactive recording loop: SPACE starts/stops each gesture segment.

    Each toggle of SPACE creates a new recording with a unique ID derived
    from ``--recording-id`` as prefix plus a zero-padded counter suffix.
    Returns ``(recordings_completed, total_records_written)``.
    """
    import cv2  # deferred: only needed in interactive mode

    recording_prefix = args.recording_id
    is_recording = False
    recording_count = 0
    current_recording_id: str | None = None
    recording_frame_count = 0
    total_written = 0
    window_name = f"Gesture Collection \u2014 {args.label}"

    with output_path.open("a", encoding="utf-8") as fout:
        while True:
            frame_pkt = container.frame_source.read()
            packet = container.landmark_extractor.extract(frame_pkt)

            # Mirror display for natural selfie orientation (does not affect
            # the landmark data, which is always raw-camera-coordinates).
            display = cv2.flip(frame_pkt.frame.copy(), 1)
            h, w = display.shape[:2]

            if is_recording:
                cv2.rectangle(display, (0, 0), (w - 1, h - 1), (0, 0, 220), 5)
                cv2.putText(
                    display,
                    f"REC  {current_recording_id}  [{recording_frame_count} frames]",
                    (12, 38),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.85,
                    (0, 0, 255),
                    2,
                )
            else:
                next_id = f"{recording_prefix}-{recording_count + 1:03d}"
                cv2.rectangle(display, (0, 0), (w - 1, h - 1), (0, 200, 0), 3)
                cv2.putText(
                    display,
                    f"IDLE  Next: {next_id}",
                    (12, 38),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.85,
                    (0, 200, 0),
                    2,
                )

            cv2.putText(
                display,
                (
                    f"Label: {args.label}   "
                    f"Recordings: {recording_count}   "
                    f"Written: {total_written}"
                ),
                (10, h - 35),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.52,
                (220, 220, 100),
                1,
            )
            cv2.putText(
                display,
                "SPACE = start / stop recording    Q / ESC = quit",
                (10, h - 12),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.48,
                (190, 190, 190),
                1,
            )

            cv2.imshow(window_name, display)

            # Non-blocking 1 ms poll for key input.
            key = cv2.waitKey(1) & 0xFF
            if key in (ord("q"), ord("Q"), 27):  # q, Q, or ESC
                if is_recording:
                    _finish_recording(current_recording_id, recording_frame_count)
                break
            elif key == ord(" "):
                if is_recording:
                    is_recording = False
                    _finish_recording(current_recording_id, recording_frame_count)
                else:
                    recording_count += 1
                    current_recording_id = (
                        f"{recording_prefix}-{recording_count:03d}"
                    )
                    is_recording = True
                    recording_frame_count = 0
                    print(f"RECORDING_START recording_id={current_recording_id}")

            # Write validated frame when a recording is active.
            if is_recording:
                should_write = packet.valid or args.include_invalid
                if should_write:
                    record = _packet_to_json_record(
                        packet,
                        recording_id=current_recording_id,
                        label=args.label,
                        frame_index=recording_frame_count,
                    )
                    fout.write(json.dumps(record) + "\n")
                    total_written += 1
                recording_frame_count += 1

                # Safety cap: auto-stop when max_frames is reached.
                if args.max_frames > 0 and recording_frame_count >= args.max_frames:
                    is_recording = False
                    _finish_recording(current_recording_id, recording_frame_count)

    cv2.destroyAllWindows()
    return recording_count, total_written


def main() -> None:
    args = _parse_args()
    container = build_container(args.config)
    output_path = Path(args.output_jsonl)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    if args.interactive:
        print("INTERACTIVE_COLLECTION_START")
        print(f"recording_id_prefix={args.recording_id}")
        print(f"label={args.label}")
        print(f"output_jsonl={output_path}")
        try:
            recordings, written = _run_interactive_collection(
                args, container, output_path
            )
        finally:
            container.frame_source.close()
            container.landmark_extractor.close()
        print("INTERACTIVE_COLLECTION_END")
        print(f"recordings_completed={recordings}")
        print(f"records_written={written}")
        return

    # ------------------------------------------------------------------ #
    # Non-interactive batch collection (original behaviour preserved)     #
    # ------------------------------------------------------------------ #
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
