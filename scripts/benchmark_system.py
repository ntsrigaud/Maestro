from __future__ import annotations

import argparse
import statistics
import time
from dataclasses import dataclass
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np

from maestro.domain.entities import FramePacket
from maestro.infrastructure.mediapipe.hands_extractor import (
    MediaPipeHandsExtractor,
)
from maestro.pipeline.hand_selection import HandSelector


@dataclass(slots=True)
class BenchmarkResult:
    frames: int
    total_seconds: float
    throughput_fps: float
    valid_ratio: float
    mean_extract_ms: float
    p95_extract_ms: float


def _frame_packet(frame_id: int) -> FramePacket:
    return FramePacket(
        frame=np.zeros((720, 1280, 3), dtype=np.uint8),
        timestamp_ms=frame_id * 33,
        frame_id=frame_id,
    )


def _build_hands_response(mode: str) -> SimpleNamespace:
    if mode == "no_hand":
        return SimpleNamespace(multi_hand_landmarks=None, multi_handedness=None)

    score = 0.95 if mode == "valid" else 0.2
    classification = SimpleNamespace(label="Right", score=score)
    handedness = SimpleNamespace(classification=[classification])
    landmarks = [SimpleNamespace(x=0.4, y=0.3, z=0.1) for _ in range(21)]
    hand_landmarks = SimpleNamespace(landmark=landmarks)
    return SimpleNamespace(
        multi_handedness=[handedness],
        multi_hand_landmarks=[hand_landmarks],
    )


def run_phase2_benchmark(frames: int) -> BenchmarkResult:
    pattern = ["valid"] * 7 + ["no_hand"] * 2 + ["low_conf"]
    hands_obj = SimpleNamespace()
    frame_times_ms: list[float] = []
    valid_count = 0
    call_count = 0

    def process(_rgb_frame: np.ndarray) -> SimpleNamespace:
        nonlocal call_count
        mode = pattern[call_count % len(pattern)]
        call_count += 1
        return _build_hands_response(mode)

    hands_obj.process = process
    hands_obj.close = lambda: None

    with patch(
        "maestro.infrastructure.mediapipe.hands_extractor.mp.solutions.hands.Hands",
        return_value=hands_obj,
    ):
        extractor = MediaPipeHandsExtractor(
            min_detection_confidence=0.7,
            min_tracking_confidence=0.5,
        )
        hand_selector = HandSelector(hand_preference="both", min_handedness_conf=0.7, min_track_conf=0.5)
        start = time.perf_counter()
        for i in range(frames):
            packet = _frame_packet(i)
            t0 = time.perf_counter()
            result = hand_selector.select(extractor.extract(packet))
            frame_times_ms.append((time.perf_counter() - t0) * 1000.0)
            if result.valid:
                valid_count += 1
        total = time.perf_counter() - start
        extractor.close()

    throughput = frames / total if total > 0 else 0.0
    valid_ratio = valid_count / frames if frames > 0 else 0.0
    mean_ms = statistics.fmean(frame_times_ms) if frame_times_ms else 0.0
    if frame_times_ms:
        sorted_ms = sorted(frame_times_ms)
        p95_index = int(0.95 * (len(sorted_ms) - 1))
        p95_ms = sorted_ms[p95_index]
    else:
        p95_ms = 0.0

    return BenchmarkResult(
        frames=frames,
        total_seconds=total,
        throughput_fps=throughput,
        valid_ratio=valid_ratio,
        mean_extract_ms=mean_ms,
        p95_extract_ms=p95_ms,
    )


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Phase 2 ingestion benchmark (deterministic synthetic workload)",
    )
    parser.add_argument(
        "--frames",
        type=int,
        default=2000,
        help="Number of synthetic frames to process",
    )
    args = parser.parse_args()

    result = run_phase2_benchmark(args.frames)
    print("PHASE2_BENCHMARK_START")
    print(f"frames={result.frames}")
    print(f"total_seconds={result.total_seconds:.6f}")
    print(f"throughput_fps={result.throughput_fps:.2f}")
    print(f"valid_ratio={result.valid_ratio:.3f}")
    print(f"mean_extract_ms={result.mean_extract_ms:.4f}")
    print(f"p95_extract_ms={result.p95_extract_ms:.4f}")
    print("PHASE2_BENCHMARK_END")


if __name__ == "__main__":
    main()
