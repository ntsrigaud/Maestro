from __future__ import annotations

import argparse
import statistics
import time
from dataclasses import dataclass
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np
import torch

from maestro.domain.entities import FramePacket
from maestro.infrastructure.mediapipe.hands_extractor import (
    MediaPipeHandsExtractor,
)
from maestro.pipeline.hand_selection import HandSelector

# ---------------------------------------------------------------------------
# Shared result types
# ---------------------------------------------------------------------------


@dataclass(slots=True)
class BenchmarkResult:
    frames: int
    total_seconds: float
    throughput_fps: float
    valid_ratio: float
    mean_extract_ms: float
    p95_extract_ms: float


@dataclass(slots=True)
class STGCNBenchmarkResult:
    """
    Per-phase timing breakdown for the STGCN pipeline.

    Gate values (from Phase 5.3 plan):
      - mean_infer_ms      < 15 ms
      - p95_end_to_end_ms  < 50 ms
      - median_end_to_end_ms < 35 ms
    """

    frames: int
    total_seconds: float
    throughput_fps: float
    valid_ratio: float
    param_count: int
    mean_extract_ms: float
    p95_extract_ms: float
    mean_infer_ms: float
    p95_infer_ms: float
    median_end_to_end_ms: float
    p95_end_to_end_ms: float

    # --- pass/fail gates ---
    def gates_pass(self) -> bool:
        return (
            self.mean_infer_ms < 15.0
            and self.p95_end_to_end_ms < 50.0
            and self.median_end_to_end_ms < 35.0
        )


# ---------------------------------------------------------------------------
# Helper factories
# ---------------------------------------------------------------------------


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


# ---------------------------------------------------------------------------
# Phase 2 (legacy two-stream) benchmark
# ---------------------------------------------------------------------------


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
        hand_selector = HandSelector(
            hand_preference="both", min_handedness_conf=0.7, min_track_conf=0.5
        )
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


# ---------------------------------------------------------------------------
# STGCN-LSTM benchmark
# ---------------------------------------------------------------------------


def _build_synthetic_holistic_packet() -> SimpleNamespace:
    """Return a mock HolisticPacket with synthetic (67, 3) keypoints."""
    from maestro.pipeline.graph_schema import NUM_COORDS, NUM_NODES

    keypoints = np.random.randn(NUM_NODES, NUM_COORDS).astype(np.float32)
    valid_mask = np.ones(NUM_NODES, dtype=bool)
    # Simulate hand not always detected (~20% of frames)
    return SimpleNamespace(keypoints=keypoints, valid_mask=valid_mask, timestamp_ms=0)


def run_stgcn_benchmark(frames: int) -> STGCNBenchmarkResult:
    """
    Deterministic STGCN-LSTM latency benchmark.

    Extraction is mocked (returns synthetic keypoints) so that the benchmark
    is reproducible without a camera or the IPN dataset.  Model inference
    uses the actual STGCN_LSTM forward pass on CPU.

    Phase 5.3 gate values:
      - mean_infer_ms          < 15 ms   (STGCN-LSTM forward pass)
      - median_end_to_end_ms   < 35 ms
      - p95_end_to_end_ms      < 50 ms
    """
    from maestro.domain.entities import GraphFrame
    from maestro.infrastructure.model.stgcn_lstm import STGCN_LSTM
    from maestro.pipeline.graph_buffering import GraphSequenceBuffer
    from maestro.pipeline.graph_schema import (
        NUM_COORDS,
        NUM_NODES,
        WINDOW_LENGTH,
        WINDOW_STEP,
        build_adjacency,
        normalise_adjacency,
    )

    rng = np.random.default_rng(42)

    # Build model on CPU
    adjacency = torch.from_numpy(normalise_adjacency(build_adjacency()).astype(np.float32))
    model = STGCN_LSTM(
        num_nodes=NUM_NODES,
        num_coords=NUM_COORDS,
        adjacency=adjacency,
        stgcn_channels=(64, 128),
        lstm_hidden=128,
        lstm_layers=2,
        num_classes=8,
        dropout=0.0,  # no dropout at inference
    )
    model.eval()
    param_count = sum(p.numel() for p in model.parameters())

    buffer = GraphSequenceBuffer(
        window_length=WINDOW_LENGTH,
        window_step=WINDOW_STEP,
        min_valid_ratio=0.8,
    )

    extract_times_ms: list[float] = []
    infer_times_ms: list[float] = []
    end_to_end_ms: list[float] = []
    valid_count = 0
    # ~80% valid frames pattern
    valid_pattern = [True] * 8 + [False] * 2

    with torch.no_grad():
        start = time.perf_counter()
        for i in range(frames):
            t_frame_start = time.perf_counter()

            # --- Simulated extraction (replaces HolisticLandmarkExtractor.extract) ---
            t0 = time.perf_counter()
            is_valid_frame = valid_pattern[i % len(valid_pattern)]
            keypoints = rng.standard_normal((NUM_NODES, NUM_COORDS)).astype(np.float32)
            valid_mask = np.ones(NUM_NODES, dtype=bool)
            if not is_valid_frame:
                # Simulate no-hand: zero out hand nodes
                valid_mask[:42] = False
            extract_ms = (time.perf_counter() - t0) * 1000.0
            extract_times_ms.append(extract_ms)

            # --- Buffer push ---
            graph_frame = GraphFrame(
                nodes=keypoints,
                valid_mask=valid_mask,
                timestamp_ms=i * 33,
                valid=bool(valid_mask[:42].any()),
            )
            buffer.push(graph_frame)

            # --- STGCN-LSTM inference (only when window is ready) ---
            if buffer.ready():
                sequences, _ = buffer.window()
                x = torch.from_numpy(sequences).unsqueeze(0)  # (1, T, N, C)
                t_infer = time.perf_counter()
                _logits = model(x)
                infer_ms = (time.perf_counter() - t_infer) * 1000.0
                infer_times_ms.append(infer_ms)
                valid_count += 1

            end_to_end_ms.append((time.perf_counter() - t_frame_start) * 1000.0)

        total = time.perf_counter() - start

    def _p95(lst: list[float]) -> float:
        if not lst:
            return 0.0
        s = sorted(lst)
        return s[int(0.95 * (len(s) - 1))]

    throughput = frames / total if total > 0 else 0.0
    valid_ratio = valid_count / frames if frames > 0 else 0.0

    return STGCNBenchmarkResult(
        frames=frames,
        total_seconds=total,
        throughput_fps=throughput,
        valid_ratio=valid_ratio,
        param_count=param_count,
        mean_extract_ms=statistics.fmean(extract_times_ms) if extract_times_ms else 0.0,
        p95_extract_ms=_p95(extract_times_ms),
        mean_infer_ms=statistics.fmean(infer_times_ms) if infer_times_ms else 0.0,
        p95_infer_ms=_p95(infer_times_ms),
        median_end_to_end_ms=statistics.median(end_to_end_ms) if end_to_end_ms else 0.0,
        p95_end_to_end_ms=_p95(end_to_end_ms),
    )


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Maestro latency benchmark. Supports legacy Phase 2 pipeline "
        "and the new STGCN-LSTM pipeline.",
    )
    parser.add_argument(
        "--model",
        type=str,
        default="two_stream",
        choices=["two_stream", "stgcn_lstm_v1"],
        help="Pipeline to benchmark (default: two_stream — legacy Phase 2)",
    )
    parser.add_argument(
        "--frames",
        type=int,
        default=500,
        help="Number of synthetic frames to process",
    )
    args = parser.parse_args()

    if args.model == "stgcn_lstm_v1":
        result = run_stgcn_benchmark(args.frames)
        print("STGCN_BENCHMARK_START")
        print("model=stgcn_lstm_v1")
        print(f"frames={result.frames}")
        print(f"total_seconds={result.total_seconds:.6f}")
        print(f"throughput_fps={result.throughput_fps:.2f}")
        print(f"valid_ratio={result.valid_ratio:.3f}")
        print(f"param_count={result.param_count}")
        print(f"mean_extract_ms={result.mean_extract_ms:.4f}")
        print(f"p95_extract_ms={result.p95_extract_ms:.4f}")
        print(f"mean_infer_ms={result.mean_infer_ms:.4f}")
        print(f"p95_infer_ms={result.p95_infer_ms:.4f}")
        print(f"median_end_to_end_ms={result.median_end_to_end_ms:.4f}")
        print(f"p95_end_to_end_ms={result.p95_end_to_end_ms:.4f}")
        status = "PASS" if result.gates_pass() else "FAIL"
        print(f"gates={status}")
        print("STGCN_BENCHMARK_END")
    else:
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
