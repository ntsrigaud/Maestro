from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np


@dataclass(slots=True)
class LatencyMetrics:
    avg_ms: float
    p95_ms: float
    max_ms: float
    count: int


@dataclass(slots=True)
class LatencyBenchmarkReport:
    frame_capture: LatencyMetrics
    landmark_extraction: LatencyMetrics
    feature_engineering: LatencyMetrics
    sequence_inference: LatencyMetrics
    end_to_end: LatencyMetrics
    throughput_fps: float
    total_frames: int
    total_seconds: float

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


class LatencyBenchmark:
    def __init__(self) -> None:
        self._capture_ms: list[float] = []
        self._extract_ms: list[float] = []
        self._feature_ms: list[float] = []
        self._infer_ms: list[float] = []
        self._e2e_ms: list[float] = []

    def record_capture(self, latency_ms: float) -> None:
        self._capture_ms.append(float(latency_ms))

    def record_extraction(self, latency_ms: float) -> None:
        self._extract_ms.append(float(latency_ms))

    def record_feature(self, latency_ms: float) -> None:
        self._feature_ms.append(float(latency_ms))

    def record_inference(self, latency_ms: float) -> None:
        self._infer_ms.append(float(latency_ms))

    def record_end_to_end(self, latency_ms: float) -> None:
        self._e2e_ms.append(float(latency_ms))

    def report(
        self, *, total_frames: int, total_seconds: float
    ) -> LatencyBenchmarkReport:
        throughput_fps = (
            float(total_frames / total_seconds) if total_seconds > 0 else 0.0
        )
        return LatencyBenchmarkReport(
            frame_capture=self._to_metrics(self._capture_ms),
            landmark_extraction=self._to_metrics(self._extract_ms),
            feature_engineering=self._to_metrics(self._feature_ms),
            sequence_inference=self._to_metrics(self._infer_ms),
            end_to_end=self._to_metrics(self._e2e_ms),
            throughput_fps=throughput_fps,
            total_frames=total_frames,
            total_seconds=float(total_seconds),
        )

    @staticmethod
    def _to_metrics(values: list[float]) -> LatencyMetrics:
        if not values:
            return LatencyMetrics(avg_ms=0.0, p95_ms=0.0, max_ms=0.0, count=0)

        arr = np.asarray(values, dtype=np.float64)
        return LatencyMetrics(
            avg_ms=float(np.mean(arr)),
            p95_ms=float(np.percentile(arr, 95)),
            max_ms=float(np.max(arr)),
            count=int(arr.size),
        )


def write_latency_report(
    report: LatencyBenchmarkReport, output_path: str | Path
) -> Path:
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(report.to_dict(), indent=2), encoding="utf-8")
    return path
