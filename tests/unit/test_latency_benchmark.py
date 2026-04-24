from __future__ import annotations

import json
from pathlib import Path

from maestro.evaluation.benchmark_latency import (
    LatencyBenchmark,
    write_latency_report,
)


def test_latency_benchmark_report_metrics() -> None:
    bench = LatencyBenchmark()
    for value in (10.0, 12.0, 14.0):
        bench.record_capture(value)
        bench.record_extraction(value + 1.0)
        bench.record_feature(value + 2.0)
        bench.record_inference(value + 3.0)
        bench.record_end_to_end(value + 4.0)

    report = bench.report(total_frames=90, total_seconds=3.0)

    assert report.frame_capture.count == 3
    assert report.frame_capture.avg_ms == 12.0
    assert report.sequence_inference.max_ms == 17.0
    assert report.throughput_fps == 30.0


def test_write_latency_report(tmp_path: Path) -> None:
    bench = LatencyBenchmark()
    bench.record_capture(1.0)
    bench.record_extraction(2.0)
    bench.record_feature(3.0)
    bench.record_inference(4.0)
    bench.record_end_to_end(5.0)

    report = bench.report(total_frames=10, total_seconds=1.0)
    output = write_latency_report(report, tmp_path / "latency.json")

    payload = json.loads(output.read_text(encoding="utf-8"))
    assert payload["throughput_fps"] == 10.0
    assert payload["end_to_end"]["max_ms"] == 5.0
