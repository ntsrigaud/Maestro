#!/usr/bin/env python3
"""
Example demonstrating timestamp precision and usage in the Input Manager.
Shows how nanosecond timestamps support temporal analysis for gesture recognition.
"""

import sys
import time

# Force fresh import - clear any cached modules
if "input_manager" in sys.modules:
    del sys.modules["input_manager"]

from input_manager import InputManager, FrameData
from shared.logger import get_logger

logger = get_logger(__name__)


# Helper functions for compatibility if methods not available
def get_timestamp_ms(frame: FrameData) -> float:
    """Get timestamp in milliseconds."""
    if hasattr(frame, "timestamp_ms"):
        return frame.timestamp_ms()
    return frame.timestamp / 1_000_000


def get_time_delta_ms(frame1: FrameData, frame2: FrameData) -> float:
    """Calculate time difference in milliseconds."""
    if hasattr(FrameData, "time_delta_ms"):
        return FrameData.time_delta_ms(frame1, frame2)
    return (frame2.timestamp - frame1.timestamp) / 1_000_000


def get_fps(frame1: FrameData, frame2: FrameData) -> float:
    """Calculate instantaneous FPS."""
    if hasattr(FrameData, "calculate_fps"):
        return FrameData.calculate_fps(frame1, frame2)
    delta_ns = frame2.timestamp - frame1.timestamp
    return 1_000_000_000 / delta_ns if delta_ns > 0 else 0.0


def demonstrate_timestamp_precision():
    """Show timestamp precision and conversion."""
    print("=" * 70)
    print("TIMESTAMP PRECISION DEMONSTRATION")
    print("=" * 70)

    # Current timestamp
    now_ns = time.time_ns()
    print(f"\nCurrent time (nanoseconds): {now_ns:,}")
    print(f"Current time (milliseconds): {now_ns / 1_000_000:,.2f}")
    print(f"Current time (seconds): {now_ns / 1_000_000_000:,.3f}")

    # Frame timing at different FPS
    print("\n" + "-" * 70)
    print("Frame intervals at different FPS:")
    print("-" * 70)
    for fps in [30, 60, 120]:
        interval_ns = 1_000_000_000 / fps
        interval_ms = interval_ns / 1_000_000
        print(f"{fps:3d} FPS: {interval_ns:12,.0f} ns  ({interval_ms:6.2f} ms)")


def demonstrate_frame_timing():
    """Show precise frame timing measurements."""
    print("\n" + "=" * 70)
    print("FRAME TIMING DEMONSTRATION")
    print("=" * 70)

    manager = InputManager(camera_index=0)

    if not manager.start_capture():
        print("Error: Could not open camera")
        return

    print("\nStarting continuous capture...")
    if not manager.start_continuous_capture():
        print("Error: Could not start continuous capture")
        manager.stop_capture()
        return

    # Collect frames
    frames = []
    print("\nCapturing 10 frames...")
    for i in range(10):
        frame = manager.get_latest_frame()
        if frame:
            frames.append(frame)
            print(
                f"  Frame {i+1}: timestamp={get_timestamp_ms(frame):,.2f}ms, "
                f"frame_num={frame.frame_number}"
            )
        time.sleep(0.033)  # ~30 FPS

    # Analyze timing
    if len(frames) >= 2:
        print("\n" + "-" * 70)
        print("Frame-to-frame timing analysis:")
        print("-" * 70)
        total_time = 0
        for i in range(len(frames) - 1):
            delta_ms = get_time_delta_ms(frames[i], frames[i + 1])
            fps = get_fps(frames[i], frames[i + 1])
            total_time += delta_ms
            print(f"  Frame {i} -> {i+1}: {delta_ms:6.2f} ms  ({fps:5.1f} FPS)")

        avg_time = total_time / (len(frames) - 1)
        avg_fps = 1000 / avg_time
        print(f"\n  Average: {avg_time:6.2f} ms/frame  ({avg_fps:5.1f} FPS)")

    manager.stop_continuous_capture()
    manager.stop_capture()


def demonstrate_sequence_timing():
    """Show timing for LSTM sequence processing."""
    print("\n" + "=" * 70)
    print("LSTM SEQUENCE TIMING DEMONSTRATION")
    print("=" * 70)

    manager = InputManager(camera_index=0)

    if not manager.start_capture():
        print("Error: Could not open camera")
        return

    if not manager.start_continuous_capture():
        print("Error: Could not start continuous capture")
        manager.stop_capture()
        return

    # Simulate collecting a sequence for dynamic gesture (16 frames)
    sequence_length = 16
    print(f"\nCollecting {sequence_length}-frame sequence for dynamic gesture...")
    sequence = []

    start_time = time.time_ns()
    while len(sequence) < sequence_length:
        frame = manager.get_latest_frame()
        if frame and (not sequence or frame.timestamp != sequence[-1].timestamp):
            sequence.append(frame)
            print(f"  Frame {len(sequence)}/{sequence_length} captured")
        time.sleep(0.016)  # ~60 FPS

    end_time = time.time_ns()
    total_duration_ms = (end_time - start_time) / 1_000_000

    print("\n" + "-" * 70)
    print("Sequence temporal properties:")
    print("-" * 70)
    print(f"  Total capture time: {total_duration_ms:.2f} ms")
    print(f"  First frame timestamp: {get_timestamp_ms(sequence[0]):,.2f} ms")
    print(f"  Last frame timestamp: {get_timestamp_ms(sequence[-1]):,.2f} ms")

    sequence_span = get_time_delta_ms(sequence[0], sequence[-1])
    print(f"  Sequence time span: {sequence_span:.2f} ms")
    print(f"  Average frame interval: {sequence_span / (len(sequence) - 1):.2f} ms")

    # Check timing consistency
    print("\n  Frame-to-frame intervals:")
    deltas = []
    for i in range(len(sequence) - 1):
        delta = get_time_delta_ms(sequence[i], sequence[i + 1])
        deltas.append(delta)
        print(f"    {i:2d} -> {i+1:2d}: {delta:6.2f} ms")

    # Statistics
    avg_delta = sum(deltas) / len(deltas)
    max_delta = max(deltas)
    min_delta = min(deltas)
    variance = sum((d - avg_delta) ** 2 for d in deltas) / len(deltas)
    std_dev = variance**0.5

    print(f"\n  Statistics:")
    print(f"    Average: {avg_delta:6.2f} ms")
    print(f"    Min:     {min_delta:6.2f} ms")
    print(f"    Max:     {max_delta:6.2f} ms")
    print(f"    Std Dev: {std_dev:6.2f} ms")

    if std_dev < 5.0:
        print(f"  ✓ Timing is consistent (σ < 5ms)")
    else:
        print(f"  ⚠ Timing variance is high (σ = {std_dev:.2f}ms)")

    manager.stop_continuous_capture()
    manager.stop_capture()


def demonstrate_latency_measurement():
    """Show how to measure end-to-end system latency."""
    print("\n" + "=" * 70)
    print("SYSTEM LATENCY MEASUREMENT")
    print("=" * 70)

    print("\nSimulating recognition pipeline latency...\n")

    # Simulate processing steps
    steps = [
        ("Frame capture", 2),
        ("MediaPipe hand detection", 8),
        ("ViT feature extraction", 15),
        ("LSTM temporal processing", 12),
        ("Semantic matching", 3),
        ("Action mapping", 1),
        ("OS command execution", 5),
    ]

    total_latency = 0
    start_time = time.time_ns()

    for step_name, duration_ms in steps:
        step_start = time.time_ns()
        time.sleep(duration_ms / 1000)  # Simulate processing
        step_end = time.time_ns()

        actual_duration = (step_end - step_start) / 1_000_000
        total_latency += actual_duration

        print(f"  {step_name:.<35} {actual_duration:6.2f} ms")

    end_time = time.time_ns()
    measured_total = (end_time - start_time) / 1_000_000

    print(f"\n  {'Total measured latency':.<35} {measured_total:6.2f} ms")
    print(f"  {'Target latency':.<35} {100.00:6.2f} ms")

    if measured_total < 100:
        print(f"\n  ✓ Performance target met! ({measured_total:.2f}ms < 100ms)")
    else:
        print(f"\n  ✗ Performance target missed ({measured_total:.2f}ms > 100ms)")


def main():
    """Run all demonstrations."""
    print("\n")
    print("╔" + "=" * 68 + "╗")
    print("║" + " " * 10 + "TIMESTAMP FORMAT DEMONSTRATION" + " " * 27 + "║")
    print("║" + " " * 15 + "Input Manager Module" + " " * 33 + "║")
    print("╚" + "=" * 68 + "╝")

    try:
        # Basic timestamp precision
        demonstrate_timestamp_precision()

        # Frame timing (requires camera)
        print("\n\n" + "=" * 70)
        response = input("Continue with camera tests? (y/n): ")
        if response.lower() == "y":
            demonstrate_frame_timing()

            print("\n\n" + "=" * 70)
            response = input("Test sequence timing for LSTM? (y/n): ")
            if response.lower() == "y":
                demonstrate_sequence_timing()

        # Latency measurement (no camera needed)
        print("\n\n" + "=" * 70)
        response = input("Demonstrate latency measurement? (y/n): ")
        if response.lower() == "y":
            demonstrate_latency_measurement()

        print("\n" + "=" * 70)
        print("DEMONSTRATION COMPLETE")
        print("=" * 70 + "\n")

    except KeyboardInterrupt:
        print("\n\nInterrupted by user")
    except Exception as e:
        logger.error(f"Error during demonstration: {e}", exc_info=True)


if __name__ == "__main__":
    main()
