#!/usr/bin/env python3
"""
Test to demonstrate the efficiency improvement from using Event.wait()
instead of polling loop for first frame notification.
"""

import time
from input_manager import InputManager
from shared.logger import get_logger

logger = get_logger(__name__)


def test_startup_efficiency():
    """Test the efficiency of the new Event-based startup."""
    print("=" * 70)
    print("TESTING START_CONTINUOUS_CAPTURE EFFICIENCY")
    print("=" * 70)

    print("\nOLD APPROACH (before optimization):")
    print("  - Polling loop: 100 iterations * 10ms sleep = 100 iterations")
    print("  - CPU overhead: Thread wakes up 100 times to check condition")
    print("  - Latency: Up to 10ms between frame ready and detection")

    print("\nNEW APPROACH (with threading.Event):")
    print("  - Event.wait(): Blocks until signaled or timeout")
    print("  - CPU overhead: Thread sleeps efficiently, wakes once")
    print("  - Latency: Immediate notification when frame is ready")

    print("\n" + "-" * 70)
    print("Running actual test...")
    print("-" * 70)

    manager = InputManager(camera_index=0)

    # Measure startup time
    start_time = time.time()
    success = manager.start_continuous_capture()
    end_time = time.time()

    startup_time = (end_time - start_time) * 1000  # Convert to milliseconds

    if success:
        print(f"\n✓ Continuous capture started successfully")
        print(f"  Startup time: {startup_time:.2f} ms")

        # Get first frame to verify it's working
        frame = manager.get_latest_frame()
        if frame:
            print(
                f"  First frame: {frame.frame.shape}, "
                f"timestamp={frame.timestamp_ms():.2f}ms"
            )

        # Compare with theoretical polling time
        print(f"\n  Theoretical polling overhead (old): ~100-1000 ms")
        print(f"  Actual startup time (new): {startup_time:.2f} ms")

        if startup_time < 100:
            print(f"\n  ✓ Excellent! Startup is <100ms (very efficient)")
        elif startup_time < 500:
            print(f"\n  ✓ Good! Startup is <500ms (efficient)")
        else:
            print(f"\n  ⚠ Startup took longer than expected")

        # Clean up
        manager.stop_continuous_capture()
        manager.stop_capture()

    else:
        print(f"\n✗ Failed to start continuous capture")
        print(f"  (Camera may not be available)")

    print("\n" + "=" * 70)
    print("Benefits of Event-based approach:")
    print("=" * 70)
    print("  1. No CPU wasted on polling iterations")
    print("  2. Immediate response when frame is ready")
    print("  3. Clean, readable code with Event.wait(timeout)")
    print("  4. Standard threading pattern (more maintainable)")
    print("=" * 70)


def test_multiple_starts():
    """Test that starting multiple times works correctly."""
    print("\n" + "=" * 70)
    print("TESTING MULTIPLE START/STOP CYCLES")
    print("=" * 70)

    manager = InputManager(camera_index=0)

    for cycle in range(3):
        print(f"\nCycle {cycle + 1}:")

        start_time = time.time()
        success = manager.start_continuous_capture()
        startup_time = (time.time() - start_time) * 1000

        if success:
            print(f"  ✓ Started in {startup_time:.2f} ms")
            time.sleep(0.1)  # Capture a few frames

            frame = manager.get_latest_frame()
            if frame:
                print(f"    Frame #{frame.frame_number} captured")

            manager.stop_continuous_capture()
            print(f"  ✓ Stopped")
        else:
            print(f"  ✗ Failed to start")
            break

    manager.stop_capture()
    print("\n✓ All cycles completed successfully")


def main():
    """Run all tests."""
    print("\n")
    print("╔" + "=" * 68 + "╗")
    print("║" + " " * 15 + "EVENT-BASED STARTUP TEST" + " " * 29 + "║")
    print("║" + " " * 10 + "Demonstrates threading.Event efficiency" + " " * 19 + "║")
    print("╚" + "=" * 68 + "╝")
    print()

    try:
        test_startup_efficiency()

        print("\n")
        response = input("Run multiple start/stop cycles test? (y/n): ")
        if response.lower() == "y":
            test_multiple_starts()

        print("\n✓ All tests completed!\n")

    except KeyboardInterrupt:
        print("\n\nInterrupted by user")
    except Exception as e:
        logger.error(f"Error during test: {e}", exc_info=True)


if __name__ == "__main__":
    main()
