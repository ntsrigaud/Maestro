"""
Example usage of InputManager module.

This script demonstrates how to use the InputManager for:
1. Single frame capture
2. Continuous capture with frame buffering
3. Frame preprocessing configuration
"""

import time
from input_manager import InputManager


def example_single_frame():
    """Capture a single frame from camera."""
    print("\n=== Example 1: Single Frame Capture ===")

    manager = InputManager(camera_index=0)

    if not manager.start_capture():
        print("Failed to start camera")
        return

    # Get one frame
    frame = manager.get_frame()
    if frame is not None:
        print(f"Captured frame: shape={frame.shape}, dtype={frame.dtype}")

    manager.stop_capture()


def example_continuous_capture():
    """Capture frames continuously in background."""
    print("\n=== Example 2: Continuous Capture ===")

    manager = InputManager(camera_index=0, buffer_size=30, target_fps=30)

    # Configure preprocessing
    manager.set_mirror_enabled(True)
    manager.set_rgb_conversion(True)

    if not manager.start_continuous_capture():
        print("Failed to start continuous capture")
        return

    # Capture for 3 seconds
    print("Capturing frames for 3 seconds...")
    start_time = time.time()
    frame_count = 0

    while time.time() - start_time < 3.0:
        frame_data = manager.get_latest_frame()
        if frame_data is not None:
            frame_count += 1
            if frame_count % 10 == 0:
                print(
                    f"Frame {frame_count}: timestamp={frame_data.timestamp}, "
                    f"shape={frame_data.frame.shape}"
                )
        time.sleep(0.033)  # ~30 FPS polling

    manager.stop_continuous_capture()
    print(f"Captured {frame_count} frames")


def example_with_preprocessing():
    """Demonstrate frame preprocessing options."""
    print("\n=== Example 3: Frame Preprocessing ===")

    manager = InputManager(camera_index=0)

    # Configure preprocessing
    manager.set_mirror_enabled(True)
    manager.set_rgb_conversion(True)
    manager.set_target_size(640, 480)
    manager.set_normalization_params(mean=127.5, std=127.5)

    if not manager.start_capture():
        print("Failed to start camera")
        return

    frame = manager.get_frame()
    if frame is not None:
        print(f"Preprocessed frame:")
        print(f"   Shape: {frame.shape}")
        print(f"   Data type: {frame.dtype}")
        print(f"   Value range: [{frame.min():.2f}, {frame.max():.2f}]")

    manager.stop_capture()


def example_context_manager():
    """Use InputManager as context manager."""
    print("\n=== Example 4: Context Manager Usage ===")

    with InputManager(camera_index=0) as manager:
        manager.start_continuous_capture()

        # Do some work
        time.sleep(1)

        frame_data = manager.get_latest_frame()
        if frame_data:
            print(f"Got frame: {frame_data.frame.shape}")

        # Camera automatically released when exiting context


def example_camera_info():
    """Display camera information."""
    print("\n=== Example 5: Camera Information ===")

    manager = InputManager(camera_index=0)
    manager.start_capture()

    info = manager.get_camera_info()
    print("Camera Info:")
    for key, value in info.items():
        print(f"  {key}: {value}")

    manager.stop_capture()


def example_frame_buffer():
    """Work with frame buffer."""
    print("\n=== Example 6: Frame Buffer ===")

    manager = InputManager(camera_index=0, buffer_size=10)  # Keep last 10 frames

    manager.start_continuous_capture()
    time.sleep(2)  # Let it capture some frames

    # Get all buffered frames
    frames = manager.get_all_frames()
    print(f"Buffer contains {len(frames)} frames")

    if frames:
        print(f"   First frame timestamp: {frames[0].timestamp}")
        print(f"   Last frame timestamp: {frames[-1].timestamp}")
        print(f"   Time span: {frames[-1].timestamp - frames[0].timestamp}ms")

    manager.stop_continuous_capture()


if __name__ == "__main__":
    print("=" * 60)
    print("InputManager Examples")
    print("=" * 60)

    try:
        example_single_frame()
        example_continuous_capture()
        example_with_preprocessing()
        example_context_manager()
        example_camera_info()
        example_frame_buffer()

        print("\n" + "=" * 60)
        print("All examples completed successfully!")
        print("=" * 60)

    except KeyboardInterrupt:
        print("\nExamples interrupted by user")
    except Exception as e:
        print(f"\nError: {e}")
        import traceback

        traceback.print_exc()
