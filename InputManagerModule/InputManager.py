import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(__file__))

try:
    import input_manager_native as im

    print("✅ Module imported successfully")
except ImportError as e:
    print(f"❌ Failed to import module: {e}")
    print("Build the project first: mkdir build && cd build && cmake .. && make")
    sys.exit(1)


def test_camera(camera_index):
    try:
        cam = im.InputManager(camera_index)
        started = cam.start_capture()
        print(f"Camera {camera_index}: started={started}")

        frame = cam.get_frame()
        print(f"Frame shape: {frame.shape}, dtype: {frame.dtype}")

        cam.stop_capture()
        return started and frame.size > 0
    except Exception as e:
        print(f"Error: {e}")
        return False


def main():
    print("Testing cameras...")

    if test_camera(0):
        print("✅ Camera 0 test passed")
    else:
        print("❌ Camera 0 test failed")

    if not test_camera(99):
        print("✅ Invalid camera test passed")
    else:
        print("❌ Invalid camera test failed")


if __name__ == "__main__":
    main()
