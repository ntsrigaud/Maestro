import os
import sys
import time

import numpy as np

try:
    import cv2
except ImportError:
    print("❌ OpenCV for Python is not installed.")
    sys.exit(1)

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "build/lib"))

try:
    import input_manager_native as im

except ImportError as e:
    print(f"❌ Failed to import native module: {e}")
    sys.exit(1)


class InputManagerDemo:
    def __init__(self, camera_index=0):
        self._cam = im.InputManager(camera_index)

    def capture_frame(self, out_file="output.npy"):
        """Capture one frame and save it as .npy"""
        if not self._cam.start_capture():
            print("❌ Could not start camera")
            return False

        frame = self._cam.get_frame()
        self._cam.stop_capture()

        if frame.size == 0:
            print("❌ Failed to capture frame")
            return False

        np.save(out_file, frame)
        print(f"✅ Saved single frame to {out_file}")
        return True

    def record_video(self, out_file="output.mp4", duration=5, fps=30):
        """Record a short video and save it"""
        if not self._cam.start_capture():
            print("❌ Could not start camera")
            return False

        frame = self._cam.get_frame()
        if frame.size == 0:
            print("❌ Failed to capture initial frame")
            self._cam.stop_capture()
            return False

        # Convert normalized float32 → uint8 for saving
        frame_uint8 = ((frame * 0.5) + 0.5) * 255
        frame_uint8 = frame_uint8.astype(np.uint8)
        h, w, _ = frame_uint8.shape

        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        out = cv2.VideoWriter(out_file, fourcc, fps, (w, h))
        if not out.isOpened():
            print(f"❌ Could not open video writer {out_file}")
            self._cam.stop_capture()
            return False

        print(f"🎥 Recording video to {out_file} for {duration}s...")
        start = time.time()
        frames_written = 0

        while (time.time() - start) < duration:
            frame = self._cam.get_frame()
            if frame.size == 0:
                continue

            frame_uint8 = ((frame * 0.5) + 0.5) * 255
            frame_uint8 = frame_uint8.astype(np.uint8)
            frame_bgr = cv2.cvtColor(frame_uint8, cv2.COLOR_RGB2BGR)
            out.write(frame_bgr)
            frames_written += 1

        self._cam.stop_capture()
        out.release()
        print(f"✅ Saved {frames_written} frames to {out_file}")
        return True


def main():
    demo = InputManagerDemo()

    if demo.capture_frame("output.npy"):
        demo.record_video("output.mp4", duration=5)


if __name__ == "__main__":
    main()

