import atexit
import os
import signal
import sys
import time
from datetime import datetime

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "build/lib"))

try:
    import input_manager_native as im
except ImportError as e:
    print(f"❌ Failed to import native module: {e}")
    sys.exit(1)


class VideoCapture:
    def __init__(self, camera_index=0):
        self._cam = im.InputManager(camera_index)
        self._is_running = False
        self.captured_frames = []  # Will store (frame_array, timestamp) tuples

        atexit.register(self.cleanup)
        signal.signal(signal.SIGINT, self.signal_handler)

    def signal_handler(self, signum, frame):
        print(f"\nStopping capture...")
        self.stop_capture()

    def cleanup(self):
        if self._is_running:
            self.stop_capture()

    def _get_frame_data(self, frame_data):
        """Extract frame data and timestamp"""
        try:
            frame = frame_data.get_frame()
            timestamp = getattr(frame_data, "timestamp", 0)
            if frame is not None and frame.size > 0:
                return frame, timestamp
        except:
            pass

        try:
            if hasattr(frame_data, "frame") and frame_data.frame.size > 0:
                frame = frame_data.frame
                timestamp = getattr(frame_data, "timestamp", 0)
                return frame, timestamp
        except:
            pass

        return None, 0

    def capture_video_continuous(self, output_file="captured_video"):
        """Capture video continuously until stopped manually"""
        if not self._cam.start_continuous_capture():
            print("❌ Could not start camera capture")
            return False

        self._is_running = True
        self.captured_frames = []
        start_time = time.time()
        frame_count = 0

        print("🎥 Continuous video capture started!")
        print("Press Ctrl+C to stop capture")
        print("Frames will be saved when you stop the capture")

        try:
            while self._is_running:
                frame_data = self._cam.get_latest_frame()
                frame_array, timestamp = self._get_frame_data(frame_data)

                if frame_array is not None:
                    frame_count += 1
                    self.captured_frames.append((frame_array.copy(), timestamp))

                    # Show progress every 30 frames
                    if frame_count % 30 == 0:
                        current_time = time.time() - start_time
                        fps = frame_count / current_time if current_time > 0 else 0
                        print(
                            f"📊 Frames captured: {frame_count}, Duration: {current_time:.1f}s, FPS: {fps:.1f}"
                        )

                time.sleep(0.033)  # ~30fps

        except KeyboardInterrupt:
            print("\nCapture interrupted by user")

        # Save data when stopped
        if self.captured_frames:
            self._save_detailed_data(output_file)
            return True
        else:
            print("❌ No frames captured")
            return False

    def stop_capture(self):
        if not self._is_running:
            return
        self._is_running = False
        self._cam.stop_continuous_capture()

    def _format_timestamp(self, timestamp_ms):
        """Convert milliseconds timestamp to readable format"""
        if timestamp_ms == 0:
            return "N/A"

        # Convert to seconds
        timestamp_sec = timestamp_ms / 1000.0

        # Format as readable time
        dt = datetime.fromtimestamp(timestamp_sec)
        return dt.strftime("%Y-%m-%d %H:%M:%S.%f")[
            :-3
        ]  # Remove last 3 digits of microseconds

    def _calculate_time_stats(self):
        """Calculate timing statistics from timestamps"""
        if not self.captured_frames:
            return None

        timestamps = [ts for _, ts in self.captured_frames if ts > 0]

        if len(timestamps) < 2:
            return None

        time_deltas = []
        for i in range(1, len(timestamps)):
            delta_ms = timestamps[i] - timestamps[i - 1]
            time_deltas.append(delta_ms)

        if time_deltas:
            return {
                "total_duration_ms": timestamps[-1] - timestamps[0],
                "avg_frame_interval_ms": np.mean(time_deltas),
                "min_frame_interval_ms": np.min(time_deltas),
                "max_frame_interval_ms": np.max(time_deltas),
                "fps": 1000.0 / np.mean(time_deltas) if np.mean(time_deltas) > 0 else 0,
            }
        return None

    def _save_detailed_data(self, filename):
        """Save detailed frame data to text file with timestamps"""
        if not self.captured_frames:
            return

        txt_file = f"{filename}_detailed.txt"

        with open(txt_file, "w") as f:
            f.write("=== CONTINUOUS VIDEO FRAME DATA WITH TIMESTAMPS ===\n\n")
            f.write(f"Total frames captured: {len(self.captured_frames)}\n")
            f.write(f"Frame shape: {self.captured_frames[0][0].shape}\n")
            f.write(f"Data type: {self.captured_frames[0][0].dtype}\n")

            # Calculate and write timing statistics
            time_stats = self._calculate_time_stats()
            if time_stats:
                f.write(f"\n=== TIMING STATISTICS ===\n")
                f.write(
                    f"Total capture duration: {time_stats['total_duration_ms']/1000:.3f} seconds\n"
                )
                f.write(
                    f"Average frame interval: {time_stats['avg_frame_interval_ms']:.2f} ms\n"
                )
                f.write(
                    f"Min frame interval: {time_stats['min_frame_interval_ms']:.2f} ms\n"
                )
                f.write(
                    f"Max frame interval: {time_stats['max_frame_interval_ms']:.2f} ms\n"
                )
                f.write(f"Calculated FPS: {time_stats['fps']:.2f}\n")
            f.write("\n")

            # Only save first 50 frames to avoid huge files
            frames_to_save = min(50, len(self.captured_frames))
            f.write(
                f"Showing first {frames_to_save} frames (out of {len(self.captured_frames)} total)\n\n"
            )

            for frame_idx, (frame, timestamp) in enumerate(
                self.captured_frames[:frames_to_save]
            ):
                f.write(f"\n{'='*60}\n")
                f.write(f"FRAME {frame_idx}\n")
                f.write(f"{'='*60}\n")
                f.write(f"Timestamp: {timestamp} ms\n")
                f.write(f"Formatted: {self._format_timestamp(timestamp)}\n")
                f.write(
                    f"Relative: +{(timestamp - self.captured_frames[0][1])/1000:.3f}s from start\n\n"
                )

                height, width, channels = frame.shape

                # Save complete 10x10 top-left corner
                f.write("10x10 Top-Left Corner [R, G, B]:\n")
                for i in range(min(10, height)):
                    for j in range(min(10, width)):
                        f.write(
                            f"[{frame[i,j,0]:7.4f}, {frame[i,j,1]:7.4f}, {frame[i,j,2]:7.4f}] "
                        )
                    f.write("\n")

                # Save channel statistics
                f.write(f"\nChannel Statistics:\n")
                for channel, name in enumerate(["Red", "Green", "Blue"]):
                    channel_data = frame[:, :, channel]
                    f.write(
                        f"{name}: min={channel_data.min():7.4f}, max={channel_data.max():7.4f}, "
                    )
                    f.write(
                        f"mean={channel_data.mean():7.4f}, std={channel_data.std():7.4f}\n"
                    )

                # Add separator for readability
                if frame_idx < frames_to_save - 1:
                    f.write("\n" + "-" * 40 + "\n")

        print(f"💾 Saved detailed data with timestamps to '{txt_file}'")
        print(f"📊 Total frames captured: {len(self.captured_frames)}")

        if time_stats:
            print(f"⏱️  Capture duration: {time_stats['total_duration_ms']/1000:.2f}s")
            print(f"📈 Calculated FPS: {time_stats['fps']:.2f}")


def main():
    capture = VideoCapture()

    print("🎥 Continuous Video Capture")
    print("=" * 40)
    print("This will capture video until you press Ctrl+C")
    print("Frames will be saved when you stop the capture")
    print()

    output_name = (
        input("Enter output filename (default: captured_video): ") or "captured_video"
    )

    print("\nStarting continuous capture...")
    print("Press Ctrl+C to stop and save data")
    print("-" * 40)

    capture.capture_video_continuous(output_name)


if __name__ == "__main__":
    main()
