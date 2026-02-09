#!/usr/bin/env python3
"""
Demo script for hand detection with real-time visualization.

This script demonstrates the hand detection capabilities by showing:
- Hand bounding boxes
- Hand landmarks (21 points per hand)
- Handedness labels (Left/Right)
- Confidence scores
- FPS counter
- Cropped hand regions
"""

import sys
import time
import numpy as np
import cv2
from pathlib import Path

# Add project root to path for imports
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.input_manager import InputManager, FrameData
from src.recognition_engine.hand_detection import HandDetector, HandCropper
from src.shared.logger import get_logger

logger = get_logger(__name__)

# Colors for visualization (BGR format)
COLOR_BBOX = (0, 255, 0)  # Green
COLOR_LANDMARKS = (255, 0, 0)  # Blue
COLOR_CONNECTIONS = (255, 255, 0)  # Cyan
COLOR_TEXT = (255, 255, 255)  # White
COLOR_LEFT_HAND = (0, 255, 255)  # Yellow
COLOR_RIGHT_HAND = (255, 0, 255)  # Magenta

# MediaPipe hand landmark connections
HAND_CONNECTIONS = [
    (0, 1),
    (1, 2),
    (2, 3),
    (3, 4),  # Thumb
    (0, 5),
    (5, 6),
    (6, 7),
    (7, 8),  # Index
    (0, 9),
    (9, 10),
    (10, 11),
    (11, 12),  # Middle
    (0, 13),
    (13, 14),
    (14, 15),
    (15, 16),  # Ring
    (0, 17),
    (17, 18),
    (18, 19),
    (19, 20),  # Pinky
    (5, 9),
    (9, 13),
    (13, 17),  # Palm connections
]


class HandDetectionDemo:
    """Demo application for hand detection visualization."""

    def __init__(
        self,
        camera_index: int = 0,
        window_name: str = "Hand Detection Demo",
        show_landmarks: bool = True,
        show_cropped: bool = True,
        max_num_hands: int = 2,
    ):
        """
        Initialize the demo.

        Args:
            camera_index: Camera device index
            window_name: Window title
            show_landmarks: Whether to draw hand landmarks
            show_cropped: Whether to show cropped hand regions
            max_num_hands: Maximum number of hands to detect
        """
        self.camera_index = camera_index
        self.window_name = window_name
        self.show_landmarks = show_landmarks
        self.show_cropped = show_cropped
        self.max_num_hands = max_num_hands

        # Initialize components
        self.input_mgr = None
        self.detector = None
        self.cropper = None

        # FPS tracking
        self.fps_history = []
        self.fps_window = 30

        # Statistics
        self.frame_count = 0
        self.detection_count = 0

    def initialize(self):
        """Initialize camera and detection components."""
        print("Initializing hand detection demo...")

        # Initialize input manager
        self.input_mgr = InputManager(camera_index=self.camera_index)

        # Initialize detector
        self.detector = HandDetector(
            max_num_hands=self.max_num_hands,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5,
            model_complexity=1,
        )

        # Initialize cropper
        self.cropper = HandCropper(
            target_size=(224, 224),
            padding=0.2,
            square_crop=True,
            normalize=False,  # Don't normalize for visualization
            bgr_to_rgb=False,  # Keep BGR for OpenCV display
        )

        print("✓ Initialization complete")

    def draw_bbox(self, frame: np.ndarray, detection, color):
        """Draw bounding box on frame."""
        x, y, w, h = detection.bbox
        cv2.rectangle(frame, (x, y), (x + w, y + h), color, 2)

    def draw_landmarks(self, frame: np.ndarray, detection):
        """Draw hand landmarks and connections."""
        frame_h, frame_w = frame.shape[:2]
        landmarks = detection.landmarks

        # Choose color based on handedness
        color = COLOR_LEFT_HAND if detection.handedness == "Left" else COLOR_RIGHT_HAND

        # Draw connections
        for connection in HAND_CONNECTIONS:
            start_idx, end_idx = connection
            start_pt = (
                int(landmarks[start_idx, 0] * frame_w),
                int(landmarks[start_idx, 1] * frame_h),
            )
            end_pt = (
                int(landmarks[end_idx, 0] * frame_w),
                int(landmarks[end_idx, 1] * frame_h),
            )
            cv2.line(frame, start_pt, end_pt, COLOR_CONNECTIONS, 2)

        # Draw landmark points
        for lm in landmarks:
            pt = (int(lm[0] * frame_w), int(lm[1] * frame_h))
            cv2.circle(frame, pt, 3, color, -1)

    def draw_label(self, frame: np.ndarray, detection, y_offset: int = 0):
        """Draw handedness and confidence label."""
        x, y, w, h = detection.bbox
        label = f"{detection.handedness} ({detection.confidence:.2f})"

        # Choose color based on handedness
        color = COLOR_LEFT_HAND if detection.handedness == "Left" else COLOR_RIGHT_HAND

        # Draw background rectangle
        (label_w, label_h), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
        label_y = max(y - 10 + y_offset, label_h + 5)
        cv2.rectangle(
            frame,
            (x, label_y - label_h - 5),
            (x + label_w + 10, label_y + 5),
            color,
            -1,
        )

        # Draw text
        cv2.putText(
            frame,
            label,
            (x + 5, label_y),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (0, 0, 0),
            2,
        )

    def draw_stats(self, frame: np.ndarray, fps: float, detections: list):
        """Draw statistics overlay."""
        stats = [
            f"FPS: {fps:.1f}",
            f"Hands: {len(detections)}",
            f"Frame: {self.frame_count}",
            f"Total detections: {self.detection_count}",
        ]

        y_offset = 30
        for stat in stats:
            cv2.putText(
                frame,
                stat,
                (10, y_offset),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                COLOR_TEXT,
                2,
            )
            y_offset += 30

    def draw_help(self, frame: np.ndarray):
        """Draw help text."""
        help_text = [
            "Controls:",
            "  L - Toggle landmarks",
            "  C - Toggle cropped view",
            "  R - Reset stats",
            "  Q/ESC - Quit",
        ]

        y_offset = frame.shape[0] - 20 * len(help_text)
        for text in help_text:
            cv2.putText(
                frame,
                text,
                (10, y_offset),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                COLOR_TEXT,
                1,
            )
            y_offset += 20

    def show_cropped_hands(self, hand_rois: list):
        """Display cropped hand regions in separate windows."""
        for i, roi in enumerate(hand_rois):
            window_name = f"Hand {i+1} - {roi.detection.handedness}"

            # Denormalize for display if needed
            if roi.image.dtype == np.float32:
                # Assuming [0, 1] range
                display_img = (roi.image * 255).astype(np.uint8)
            else:
                display_img = roi.image

            # Resize for better visibility
            display_img = cv2.resize(display_img, (224, 224))

            cv2.imshow(window_name, display_img)

    def calculate_fps(
        self, frame_data_prev: FrameData, frame_data_curr: FrameData
    ) -> float:
        """Calculate FPS from frame timestamps."""
        if frame_data_prev is None or frame_data_curr is None:
            return 0.0

        fps = FrameData.calculate_fps(frame_data_prev, frame_data_curr)
        self.fps_history.append(fps)

        # Keep only recent history
        if len(self.fps_history) > self.fps_window:
            self.fps_history.pop(0)

        # Return average FPS
        return sum(self.fps_history) / len(self.fps_history)

    def run(self):
        """Run the demo application."""
        self.initialize()

        # Start camera capture
        self.input_mgr.start_continuous_capture()
        print("\n" + "=" * 60)
        print("Hand Detection Demo Running")
        print("=" * 60)
        print("\nControls:")
        print("  L - Toggle landmarks")
        print("  C - Toggle cropped view")
        print("  R - Reset statistics")
        print("  Q/ESC - Quit")
        print()

        frame_data_prev = None

        try:
            while True:
                # Get latest frame
                frame_data = self.input_mgr.get_latest_frame()
                if frame_data is None:
                    time.sleep(0.001)
                    continue

                self.frame_count += 1

                # Calculate FPS
                fps = self.calculate_fps(frame_data_prev, frame_data)
                frame_data_prev = frame_data

                # Detect hands
                detections = self.detector.detect(
                    frame_data.frame, frame_data.timestamp
                )
                self.detection_count += len(detections)

                # Create visualization frame (copy to avoid modifying original)
                vis_frame = frame_data.frame.copy()

                # Draw detections
                for detection in detections:
                    # Draw bounding box
                    color = (
                        COLOR_LEFT_HAND
                        if detection.handedness == "Left"
                        else COLOR_RIGHT_HAND
                    )
                    self.draw_bbox(vis_frame, detection, color)

                    # Draw landmarks if enabled
                    if self.show_landmarks:
                        self.draw_landmarks(vis_frame, detection)

                    # Draw label
                    self.draw_label(vis_frame, detection)

                # Draw statistics
                self.draw_stats(vis_frame, fps, detections)

                # Draw help
                self.draw_help(vis_frame)

                # Show main window
                cv2.imshow(self.window_name, vis_frame)

                # Show cropped hands if enabled
                if self.show_cropped and detections:
                    hand_rois = self.cropper.crop_multiple(frame_data.frame, detections)
                    self.show_cropped_hands(hand_rois)

                # Handle keyboard input
                key = cv2.waitKey(1) & 0xFF

                if key == ord("q") or key == 27:  # Q or ESC
                    print("\nQuitting...")
                    break
                elif key == ord("l"):  # Toggle landmarks
                    self.show_landmarks = not self.show_landmarks
                    print(f"Landmarks: {'ON' if self.show_landmarks else 'OFF'}")
                elif key == ord("c"):  # Toggle cropped view
                    self.show_cropped = not self.show_cropped
                    print(f"Cropped view: {'ON' if self.show_cropped else 'OFF'}")
                    if not self.show_cropped:
                        cv2.destroyWindow("Hand 1 - Left")
                        cv2.destroyWindow("Hand 1 - Right")
                        cv2.destroyWindow("Hand 2 - Left")
                        cv2.destroyWindow("Hand 2 - Right")
                elif key == ord("r"):  # Reset stats
                    self.frame_count = 0
                    self.detection_count = 0
                    self.fps_history = []
                    self.detector.reset_stats()
                    self.cropper.reset_stats()
                    print("Statistics reset")

        finally:
            self.cleanup()

    def cleanup(self):
        """Clean up resources."""
        print("\nCleaning up...")

        # Stop camera
        if self.input_mgr:
            self.input_mgr.stop_continuous_capture()

        # Close detector
        if self.detector:
            self.detector.close()

        # Close windows
        cv2.destroyAllWindows()

        # Print final statistics
        print("\n" + "=" * 60)
        print("Session Statistics")
        print("=" * 60)

        if self.detector:
            detector_stats = self.detector.get_stats()
            print("\nDetector stats:")
            for key, value in detector_stats.items():
                print(f"  {key}: {value}")

        if self.cropper:
            cropper_stats = self.cropper.get_stats()
            print("\nCropper stats:")
            for key, value in cropper_stats.items():
                print(f"  {key}: {value}")

        print("\n✓ Cleanup complete")


def main():
    """Main entry point."""
    import argparse

    parser = argparse.ArgumentParser(
        description="Hand detection demo with visualization"
    )
    parser.add_argument(
        "--camera",
        type=int,
        default=0,
        help="Camera device index (default: 0)",
    )
    parser.add_argument(
        "--no-landmarks",
        action="store_true",
        help="Disable landmark visualization",
    )
    parser.add_argument(
        "--no-cropped",
        action="store_true",
        help="Disable cropped hand windows",
    )
    parser.add_argument(
        "--max-hands",
        type=int,
        default=2,
        help="Maximum number of hands to detect (default: 2)",
    )

    args = parser.parse_args()

    # Create and run demo
    demo = HandDetectionDemo(
        camera_index=args.camera,
        show_landmarks=not args.no_landmarks,
        show_cropped=not args.no_cropped,
        max_num_hands=args.max_hands,
    )

    try:
        demo.run()
    except KeyboardInterrupt:
        print("\nInterrupted by user")
    except Exception as e:
        print(f"\nError: {e}")
        logger.error("Demo failed", exc_info=True)
        sys.exit(1)


if __name__ == "__main__":
    main()
