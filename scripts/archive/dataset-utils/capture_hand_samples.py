#!/usr/bin/env python3
"""
Capture hand samples for testing the detection and preprocessing pipeline.

This script captures frames from the camera, detects hands, crops them, and saves
the preprocessed images to a temporary directory for manual inspection.
"""

import sys
import time
import cv2
from pathlib import Path
from datetime import datetime

# Add project root to path for imports
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.input_manager import InputManager
from src.recognition_engine.hand_detection import HandDetector, HandCropper
from src.shared.logger import get_logger

logger = get_logger(__name__)


class HandSampleCapture:
    """Capture hand samples for pipeline testing."""

    def __init__(
        self,
        output_dir: Path,
        camera_index: int = 0,
        num_samples: int = 50,
        capture_interval: float = 0.5,
    ):
        """
        Initialize the sample capture.

        Args:
            output_dir: Directory to save captured hand samples
            camera_index: Camera device index
            num_samples: Number of hand samples to capture
            capture_interval: Time between captures in seconds
        """
        self.output_dir = output_dir
        self.camera_index = camera_index
        self.num_samples = num_samples
        self.capture_interval = capture_interval

        # Create output directory
        self.output_dir.mkdir(parents=True, exist_ok=True)

        # Create subdirectories for raw crops and preprocessed images
        self.raw_dir = self.output_dir / "raw_crops"
        self.processed_dir = self.output_dir / "preprocessed"
        self.raw_dir.mkdir(exist_ok=True)
        self.processed_dir.mkdir(exist_ok=True)

        # Initialize components
        self.input_mgr = None
        self.detector = None
        self.cropper_viz = None  # Without normalization (for visualization)
        self.cropper_model = None  # With normalization (for model input)

        # Statistics
        self.samples_captured = 0
        self.frames_processed = 0

    def initialize(self):
        """Initialize camera and detection components."""
        print("Initializing hand sample capture...")

        # Initialize input manager
        self.input_mgr = InputManager(camera_index=self.camera_index)

        # Initialize detector
        self.detector = HandDetector(
            max_num_hands=2,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5,
            model_complexity=1,
        )

        # Initialize cropper for visualization (no normalization, BGR)
        self.cropper_viz = HandCropper(
            target_size=(224, 224),
            padding=0.2,
            square_crop=True,
            normalize=False,
            bgr_to_rgb=False,
        )

        # Initialize cropper for model input (with normalization, RGB)
        self.cropper_model = HandCropper(
            target_size=(224, 224),
            padding=0.2,
            square_crop=True,
            normalize=True,
            normalization_mean=(0.485, 0.456, 0.406),
            normalization_std=(0.229, 0.224, 0.225),
            bgr_to_rgb=True,
        )

        print("✓ Initialization complete")
        print(f"Output directory: {self.output_dir.absolute()}")
        print(f"  Raw crops: {self.raw_dir.name}/")
        print(f"  Preprocessed: {self.processed_dir.name}/")

    def denormalize_image(self, normalized_img):
        """
        Denormalize a preprocessed image for saving.

        Args:
            normalized_img: Normalized image (float32, RGB)

        Returns:
            Denormalized image (uint8, RGB)
        """
        import numpy as np

        mean = self.cropper_model._normalization_mean
        std = self.cropper_model._normalization_std

        # Reverse normalization
        img = normalized_img * std + mean

        # Clip to [0, 1] and convert to uint8
        img = (img * 255).clip(0, 255).astype("uint8")

        return img

    def save_hand_sample(self, hand_roi_viz, hand_roi_model, sample_id: int):
        """
        Save a hand sample (both raw crop and preprocessed).

        Args:
            hand_roi_viz: HandROI with visualization-friendly image
            hand_roi_model: HandROI with model-ready preprocessed image
            sample_id: Unique sample identifier
        """
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        handedness = hand_roi_viz.detection.handedness.lower()
        confidence = hand_roi_viz.detection.confidence

        # Save raw crop (BGR format, can be viewed directly)
        raw_filename = (
            f"{timestamp}_sample{sample_id:03d}_{handedness}_conf{confidence:.2f}.jpg"
        )
        raw_path = self.raw_dir / raw_filename
        cv2.imwrite(str(raw_path), hand_roi_viz.image)

        # Save preprocessed in two formats:
        # 1. As numpy array (.npy) - actual model input format
        import numpy as np

        npy_filename = f"{timestamp}_sample{sample_id:03d}_{handedness}_normalized.npy"
        npy_path = self.processed_dir / npy_filename
        np.save(str(npy_path), hand_roi_model.image)

        # 2. As denormalized image (.jpg) - for visual comparison
        preprocessed_rgb = self.denormalize_image(hand_roi_model.image)
        preprocessed_bgr = cv2.cvtColor(preprocessed_rgb, cv2.COLOR_RGB2BGR)
        processed_filename = (
            f"{timestamp}_sample{sample_id:03d}_{handedness}_denormalized.jpg"
        )
        processed_path = self.processed_dir / processed_filename
        cv2.imwrite(str(processed_path), preprocessed_bgr)

        print(
            f"  ✓ Saved sample {sample_id}: {handedness} hand (confidence: {confidence:.2f})"
        )

        self.samples_captured += 1

    def run(self):
        """Run the sample capture."""
        self.initialize()

        # Start camera capture
        self.input_mgr.start_continuous_capture()

        print("\n" + "=" * 60)
        print(f"Capturing {self.num_samples} hand samples")
        print("=" * 60)
        print(f"\nCapture interval: {self.capture_interval}s")
        print("Instructions:")
        print("  - Position your hand(s) in front of the camera")
        print("  - Move your hand to different positions and orientations")
        print("  - Try different gestures")
        print("  - Press Q to quit early")
        print()

        last_capture_time = 0

        try:
            while self.samples_captured < self.num_samples:
                # Get latest frame
                frame_data = self.input_mgr.get_latest_frame()
                if frame_data is None:
                    time.sleep(0.001)
                    continue

                self.frames_processed += 1
                current_time = time.time()

                # Display preview window
                preview = frame_data.frame.copy()

                # Detect hands
                detections = self.detector.detect(
                    frame_data.frame, frame_data.timestamp
                )

                # Draw detections on preview
                for detection in detections:
                    x, y, w, h = detection.bbox
                    color = (
                        (0, 255, 255)
                        if detection.handedness == "Left"
                        else (255, 0, 255)
                    )
                    cv2.rectangle(preview, (x, y), (x + w, y + h), color, 2)
                    label = f"{detection.handedness} ({detection.confidence:.2f})"
                    cv2.putText(
                        preview,
                        label,
                        (x, y - 10),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.6,
                        color,
                        2,
                    )

                # Show progress
                cv2.putText(
                    preview,
                    f"Samples: {self.samples_captured}/{self.num_samples}",
                    (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.8,
                    (255, 255, 255),
                    2,
                )

                # Show countdown to next capture
                time_since_capture = current_time - last_capture_time
                if time_since_capture < self.capture_interval:
                    countdown = self.capture_interval - time_since_capture
                    cv2.putText(
                        preview,
                        f"Next capture in: {countdown:.1f}s",
                        (10, 60),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.6,
                        (0, 255, 0),
                        2,
                    )

                cv2.imshow("Hand Sample Capture", preview)

                # Capture samples at intervals
                if (
                    detections
                    and (current_time - last_capture_time) >= self.capture_interval
                ):
                    # Crop hands with both croppers
                    hands_viz = self.cropper_viz.crop_multiple(
                        frame_data.frame, detections
                    )
                    hands_model = self.cropper_model.crop_multiple(
                        frame_data.frame, detections
                    )

                    # Save each detected hand
                    for hand_viz, hand_model in zip(hands_viz, hands_model):
                        if self.samples_captured >= self.num_samples:
                            break
                        self.save_hand_sample(
                            hand_viz, hand_model, self.samples_captured + 1
                        )

                    last_capture_time = current_time

                # Handle keyboard input
                key = cv2.waitKey(1) & 0xFF
                if key == ord("q") or key == 27:  # Q or ESC
                    print("\nCapture interrupted by user")
                    break

        finally:
            self.cleanup()

    def cleanup(self):
        """Clean up resources and print summary."""
        print("\nCleaning up...")

        # Stop camera
        if self.input_mgr:
            self.input_mgr.stop_continuous_capture()

        # Close detector
        if self.detector:
            self.detector.close()

        # Close windows
        cv2.destroyAllWindows()

        # Print summary
        print("\n" + "=" * 60)
        print("Capture Summary")
        print("=" * 60)
        print(f"Samples captured: {self.samples_captured}/{self.num_samples}")
        print(f"Frames processed: {self.frames_processed}")
        print(f"\nOutput directory: {self.output_dir.absolute()}")
        print(f"  Raw crops: {len(list(self.raw_dir.glob('*.jpg')))} images")
        print(f"  Preprocessed: {len(list(self.processed_dir.glob('*.jpg')))} images")

        if self.detector:
            detector_stats = self.detector.get_stats()
            print(f"\nDetector stats:")
            print(f"  Total detections: {detector_stats['total_detections']}")
            print(
                f"  Avg detections/frame: {detector_stats['avg_detections_per_frame']:.2f}"
            )

        if self.cropper_viz:
            cropper_stats = self.cropper_viz.get_stats()
            print(f"\nCropper stats:")
            print(f"  Crops performed: {cropper_stats['crops_performed']}")
            print(f"  Boundary clips: {cropper_stats['boundary_clips']}")

        print("\n✓ Cleanup complete")
        print(f"\nYou can now manually inspect the images in:")
        print(f"  {self.raw_dir.absolute()}")
        print(f"  {self.processed_dir.absolute()}")


def main():
    """Main entry point."""
    import argparse
    import tempfile

    parser = argparse.ArgumentParser(
        description="Capture hand samples for testing the detection pipeline"
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=None,
        help="Output directory for samples (default: temp directory)",
    )
    parser.add_argument(
        "--camera",
        type=int,
        default=0,
        help="Camera device index (default: 0)",
    )
    parser.add_argument(
        "--num-samples",
        type=int,
        default=50,
        help="Number of samples to capture (default: 50)",
    )
    parser.add_argument(
        "--interval",
        type=float,
        default=0.5,
        help="Capture interval in seconds (default: 0.5)",
    )

    args = parser.parse_args()

    # Determine output directory
    if args.output_dir:
        output_dir = Path(args.output_dir)
    else:
        # Use temp directory with timestamp
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        temp_base = Path(tempfile.gettempdir())
        output_dir = temp_base / f"hand_samples_{timestamp}"

    # Create and run capture
    capture = HandSampleCapture(
        output_dir=output_dir,
        camera_index=args.camera,
        num_samples=args.num_samples,
        capture_interval=args.interval,
    )

    try:
        capture.run()
    except KeyboardInterrupt:
        print("\nInterrupted by user")
    except Exception as e:
        print(f"\nError: {e}")
        logger.error("Sample capture failed", exc_info=True)
        sys.exit(1)


if __name__ == "__main__":
    main()
