#!/usr/bin/env python3
"""
Real-time gesture recognition using fine-tuned ViT + LSTM.

This script:
1. Loads the fine-tuned ViT model (best_model.pth from checkpoints/)
2. Loads the trained LSTM model (best_lstm_model79.h5)
3. Captures video from camera
4. Extracts ViT features per frame
5. Feeds sequences to LSTM for dynamic gesture recognition
6. Displays results with confidence scores

Usage:
    python scripts/infer_lstm_real_time.py [--confidence-threshold 0.6] [--device cpu]

Controls:
    - 'q': Quit
    - 'r': Reset feature buffer
    - 's': Save current frame
"""

import sys
import time
import argparse
import numpy as np
import cv2
import torch
from pathlib import Path
from collections import deque
from typing import Tuple, Optional, Dict, List

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.shared.logger import get_logger
from src.recognition_engine.temporal.lstm_model import LSTMGestureModel
from src.recognition_engine.temporal.vit_feature_extractor import ViTFeatureExtractor

logger = get_logger(__name__)


# Gesture class mapping (update to match your training data)
GESTURE_CLASSES = {
    0: "unknown",
    1: "thumbs_up",
    2: "thumbs_down",
    3: "open_palm",
    4: "peace_sign",
    5: "ok_sign",
    6: "pointing",
    7: "wave",
    8: "swipe_left",
    9: "swipe_right",
}

# Configuration defaults
DEFAULT_CONFIG = {
    "vit_checkpoint": "checkpoints/best_model.pth",
    "lstm_model_path": "notebooks/best_lstm_model79.h5",
    "sequence_length": 16,
    "confidence_threshold": 0.6,
    "target_fps": 30,
    "display_width": 1280,
    "display_height": 720,
    "camera_index": 0,
}


class ViTLSTMRecognizer:
    """Real-time gesture recognition using ViT feature extraction + LSTM."""

    def __init__(
        self,
        vit_checkpoint: str,
        lstm_model_path: str,
        sequence_length: int = 16,
        device: str = "cuda" if torch.cuda.is_available() else "cpu",
    ):
        """
        Initialize the recognizer.

        Args:
            vit_checkpoint: Path to fine-tuned ViT model
            lstm_model_path: Path to trained LSTM model
            sequence_length: Number of frames for LSTM sequence
            device: torch device (cuda or cpu)
        """
        self.device = device
        self.sequence_length = sequence_length

        logger.info(f"Initializing ViTLSTMRecognizer on {device}")

        # Load ViT feature extractor
        logger.info(f"Loading ViT from {vit_checkpoint}")
        self.vit_extractor = ViTFeatureExtractor(
            checkpoint_path=vit_checkpoint,
            device=device,
        )

        # Load LSTM model
        logger.info(f"Loading LSTM from {lstm_model_path}")
        self.lstm_model = LSTMGestureModel(
            model_path=lstm_model_path,
            sequence_length=sequence_length,
        )

        # Feature sequence buffer
        self.feature_buffer = deque(maxlen=sequence_length)
        self.last_prediction = None
        self.last_confidence = 0.0

        logger.info("✓ ViTLSTMRecognizer initialized successfully")

    def extract_features(self, image: np.ndarray) -> Optional[np.ndarray]:
        """
        Extract ViT features from a single image.

        Args:
            image: RGB image (H, W, 3), values in [0, 1] or [0, 255]

        Returns:
            Feature vector (768,) or None if extraction fails
        """
        try:
            # Normalize if needed
            if image.max() > 1.0:
                image = image.astype(np.float32) / 255.0

            features = self.vit_extractor.extract_features(image)
            return features
        except Exception as e:
            logger.warning(f"Feature extraction failed: {e}")
            return None

    def predict_dynamic_gesture(self) -> Tuple[Optional[str], float]:
        """
        Predict gesture from buffered feature sequence.

        Returns:
            (gesture_name, confidence) or (None, 0.0) if buffer not full
        """
        if len(self.feature_buffer) < self.sequence_length:
            return None, 0.0

        try:
            # Prepare sequence for LSTM
            sequence = np.array(list(self.feature_buffer), dtype=np.float32)
            class_id, confidence = self.lstm_model.predict_single(sequence)

            self.last_prediction = class_id
            self.last_confidence = float(confidence)

            gesture_name = GESTURE_CLASSES.get(class_id, f"unknown_{class_id}")
            return gesture_name, float(confidence)

        except Exception as e:
            logger.warning(f"Prediction failed: {e}")
            return None, 0.0

    def reset_buffer(self):
        """Clear feature buffer."""
        self.feature_buffer.clear()
        self.last_prediction = None
        self.last_confidence = 0.0

    def get_buffer_status(self) -> Dict[str, int]:
        """Get buffer status info."""
        return {
            "current": len(self.feature_buffer),
            "max": self.sequence_length,
            "filled": len(self.feature_buffer) == self.sequence_length,
        }


class LiveGestureDemo:
    """Live demonstration of LSTM gesture recognition."""

    def __init__(
        self,
        vit_checkpoint: str = DEFAULT_CONFIG["vit_checkpoint"],
        lstm_model_path: str = DEFAULT_CONFIG["lstm_model_path"],
        confidence_threshold: float = DEFAULT_CONFIG["confidence_threshold"],
        device: str = "cuda" if torch.cuda.is_available() else "cpu",
        camera_index: int = DEFAULT_CONFIG["camera_index"],
    ):
        """Initialize demo components."""
        self.recognizer = ViTLSTMRecognizer(
            vit_checkpoint=vit_checkpoint,
            lstm_model_path=lstm_model_path,
            device=device,
        )
        self.confidence_threshold = confidence_threshold
        self.camera_index = camera_index

        # Camera setup
        self.cap = cv2.VideoCapture(camera_index)
        if not self.cap.isOpened():
            raise RuntimeError(f"Failed to open camera {camera_index}")

        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, DEFAULT_CONFIG["display_width"])
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, DEFAULT_CONFIG["display_height"])
        self.cap.set(cv2.CAP_PROP_FPS, DEFAULT_CONFIG["target_fps"])

        # Stats
        self.frame_count = 0
        self.fps = 0.0
        self.last_time = time.time()
        self.frames_processed = 0
        self.predictions_made = 0

        logger.info(f"Camera opened: {camera_index}")

    def draw_ui(self, frame: np.ndarray) -> np.ndarray:
        """Draw UI elements on frame."""
        h, w = frame.shape[:2]
        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 0.7
        thickness = 2
        line_height = 30

        # FPS
        cv2.putText(
            frame,
            f"FPS: {self.fps:.1f}",
            (10, line_height),
            font,
            font_scale,
            (0, 255, 0),
            thickness,
        )

        # Buffer status
        buffer_status = self.recognizer.get_buffer_status()
        color = (0, 255, 0) if buffer_status["filled"] else (0, 165, 255)
        cv2.putText(
            frame,
            f"Buffer: {buffer_status['current']}/{buffer_status['max']}",
            (10, line_height * 2),
            font,
            font_scale,
            color,
            thickness,
        )

        # Device info
        device_text = f"Device: {self.recognizer.device}"
        cv2.putText(
            frame,
            device_text,
            (10, line_height * 3),
            font,
            0.6,
            (200, 200, 200),
            1,
        )

        # Prediction (bottom)
        if self.recognizer.last_prediction is not None:
            gesture_name = GESTURE_CLASSES.get(
                self.recognizer.last_prediction, "unknown"
            )
            text = f"{gesture_name} ({self.recognizer.last_confidence:.2f})"
            color = (
                (0, 255, 0) if self.recognizer.last_confidence > 0.7 else (0, 165, 255)
            )

            # Background box
            text_size = cv2.getTextSize(text, font, 1.0, 2)[0]
            box_margin = 10
            cv2.rectangle(
                frame,
                (box_margin, h - text_size[1] - 30),
                (box_margin + text_size[0] + 20, h - 10),
                (50, 50, 50),
                -1,
            )

            cv2.putText(
                frame,
                text,
                (box_margin + 10, h - 20),
                font,
                1.0,
                color,
                2,
            )

        # Controls
        controls_text = "q: quit | r: reset buffer | s: screenshot"
        cv2.putText(
            frame,
            controls_text,
            (10, h - 10),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (150, 150, 150),
            1,
        )

        return frame

    def run(self):
        """Run the demo loop."""
        print("\n" + "=" * 80)
        print("REAL-TIME LSTM + ViT GESTURE RECOGNITION DEMO")
        print("=" * 80)
        print(f"\nConfiguration:")
        print(f"  Sequence length: {self.recognizer.sequence_length} frames")
        print(f"  Gesture classes: {len(GESTURE_CLASSES)}")
        print(f"  Device: {self.recognizer.device}")
        print(f"  Confidence threshold: {self.confidence_threshold:.2f}")
        print(f"\nControls:")
        print(f"  'q' - Quit")
        print(f"  'r' - Reset feature buffer")
        print(f"  's' - Save screenshot")
        print("=" * 80 + "\n")

        try:
            while True:
                ret, frame = self.cap.read()
                if not ret:
                    logger.error("Failed to read frame")
                    break

                self.frame_count += 1
                self.frames_processed += 1

                # Convert BGR to RGB for processing
                rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                rgb_normalized = rgb_frame.astype(np.float32) / 255.0

                # Extract features
                features = self.recognizer.extract_features(rgb_normalized)

                if features is not None:
                    # Add to buffer
                    self.recognizer.feature_buffer.append(features)

                    # Predict if buffer is full
                    if (
                        len(self.recognizer.feature_buffer)
                        == self.recognizer.sequence_length
                    ):
                        gesture, confidence = self.recognizer.predict_dynamic_gesture()
                        if gesture and confidence >= self.confidence_threshold:
                            self.predictions_made += 1
                            logger.info(
                                f"✓ Predicted: {gesture} (confidence: {confidence:.3f})"
                            )

                # Draw UI
                frame = self.draw_ui(frame)

                # Calculate FPS
                current_time = time.time()
                if current_time - self.last_time > 0.5:
                    self.fps = self.frame_count / (current_time - self.last_time)
                    self.frame_count = 0
                    self.last_time = current_time

                # Display
                cv2.imshow("LSTM Gesture Recognition", frame)

                # Handle input
                key = cv2.waitKey(1) & 0xFF
                if key == ord("q"):
                    print("\nQuitting...")
                    break
                elif key == ord("r"):
                    self.recognizer.reset_buffer()
                    logger.info("Buffer reset")
                elif key == ord("s"):
                    filename = f"screenshot_{int(time.time())}.png"
                    cv2.imwrite(filename, frame)
                    logger.info(f"Screenshot saved: {filename}")

        except KeyboardInterrupt:
            print("\nInterrupted by user")
        except Exception as e:
            logger.error(f"Error during demo: {e}", exc_info=True)
        finally:
            self.cleanup()

    def cleanup(self):
        """Clean up resources."""
        self.cap.release()
        cv2.destroyAllWindows()

        # Print stats
        print("\n" + "=" * 80)
        print("DEMO STATISTICS")
        print("=" * 80)
        print(f"  Total frames processed: {self.frames_processed}")
        print(f"  Predictions made: {self.predictions_made}")
        print(f"  Average FPS: {self.fps:.1f}")
        print("=" * 80)
        logger.info("✓ Demo closed cleanly")


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="Real-time LSTM gesture recognition demo",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Run with default settings
  python scripts/infer_lstm_real_time.py

  # Run with custom confidence threshold
  python scripts/infer_lstm_real_time.py --confidence-threshold 0.7

  # Force CPU mode
  python scripts/infer_lstm_real_time.py --device cpu
        """,
    )

    parser.add_argument(
        "--vit-checkpoint",
        type=str,
        default=DEFAULT_CONFIG["vit_checkpoint"],
        help=f"Path to ViT checkpoint (default: {DEFAULT_CONFIG['vit_checkpoint']})",
    )
    parser.add_argument(
        "--lstm-model",
        type=str,
        default=DEFAULT_CONFIG["lstm_model_path"],
        help=f"Path to LSTM model (default: {DEFAULT_CONFIG['lstm_model_path']})",
    )
    parser.add_argument(
        "--confidence-threshold",
        type=float,
        default=DEFAULT_CONFIG["confidence_threshold"],
        help=f"Confidence threshold for predictions (default: {DEFAULT_CONFIG['confidence_threshold']})",
    )
    parser.add_argument(
        "--device",
        type=str,
        choices=["cuda", "cpu"],
        default="cuda" if torch.cuda.is_available() else "cpu",
        help="Device to use (cuda or cpu)",
    )
    parser.add_argument(
        "--camera",
        type=int,
        default=DEFAULT_CONFIG["camera_index"],
        help="Camera index (default: 0)",
    )

    args = parser.parse_args()

    try:
        demo = LiveGestureDemo(
            vit_checkpoint=args.vit_checkpoint,
            lstm_model_path=args.lstm_model,
            confidence_threshold=args.confidence_threshold,
            device=args.device,
            camera_index=args.camera,
        )
        demo.run()
    except FileNotFoundError as e:
        logger.error(f"File not found: {e}")
        sys.exit(1)
    except Exception as e:
        logger.error(f"Demo failed: {e}", exc_info=True)
        sys.exit(1)


if __name__ == "__main__":
    main()
