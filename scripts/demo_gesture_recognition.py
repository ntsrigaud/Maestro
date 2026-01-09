#!/usr/bin/env python3
"""
Clean Baseline Gesture Recognition Demo

Demonstrates real-time gesture recognition using the clean baseline ViT model.
Architecture matches finetune_vit_hagrid_CLEAN_BASELINE.ipynb exactly:
- Single preprocessing pipeline (no augmentation)
- Standard ViT-B/16 + linear head
- No TTA, no experimental components
"""

import sys
import time
from pathlib import Path
from collections import deque

import numpy as np
import cv2
from PIL import Image

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

import torch
import timm
from torchvision import transforms

from src.input_manager import InputManager
from src.recognition_engine.hand_detection import HandDetector, HandCropper
from src.shared.logger import get_logger

logger = get_logger(__name__)

# Default Hugging Face repository
DEFAULT_HF_REPO = "YOUR_USERNAME/hagrid-vit-gesture"
DEFAULT_MODEL_DIR = project_root / "checkpoints"

# Gesture classes (alphabetically sorted - MUST match training)
GESTURE_CLASSES = [
    "call",
    "dislike",
    "fist",
    "four",
    "like",
    "mute",
    "ok",
    "one",
    "palm",
    "peace",
    "peace_inverted",
    "rock",
    "stop",
    "stop_inverted",
    "three",
    "three2",
    "two_up",
    "two_up_inverted",
]

# Visualization colors (BGR format)
COLOR_BBOX = (0, 255, 0)
COLOR_TEXT = (255, 255, 255)
COLOR_BG = (0, 0, 0)


def download_model_from_hf(repo_id: str, cache_dir: Path) -> Path:
    """
    Download model from Hugging Face Hub if not cached.

    Args:
        repo_id: Hugging Face repository ID
        cache_dir: Local cache directory

    Returns:
        Path to model checkpoint
    """
    cache_dir.mkdir(parents=True, exist_ok=True)
    model_path = cache_dir / "best_model.pth"

    if model_path.exists():
        logger.info(f"Model found in cache: {model_path}")
        return model_path

    logger.info(f"Downloading model from Hugging Face: {repo_id}")

    try:
        from huggingface_hub import hf_hub_download

        downloaded_model = hf_hub_download(
            repo_id=repo_id,
            filename="best_model.pth",
            cache_dir=cache_dir,
            force_download=False,
        )

        import shutil

        shutil.copy(downloaded_model, model_path)

        logger.info(f"Model downloaded to: {model_path}")
        return model_path

    except ImportError:
        raise RuntimeError(
            "huggingface_hub not installed. Install with: pip install huggingface_hub"
        )
    except Exception as e:
        raise RuntimeError(
            f"Failed to download model from {repo_id}: {e}\n"
            f"Manual download: https://huggingface.co/{repo_id}"
        )


class GestureRecognitionDemo:
    """Clean baseline gesture recognition demo."""

    def __init__(
        self,
        checkpoint_path: str = None,
        hf_repo_id: str = None,
        camera_index: int = 0,
        confidence_threshold: float = 0.5,
        device: str = None,
    ):
        """
        Initialize demo.

        Args:
            checkpoint_path: Path to best_model.pth (if None, downloads from HF)
            hf_repo_id: Hugging Face repo ID (used if checkpoint_path is None)
            camera_index: Camera device index
            confidence_threshold: Minimum confidence for display
            device: 'cuda', 'mps', 'cpu', or None (auto-detect)
        """
        # Handle model path
        if checkpoint_path is None:
            if hf_repo_id is None:
                hf_repo_id = DEFAULT_HF_REPO
            logger.info(f"No checkpoint provided, downloading from HF: {hf_repo_id}")
            checkpoint_path = download_model_from_hf(hf_repo_id, DEFAULT_MODEL_DIR)

        self.checkpoint_path = Path(checkpoint_path)
        if not self.checkpoint_path.exists():
            raise FileNotFoundError(
                f"Model checkpoint not found: {self.checkpoint_path}\n"
                f"Train model using: notebooks/finetune_vit_hagrid_CLEAN_BASELINE.ipynb\n"
                f"Or set --hf-repo to download from Hugging Face"
            )

        self.camera_index = camera_index
        self.confidence_threshold = confidence_threshold

        # Auto-detect device
        if device is None:
            if torch.backends.mps.is_available():
                self.device = torch.device("mps")
            elif torch.cuda.is_available():
                self.device = torch.device("cuda")
            else:
                self.device = torch.device("cpu")
        else:
            self.device = torch.device(device)

        # Components
        self.input_mgr = None
        self.detector = None
        self.cropper = None
        self.model = None
        self.transform = None

        # Statistics
        self.fps_history = deque(maxlen=30)

    def initialize(self):
        """Initialize all components."""
        logger.info(f"Initializing clean baseline demo...")
        logger.info(f"Device: {self.device}")

        # Input manager
        self.input_mgr = InputManager(camera_index=self.camera_index)
        self.input_mgr._convert_to_rgb = False  # Keep BGR for MediaPipe

        # Start continuous capture (handles camera opening and frame thread)
        # This method internally waits for first frame with 1 second timeout
        if not self.input_mgr.start_continuous_capture():
            raise RuntimeError(
                f"Failed to open camera {self.camera_index}. "
                f"Please check camera is connected and not in use by another application."
            )

        logger.info("✓ Camera ready")

        # Hand detector
        self.detector = HandDetector(
            max_num_hands=2, min_detection_confidence=0.5, min_tracking_confidence=0.5
        )

        # Hand cropper
        self.cropper = HandCropper(
            target_size=(224, 224),
            padding=0.2,
            square_crop=True,
            normalize=False,
            bgr_to_rgb=True,
        )

        # Load model
        logger.info(f"Loading model from {self.checkpoint_path}")
        checkpoint = torch.load(self.checkpoint_path, map_location=self.device)

        # Load clean baseline ViT (standard architecture)
        self.model = timm.create_model(
            "vit_base_patch16_224", pretrained=False, num_classes=len(GESTURE_CLASSES)
        )

        # Handle both full checkpoint and state_dict-only formats
        if "model_state_dict" in checkpoint:
            self.model.load_state_dict(checkpoint["model_state_dict"])
            logger.info(f"Model from epoch {checkpoint.get('epoch', 'unknown')}")
        else:
            self.model.load_state_dict(checkpoint)

        self.model = self.model.to(self.device)
        self.model.eval()

        logger.info(f"✓ Model loaded (ViT-B/16, {len(GESTURE_CLASSES)} classes)")

        # Preprocessing transform (MUST match training exactly)
        self.transform = transforms.Compose(
            [
                transforms.Resize((224, 224)),
                transforms.ToTensor(),
                transforms.Normalize(
                    mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
                ),
            ]
        )

        logger.info("✓ Initialization complete")

    def classify_gesture(self, hand_roi: np.ndarray) -> tuple:
        """
        Classify gesture from cropped hand image.

        Args:
            hand_roi: Cropped hand image (224x224, RGB)

        Returns:
            (gesture_name, confidence)
        """
        # Convert to PIL Image
        pil_img = Image.fromarray(hand_roi)

        # Apply transform
        img_tensor = self.transform(pil_img).unsqueeze(0).to(self.device)

        # Inference (single forward pass, no TTA)
        with torch.no_grad():
            output = self.model(img_tensor)
            probs = torch.softmax(output, dim=1)[0]

        # Get prediction
        pred_idx = output.argmax(dim=1).item()
        confidence = probs[pred_idx].item()
        gesture = GESTURE_CLASSES[pred_idx]

        return gesture, confidence

    def process_frame(self, frame: np.ndarray) -> np.ndarray:
        """Process single frame."""
        # Detect hands
        detections = self.detector.detect(frame)

        # Process each detection
        for detection in detections:
            # Extract bbox coordinates (x, y, width, height)
            x, y, w, h = detection.bbox
            x1, y1 = x, y
            x2, y2 = x + w, y + h

            # Crop hand region using the detection object
            hand_roi = self.cropper.crop_hand(frame, detection)
            if hand_roi is None or hand_roi.image is None:
                continue

            # Classify gesture using the cropped image
            gesture, confidence = self.classify_gesture(hand_roi.image)

            # Draw results if confidence above threshold
            if confidence >= self.confidence_threshold:
                # Bounding box
                cv2.rectangle(frame, (x1, y1), (x2, y2), COLOR_BBOX, 2)

                # Label
                label = f"{gesture}: {confidence:.2f}"
                (label_w, label_h), _ = cv2.getTextSize(
                    label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2
                )
                cv2.rectangle(
                    frame, (x1, y1 - label_h - 10), (x1 + label_w, y1), COLOR_BG, -1
                )
                cv2.putText(
                    frame,
                    label,
                    (x1, y1 - 5),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.6,
                    COLOR_TEXT,
                    2,
                )

        return frame

    def run(self):
        """Run demo loop."""
        try:
            self.initialize()
        except RuntimeError as e:
            logger.error(f"Initialization failed: {e}")
            logger.info("Demo cannot run without camera access")
            return

        logger.info("Starting demo...")
        logger.info("Press 'q' to quit")

        prev_time = time.time()

        try:
            while True:
                # Get frame
                frame_data = self.input_mgr.get_latest_frame()
                if frame_data is None or frame_data.frame is None:
                    logger.warning("Failed to get frame")
                    time.sleep(0.1)
                    continue

                frame = frame_data.frame

                # Process frame
                processed_frame = self.process_frame(frame)

                # Calculate FPS
                curr_time = time.time()
                fps = 1.0 / (curr_time - prev_time) if curr_time != prev_time else 0
                self.fps_history.append(fps)
                avg_fps = np.mean(self.fps_history) if self.fps_history else 0
                prev_time = curr_time

                # Display FPS
                cv2.putText(
                    processed_frame,
                    f"FPS: {avg_fps:.1f}",
                    (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    1,
                    COLOR_TEXT,
                    2,
                )

                # Show frame
                cv2.imshow("Gesture Recognition - Clean Baseline", processed_frame)

                # Check for quit
                if cv2.waitKey(1) & 0xFF == ord("q"):
                    break

        except KeyboardInterrupt:
            logger.info("Interrupted by user")
        finally:
            self.cleanup()

    def cleanup(self):
        """Cleanup resources."""
        if self.input_mgr:
            try:
                self.input_mgr.stop_capture()
            except Exception as e:
                logger.warning(f"Error during camera cleanup: {e}")
        cv2.destroyAllWindows()
        logger.info("Demo stopped")


def main():
    """Main entry point."""
    import argparse

    parser = argparse.ArgumentParser(
        description="Clean Baseline Gesture Recognition Demo"
    )
    parser.add_argument(
        "--checkpoint",
        type=str,
        default=None,
        help="Path to model checkpoint (downloads from HF if not provided)",
    )
    parser.add_argument(
        "--hf-repo",
        type=str,
        default=None,
        help="Hugging Face repository ID (e.g., username/repo-name)",
    )
    parser.add_argument("--camera", type=int, default=0, help="Camera device index")
    parser.add_argument(
        "--confidence", type=float, default=0.5, help="Minimum confidence threshold"
    )
    parser.add_argument(
        "--device",
        type=str,
        choices=["cuda", "mps", "cpu", "auto"],
        default="auto",
        help="Compute device",
    )

    args = parser.parse_args()

    # Check for local checkpoint first
    if args.checkpoint is None:
        local_checkpoint = Path("./checkpoints/best_model.pth")
        if local_checkpoint.exists():
            args.checkpoint = str(local_checkpoint)
            logger.info(f"Using local checkpoint: {args.checkpoint}")

    device = None if args.device == "auto" else args.device

    demo = GestureRecognitionDemo(
        checkpoint_path=args.checkpoint,
        hf_repo_id=args.hf_repo,
        camera_index=args.camera,
        confidence_threshold=args.confidence,
        device=device,
    )

    demo.run()


if __name__ == "__main__":
    main()
