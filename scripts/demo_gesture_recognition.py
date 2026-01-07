#!/usr/bin/env python3
"""
Real-time Gesture Recognition Demo

This script demonstrates end-to-end gesture recognition by:
1. Capturing live camera feed via InputManager
2. Detecting hands using MediaPipe HandDetector
3. Cropping hand regions using HandCropper
4. Classifying gestures using fine-tuned ViT-B/16 model
5. Displaying results with bounding boxes and gesture labels
"""

import sys
import time
import numpy as np
import cv2
from pathlib import Path
from collections import deque

# Add project root to path for imports
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

import torch
import torch.nn as nn
import timm
from torchvision import transforms
from PIL import Image

from src.input_manager import InputManager, FrameData
from src.recognition_engine.hand_detection import HandDetector, HandCropper
from src.shared.logger import get_logger

logger = get_logger(__name__)


# ============================================================================
# Model Architecture Definitions (must match training notebook)
# ============================================================================

class ImprovedClassificationHead(nn.Module):
    """
    Enhanced classification head used in training.
    3-layer architecture: 768 → 512 → 256 → num_classes
    """
    def __init__(self, in_features, num_classes, hidden_dim=512, dropout=0.3):
        super().__init__()
        self.head = nn.Sequential(
            nn.LayerNorm(in_features),
            nn.Linear(in_features, hidden_dim),
            nn.GELU(),
            nn.BatchNorm1d(hidden_dim),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.GELU(),
            nn.BatchNorm1d(hidden_dim // 2),
            nn.Dropout(dropout / 2),
            nn.Linear(hidden_dim // 2, num_classes)
        )
    
    def forward(self, x):
        return self.head(x)


class EnhancedViT(nn.Module):
    """
    Enhanced ViT wrapper used in training.
    Consists of ViT backbone + custom classification head.
    """
    def __init__(self, backbone, num_classes, hidden_dim=512, dropout=0.3):
        super().__init__()
        self.backbone = backbone
        self.head = ImprovedClassificationHead(
            in_features=backbone.num_features,
            num_classes=num_classes,
            hidden_dim=hidden_dim,
            dropout=dropout
        )
    
    def forward(self, x):
        features = self.backbone(x)
        return self.head(features)

# Hugging Face model repository
DEFAULT_HF_REPO = "YOUR_USERNAME/hagrid-vit-gesture"  # Replace with your HF repo
DEFAULT_MODEL_DIR = project_root / "models" / "gesture_recognition"

# Gesture classes (alphabetically sorted - must match training)
GESTURE_CLASSES = [
    'call', 'dislike', 'fist', 'four', 'like', 'mute', 'ok', 'one',
    'palm', 'peace', 'peace_inverted', 'rock', 'stop', 'stop_inverted',
    'three', 'three2', 'two_up', 'two_up_inverted'
]

# Colors for visualization (BGR format)
COLOR_BBOX_HIGH_CONF = (0, 255, 0)  # Green - high confidence
COLOR_BBOX_MED_CONF = (0, 165, 255)  # Orange - medium confidence
COLOR_BBOX_LOW_CONF = (0, 0, 255)  # Red - low confidence
COLOR_LEFT_HAND = (0, 255, 255)  # Yellow
COLOR_RIGHT_HAND = (255, 0, 255)  # Magenta
COLOR_TEXT = (255, 255, 255)  # White
COLOR_BG = (0, 0, 0)  # Black


def download_model_from_hf(repo_id: str = DEFAULT_HF_REPO, cache_dir: Path = DEFAULT_MODEL_DIR) -> Path:
    """
    Download model from Hugging Face Hub if not already cached.
    
    Args:
        repo_id: Hugging Face repository ID (username/repo-name)
        cache_dir: Local directory to cache downloaded model
    
    Returns:
        Path to downloaded model checkpoint
    """
    cache_dir.mkdir(parents=True, exist_ok=True)
    model_path = cache_dir / "best_model.pth"
    metadata_path = cache_dir / "model_metadata.json"
    
    # Check if model already exists
    if model_path.exists() and metadata_path.exists():
        logger.info(f"✓ Model found in cache: {model_path}")
        return model_path
    
    # Download from Hugging Face
    logger.info(f"Downloading model from Hugging Face: {repo_id}")
    logger.info("This is a one-time download (~330 MB)...")
    
    try:
        from huggingface_hub import hf_hub_download
        
        # Download model weights
        logger.info("Downloading model weights...")
        downloaded_model = hf_hub_download(
            repo_id=repo_id,
            filename="best_model.pth",
            cache_dir=cache_dir,
            force_download=False
        )
        
        # Download metadata
        logger.info("Downloading metadata...")
        downloaded_metadata = hf_hub_download(
            repo_id=repo_id,
            filename="model_metadata.json",
            cache_dir=cache_dir,
            force_download=False
        )
        
        # Copy to cache location
        import shutil
        shutil.copy(downloaded_model, model_path)
        shutil.copy(downloaded_metadata, metadata_path)
        
        logger.info(f"✓ Model downloaded successfully to: {model_path}")
        return model_path
        
    except ImportError:
        logger.error("huggingface_hub not installed. Install with: pip install huggingface_hub")
        raise
    except Exception as e:
        logger.error(f"Failed to download model: {e}")
        logger.info(f"\nManual download instructions:")
        logger.info(f"1. Visit: https://huggingface.co/{repo_id}")
        logger.info(f"2. Download 'best_model.pth' to: {model_path}")
        logger.info(f"3. Download 'model_metadata.json' to: {metadata_path}")
        raise


class GestureRecognitionDemo:
    """Real-time gesture recognition demo application."""

    def __init__(
        self,
        checkpoint_path: str = None,
        hf_repo_id: str = None,
        camera_index: int = 0,
        window_name: str = "Gesture Recognition Demo",
        max_num_hands: int = 2,
        show_cropped: bool = False,
        confidence_threshold: float = 0.5,
        device: str = None
    ):
        """
        Initialize the gesture recognition demo.

        Args:
            checkpoint_path: Path to fine-tuned model checkpoint (best_model.pth).
                           If None, will download from Hugging Face.
            hf_repo_id: Hugging Face repository ID (e.g., 'username/repo-name').
                       Used if checkpoint_path is None.
            camera_index: Camera device index
            window_name: Window title
            max_num_hands: Maximum number of hands to detect
            show_cropped: Whether to show cropped hand regions
            confidence_threshold: Minimum confidence for gesture display
            device: 'cuda', 'mps', 'cpu', or None (auto-detect)
        """
        # Handle model path
        if checkpoint_path is None:
            if hf_repo_id is None:
                hf_repo_id = DEFAULT_HF_REPO
            
            logger.info(f"No checkpoint provided, downloading from Hugging Face: {hf_repo_id}")
            checkpoint_path = download_model_from_hf(repo_id=hf_repo_id)
        
        self.checkpoint_path = Path(checkpoint_path)
        if not self.checkpoint_path.exists():
            raise FileNotFoundError(
                f"Model checkpoint not found: {self.checkpoint_path}\n"
                f"Please either:\n"
                f"  1. Train the model using notebooks/finetune_vit_hagrid.ipynb, or\n"
                f"  2. Download from Hugging Face (set hf_repo_id parameter)"
            )
        
        self.camera_index = camera_index
        self.window_name = window_name
        self.max_num_hands = max_num_hands
        self.show_cropped = show_cropped
        self.confidence_threshold = confidence_threshold

        # Auto-detect device
        if device is None:
            if torch.backends.mps.is_available():
                self.device = torch.device('mps')
            elif torch.cuda.is_available():
                self.device = torch.device('cuda')
            else:
                self.device = torch.device('cpu')
        else:
            self.device = torch.device(device)

        # Initialize components
        self.input_mgr = None
        self.detector = None
        self.cropper = None
        self.model = None
        self.transform = None

        # Statistics
        self.frame_count = 0
        self.detection_count = 0
        self.classification_count = 0
        self.fps_history = deque(maxlen=30)

        # Gesture smoothing (temporal averaging)
        self.gesture_history = {}  # hand_id -> deque of (gesture, confidence)
        self.history_length = 5

    def initialize(self):
        """Initialize all components."""
        print("Initializing gesture recognition demo...")
        print(f"Device: {self.device}")

        # Initialize input manager (keep BGR format for MediaPipe)
        self.input_mgr = InputManager(camera_index=self.camera_index)
        self.input_mgr._convert_to_rgb = False  # MediaPipe expects BGR

        # Initialize hand detector
        self.detector = HandDetector(
            max_num_hands=self.max_num_hands,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5,
            model_complexity=1,
        )

        # Initialize hand cropper (with preprocessing for ViT)
        self.cropper = HandCropper(
            target_size=(224, 224),
            padding=0.2,
            square_crop=True,
            normalize=False,  # We'll normalize with torchvision transforms
            bgr_to_rgb=True,  # Convert BGR to RGB for model
        )

        # Load gesture classification model
        print(f"Loading model from {self.checkpoint_path}...")
        
        # Load checkpoint
        checkpoint = torch.load(self.checkpoint_path, map_location=self.device)
        
        # Detect architecture by inspecting state_dict keys
        state_dict_keys = list(checkpoint.keys())
        is_enhanced_vit = any(key.startswith('backbone.') or key.startswith('head.') 
                             for key in state_dict_keys)
        
        if is_enhanced_vit:
            logger.info("Detected EnhancedViT architecture (backbone + custom head)")
            # Create EnhancedViT architecture
            base_model = timm.create_model('vit_base_patch16_224', pretrained=False, num_classes=0)
            self.model = EnhancedViT(
                backbone=base_model,
                num_classes=len(GESTURE_CLASSES),
                hidden_dim=512,
                dropout=0.3
            )
            self.model.load_state_dict(checkpoint)
        else:
            logger.info("Detected plain ViT architecture")
            # Create plain ViT
            self.model = timm.create_model(
                'vit_base_patch16_224',
                pretrained=False,
                num_classes=len(GESTURE_CLASSES)
            )
            self.model.load_state_dict(checkpoint)
        
        self.model = self.model.to(self.device)
        self.model.eval()
        
        logger.info(f"✓ Model loaded successfully ({type(self.model).__name__})")

        # Define preprocessing transform (matches training)
        self.transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                               std=[0.229, 0.224, 0.225])
        ])

        print("✓ Initialization complete")

    def classify_gesture(self, hand_roi_img: np.ndarray) -> tuple:
        """
        Classify gesture from cropped hand image.

        Args:
            hand_roi_img: Cropped hand image (224x224, RGB)

        Returns:
            Tuple of (gesture_name, confidence, top_3_predictions)
        """
        # Convert numpy array to PIL Image
        pil_img = Image.fromarray(hand_roi_img)

        # Apply transforms
        img_tensor = self.transform(pil_img).unsqueeze(0).to(self.device)

        # Inference
        with torch.no_grad():
            output = self.model(img_tensor)
            probs = torch.softmax(output, dim=1)[0]

        # Get top prediction
        pred_idx = output.argmax(dim=1).item()
        confidence = probs[pred_idx].item()
        gesture = GESTURE_CLASSES[pred_idx]

        # Get top-3 predictions
        top_probs, top_indices = torch.topk(probs, k=3)
        top_3 = [
            (GESTURE_CLASSES[idx.item()], prob.item())
            for idx, prob in zip(top_indices, top_probs)
        ]

        return gesture, confidence, top_3

    def smooth_gesture(self, hand_id: str, gesture: str, confidence: float) -> tuple:
        """
        Apply temporal smoothing to gesture predictions.

        Args:
            hand_id: Unique identifier for hand (e.g., "Left_0")
            gesture: Current gesture prediction
            confidence: Current confidence score

        Returns:
            Tuple of (smoothed_gesture, smoothed_confidence)
        """
        # Initialize history for new hands
        if hand_id not in self.gesture_history:
            self.gesture_history[hand_id] = deque(maxlen=self.history_length)

        # Add current prediction to history
        self.gesture_history[hand_id].append((gesture, confidence))

        # If not enough history, return current prediction
        if len(self.gesture_history[hand_id]) < 3:
            return gesture, confidence

        # Vote for most common gesture in history
        gesture_votes = {}
        for hist_gesture, hist_conf in self.gesture_history[hand_id]:
            if hist_gesture not in gesture_votes:
                gesture_votes[hist_gesture] = []
            gesture_votes[hist_gesture].append(hist_conf)

        # Get gesture with highest average confidence
        best_gesture = max(
            gesture_votes.keys(),
            key=lambda g: sum(gesture_votes[g]) / len(gesture_votes[g])
        )
        avg_confidence = sum(gesture_votes[best_gesture]) / len(gesture_votes[best_gesture])

        return best_gesture, avg_confidence

    def draw_gesture_bbox(self, frame: np.ndarray, detection, gesture: str, confidence: float):
        """Draw bounding box with gesture label."""
        x, y, w, h = detection.bbox

        # Choose color based on confidence
        if confidence >= 0.8:
            color = COLOR_BBOX_HIGH_CONF
        elif confidence >= 0.5:
            color = COLOR_BBOX_MED_CONF
        else:
            color = COLOR_BBOX_LOW_CONF

        # Draw bounding box
        cv2.rectangle(frame, (x, y), (x + w, y + h), color, 2)

        # Prepare label text
        handedness_label = detection.handedness
        gesture_label = f"{gesture} ({confidence:.0%})"

        # Draw background for labels
        label_height = 60
        cv2.rectangle(
            frame,
            (x, y - label_height),
            (x + w, y),
            color,
            -1
        )

        # Draw handedness
        cv2.putText(
            frame,
            handedness_label,
            (x + 5, y - 35),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            COLOR_TEXT,
            2
        )

        # Draw gesture
        cv2.putText(
            frame,
            gesture_label,
            (x + 5, y - 10),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            COLOR_TEXT,
            2
        )

    def draw_top_predictions(self, frame: np.ndarray, detection, top_3: list):
        """Draw top-3 predictions next to hand."""
        x, y, w, h = detection.bbox

        # Position to the right of bounding box
        text_x = x + w + 10
        text_y = y + 20

        # Draw semi-transparent background
        overlay = frame.copy()
        cv2.rectangle(
            overlay,
            (text_x - 5, text_y - 15),
            (text_x + 180, text_y + 65),
            (0, 0, 0),
            -1
        )
        cv2.addWeighted(overlay, 0.6, frame, 0.4, 0, frame)

        # Draw top-3 predictions
        cv2.putText(
            frame,
            "Top 3:",
            (text_x, text_y),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.4,
            COLOR_TEXT,
            1
        )

        for i, (gest, conf) in enumerate(top_3):
            y_offset = text_y + 20 + (i * 20)
            text = f"{i+1}. {gest}: {conf:.0%}"
            cv2.putText(
                frame,
                text,
                (text_x, y_offset),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.4,
                COLOR_TEXT,
                1
            )

    def draw_stats(self, frame: np.ndarray, fps: float, num_hands: int):
        """Draw statistics overlay."""
        stats = [
            f"FPS: {fps:.1f}",
            f"Hands: {num_hands}",
            f"Gestures: {self.classification_count}",
            f"Device: {self.device.type}",
        ]

        y_offset = 30
        for stat in stats:
            cv2.putText(
                frame,
                stat,
                (10, y_offset),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                COLOR_TEXT,
                2
            )
            y_offset += 30

    def draw_help(self, frame: np.ndarray):
        """Draw help text at bottom."""
        height = frame.shape[0]
        help_texts = [
            "Q/ESC: Quit  |  C: Toggle cropped view  |  R: Reset stats"
        ]

        y_offset = height - 20
        for text in help_texts:
            cv2.putText(
                frame,
                text,
                (10, y_offset),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                COLOR_TEXT,
                1
            )
            y_offset += 25

    def show_cropped_hands(self, hand_rois: list, gestures: list):
        """Display cropped hand regions with gesture labels in separate windows."""
        for i, (hand_roi, (gesture, conf, _)) in enumerate(zip(hand_rois, gestures)):
            if hand_roi.image is None:
                continue

            # Get image and convert RGB back to BGR for display
            img = cv2.cvtColor(hand_roi.image, cv2.COLOR_RGB2BGR)

            # Add gesture label to image
            img_with_label = img.copy()
            cv2.putText(
                img_with_label,
                f"{gesture} ({conf:.0%})",
                (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (0, 255, 0),
                2
            )

            # Window name includes handedness
            window_name = f"Hand {i+1} - {hand_roi.detection.handedness}"
            cv2.imshow(window_name, img_with_label)

    def calculate_fps(self, prev_time: float, curr_time: float) -> float:
        """Calculate FPS from timestamps."""
        if prev_time is None or curr_time <= prev_time:
            return 0.0

        time_diff = (curr_time - prev_time) / 1_000_000_000  # Convert ns to seconds
        fps = 1.0 / time_diff if time_diff > 0 else 0.0

        self.fps_history.append(fps)
        return sum(self.fps_history) / len(self.fps_history)

    def run(self):
        """Run the gesture recognition demo."""
        self.initialize()

        # Start camera capture
        self.input_mgr.start_continuous_capture()
        print("\n" + "=" * 60)
        print("Gesture Recognition Demo Running")
        print("=" * 60)
        print("\nControls:")
        print("  C - Toggle cropped hand view")
        print("  R - Reset statistics")
        print("  Q/ESC - Quit")
        print()

        prev_timestamp = None

        try:
            while True:
                # Get latest frame
                frame_data = self.input_mgr.get_latest_frame()
                if frame_data is None:
                    time.sleep(0.001)
                    continue

                self.frame_count += 1

                # Calculate FPS
                fps = self.calculate_fps(prev_timestamp, frame_data.timestamp)
                prev_timestamp = frame_data.timestamp

                # Detect hands
                detections = self.detector.detect(
                    frame_data.frame, frame_data.timestamp
                )
                self.detection_count += len(detections)

                # Create visualization frame
                vis_frame = frame_data.frame.copy()

                # Process each detected hand
                gestures = []
                if detections:
                    # Crop hand regions
                    hand_rois = self.cropper.crop_multiple(frame_data.frame, detections)

                    # Classify gestures
                    for hand_roi, detection in zip(hand_rois, detections):
                        if hand_roi.image is not None:
                            # Get gesture prediction
                            gesture, confidence, top_3 = self.classify_gesture(
                                hand_roi.image
                            )

                            # Apply temporal smoothing
                            hand_id = f"{detection.handedness}_{detection.bbox}"
                            gesture, confidence = self.smooth_gesture(
                                hand_id, gesture, confidence
                            )

                            gestures.append((gesture, confidence, top_3))
                            self.classification_count += 1

                            # Draw bounding box and gesture label
                            if confidence >= self.confidence_threshold:
                                self.draw_gesture_bbox(
                                    vis_frame, detection, gesture, confidence
                                )
                                # Optionally draw top-3 predictions
                                # self.draw_top_predictions(vis_frame, detection, top_3)
                        else:
                            gestures.append(("Unknown", 0.0, []))

                    # Show cropped hands if enabled
                    if self.show_cropped:
                        self.show_cropped_hands(hand_rois, gestures)

                # Draw statistics
                self.draw_stats(vis_frame, fps, len(detections))

                # Draw help
                self.draw_help(vis_frame)

                # Show main window
                cv2.imshow(self.window_name, vis_frame)

                # Handle keyboard input
                key = cv2.waitKey(1) & 0xFF

                if key == ord('q') or key == 27:  # Q or ESC
                    print("\nQuitting...")
                    break
                elif key == ord('c'):  # Toggle cropped view
                    self.show_cropped = not self.show_cropped
                    print(f"Cropped view: {'ON' if self.show_cropped else 'OFF'}")
                    if not self.show_cropped:
                        cv2.destroyWindow("Hand 1 - Left")
                        cv2.destroyWindow("Hand 1 - Right")
                        cv2.destroyWindow("Hand 2 - Left")
                        cv2.destroyWindow("Hand 2 - Right")
                elif key == ord('r'):  # Reset statistics
                    self.frame_count = 0
                    self.detection_count = 0
                    self.classification_count = 0
                    self.fps_history.clear()
                    self.gesture_history.clear()
                    print("Statistics reset")

        except KeyboardInterrupt:
            print("\nInterrupted by user")
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
        print(f"\nFrames processed: {self.frame_count}")
        print(f"Hands detected: {self.detection_count}")
        print(f"Gestures classified: {self.classification_count}")

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

        avg_fps = sum(self.fps_history) / len(self.fps_history) if self.fps_history else 0
        print(f"\nAverage FPS: {avg_fps:.1f}")

        print("\n✓ Cleanup complete")


def main():
    """Main entry point."""
    import argparse

    parser = argparse.ArgumentParser(
        description="Real-time gesture recognition demo",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Use local checkpoint
  python demo_gesture_recognition.py --checkpoint notebooks/checkpoints/best_model.pth
  
  # Auto-download from Hugging Face
  python demo_gesture_recognition.py --hf-repo username/hagrid-vit-gesture
  
  # Use cached model (if already downloaded)
  python demo_gesture_recognition.py
        """
    )
    parser.add_argument(
        "--checkpoint",
        type=str,
        default=None,
        help="Path to model checkpoint (best_model.pth). If not provided, downloads from Hugging Face."
    )
    parser.add_argument(
        "--hf-repo",
        type=str,
        default=DEFAULT_HF_REPO,
        help=f"Hugging Face repository ID (default: {DEFAULT_HF_REPO})"
    )
    parser.add_argument(
        "--camera",
        type=int,
        default=0,
        help="Camera device index (default: 0)"
    )
    parser.add_argument(
        "--max-hands",
        type=int,
        default=2,
        help="Maximum number of hands to detect (default: 2)"
    )
    parser.add_argument(
        "--show-cropped",
        action="store_true",
        help="Show cropped hand regions in separate windows"
    )
    parser.add_argument(
        "--confidence",
        type=float,
        default=0.5,
        help="Minimum confidence threshold for display (default: 0.5)"
    )
    parser.add_argument(
        "--device",
        type=str,
        choices=['cuda', 'mps', 'cpu'],
        help="Device to use (default: auto-detect)"
    )

    args = parser.parse_args()

    # Create and run demo
    try:
        demo = GestureRecognitionDemo(
            checkpoint_path=args.checkpoint,
            hf_repo_id=args.hf_repo,
            camera_index=args.camera,
            max_num_hands=args.max_hands,
            show_cropped=args.show_cropped,
            confidence_threshold=args.confidence,
            device=args.device
        )
        demo.run()
    except KeyboardInterrupt:
        print("\nInterrupted by user")
    except Exception as e:
        logger.error(f"Error during demo: {e}", exc_info=True)
        print(f"\n❌ Error: {e}")
    finally:
        print("\nDemo finished")


if __name__ == "__main__":
    main()
