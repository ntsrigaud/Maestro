#!/usr/bin/env python3
"""
Clean Baseline Gesture Recognition Demo

Demonstrates real-time gesture recognition using the clean baseline ViT model.
Architecture matches training exactly:
- Center-crop preprocessing (no hand detection)
- Standard ViT-B/16 + linear head
- Full-frame inference aligned with HaGRID training distribution
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
from src.recognition_engine.preprocessing import center_crop_frame, get_crop_coordinates
from src.shared.logger import get_logger

# --- LSTM-ViT imports ---
from src.recognition_engine.temporal.dynamic_gesture_recognizer import DynamicGestureRecognizer

logger = get_logger(__name__)

# Default Hugging Face repository
DEFAULT_HF_REPO = "YOUR_USERNAME/hagrid-vit-gesture"
DEFAULT_MODEL_DIR = project_root / "checkpoints"

# Visualization colors (BGR format)
COLOR_BBOX = (0, 255, 0)  # Green - high confidence
COLOR_BBOX_LOW = (0, 165, 255)  # Orange - low confidence
COLOR_TEXT = (255, 255, 255)
COLOR_BG = (0, 0, 0)

# HaGRID gesture classes (alphabetically sorted as in training)
# This is the standard 18-class set from the full HaGRID dataset
HAGRID_GESTURE_NAMES = [
    'call', 'dislike', 'fist', 'four', 'like', 'mute', 'ok', 'one',
    'palm', 'peace', 'peace_inverted', 'rock', 'stop', 'stop_inverted',
    'three', 'three2', 'two_up', 'two_up_inverted'
]


def load_class_names(checkpoint_path: Path) -> list:
    """
    Load class names dynamically from checkpoint.
    
    Args:
        checkpoint_path: Path to model checkpoint
        
    Returns:
        List of class names in correct order
        
    Raises:
        ValueError: If class names cannot be determined
    """
    checkpoint = torch.load(checkpoint_path, map_location='cpu')
    
    # Try to get class names from checkpoint metadata
    if 'class_names' in checkpoint:
        class_names = checkpoint['class_names']
        logger.info(f"Loaded {len(class_names)} classes from checkpoint metadata")
        return class_names
    
    # Fallback: use HaGRID gesture names if class count matches
    state_dict = checkpoint.get('model_state_dict', checkpoint)
    if 'head.weight' in state_dict:
        num_classes = state_dict['head.weight'].shape[0]
        
        # If model has 18 classes, use the standard HaGRID gesture names
        if num_classes == len(HAGRID_GESTURE_NAMES):
            preview_count = 3
            logger.warning(
                f"No class_names in checkpoint. Using standard HaGRID gesture names "
                f"({num_classes} classes). Ensure your model was trained with HaGRID "
                f"in alphabetical order: {', '.join(HAGRID_GESTURE_NAMES[:preview_count])}..."
            )
            return HAGRID_GESTURE_NAMES
        
        # For non-standard class counts, use generic labels
        logger.warning(
            f"No class_names in checkpoint. Inferred {num_classes} classes from model architecture. "
            f"Class count doesn't match standard HaGRID (18 classes). "
            f"Using generic labels (class_0, class_1, ...). "
            f"Consider re-saving checkpoint with 'class_names' metadata."
        )
        return [f"class_{i}" for i in range(num_classes)]
    
    raise ValueError(
        "Cannot determine class names from checkpoint. "
        "Checkpoint must contain 'class_names' metadata or 'head.weight' layer."
    )


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
        show_all_predictions: bool = True,
        enable_dynamic_gesture: bool = True,
        lstm_seq_len: int = 12,
    ):
        """
        Initialize demo.

        Args:
            checkpoint_path: Path to best_model.pth (if None, downloads from HF)
            hf_repo_id: Hugging Face repo ID (used if checkpoint_path is None)
            camera_index: Camera device index
            confidence_threshold: Minimum confidence for display
            device: 'cuda', 'mps', 'cpu', or None (auto-detect)
            show_all_predictions: If True, always show predictions. If False, only show high-confidence predictions.
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
        self.show_all_predictions = show_all_predictions

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
        self.model = None
        self.transform = None
        self.gesture_classes = None  # Loaded dynamically from checkpoint

        # Statistics
        self.fps_history = deque(maxlen=30)
        self.last_prediction = None  # Track for display
        self.last_confidence = 0.0

        # --- Dynamic gesture (LSTM-ViT) ---
        self.enable_dynamic_gesture = enable_dynamic_gesture
        self.lstm_seq_len = lstm_seq_len
        self.frame_buffer = deque(maxlen=lstm_seq_len)
        self.dynamic_gesture_result = None
        self.dynamic_gesture_conf = 0.0
        self.dynamic_gesture_model = None

    def initialize(self):
        """Initialize all components."""
        logger.info(f"Initializing clean baseline demo...")
        logger.info(f"Device: {self.device}")

        # Input manager
        self.input_mgr = InputManager(camera_index=self.camera_index)
        self.input_mgr._convert_to_rgb = False  # Keep BGR for OpenCV

        # Start continuous capture
        if not self.input_mgr.start_continuous_capture():
            raise RuntimeError(
                f"Failed to open camera {self.camera_index}. "
                f"Please check camera is connected and not in use by another application."
            )

        logger.info("✓ Camera ready")

        # Load class names dynamically from checkpoint
        logger.info("Loading class names from checkpoint...")
        self.gesture_classes = load_class_names(self.checkpoint_path)
        logger.info(f"✓ Loaded {len(self.gesture_classes)} gesture classes")
        logger.info(f"  Classes: {', '.join(self.gesture_classes[:5])}{'...' if len(self.gesture_classes) > 5 else ''}")
        
        # Load model
        logger.info(f"Loading model from {self.checkpoint_path}")
        checkpoint = torch.load(self.checkpoint_path, map_location=self.device)

        # Load clean baseline ViT (standard architecture)
        self.model = timm.create_model(
            "vit_base_patch16_224", pretrained=False, num_classes=len(self.gesture_classes)
        )

        # Handle both full checkpoint and state_dict-only formats
        if "model_state_dict" in checkpoint:
            self.model.load_state_dict(checkpoint["model_state_dict"])
            logger.info(f"Model from epoch {checkpoint.get('epoch', 'unknown')}")
        else:
            self.model.load_state_dict(checkpoint)

        self.model = self.model.to(self.device)
        self.model.eval()

        logger.info(f"✓ Model loaded (ViT-B/16, {len(self.gesture_classes)} classes)")

        # --- Initialize LSTM-ViT dynamic gesture recognizer ---
        if self.enable_dynamic_gesture:
            try:
                lstm_params = dict(input_dim=768, hidden_dim=256, num_layers=2, num_classes=len(self.gesture_classes))
                self.dynamic_gesture_model = DynamicGestureRecognizer(
                    vit_model_name="vit_base_patch16_224",
                    lstm_params=lstm_params,
                    gesture_labels=self.gesture_classes,
                    device=self.device,
                )
                logger.info("✓ Dynamic gesture recognizer (LSTM-ViT) initialized")
            except Exception as e:
                logger.warning(f"Could not initialize dynamic gesture recognizer: {e}")
                self.enable_dynamic_gesture = False
        
        # Verify label alignment
        if 'model_state_dict' in checkpoint:
            state_dict = checkpoint['model_state_dict']
        else:
            state_dict = checkpoint
        
        model_num_classes = state_dict['head.weight'].shape[0]
        if model_num_classes != len(self.gesture_classes):
            logger.error(
                f"LABEL MISMATCH: Model has {model_num_classes} outputs but "
                f"{len(self.gesture_classes)} class names loaded!"
            )
            raise ValueError("Class count mismatch between model and labels")
        
        logger.info(f"✓ Label alignment verified ({model_num_classes} classes)")

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

    def classify_gesture(self, frame_rgb: np.ndarray) -> tuple:
        """
        Classify gesture from preprocessed frame.

        Args:
            frame_rgb: Preprocessed frame (224x224, RGB)

        Returns:
            (gesture_name, confidence)
        """
        # Convert to PIL Image
        pil_img = Image.fromarray(frame_rgb)

        # Apply transform
        img_tensor = self.transform(pil_img).unsqueeze(0).to(self.device)

        # Inference (single forward pass, no TTA)
        with torch.no_grad():
            output = self.model(img_tensor)
            probs = torch.softmax(output, dim=1)[0]

        # Get prediction
        pred_idx = output.argmax(dim=1).item()
        confidence = probs[pred_idx].item()
        gesture = self.gesture_classes[pred_idx]
        
        # Store for display
        self.last_prediction = gesture
        self.last_confidence = confidence

        return gesture, confidence

    def process_frame(self, frame: np.ndarray) -> np.ndarray:
        """Process single frame with center-crop preprocessing."""
        # Get crop coordinates for visualization
        h, w = frame.shape[:2]
        x1, y1, x2, y2 = get_crop_coordinates((h, w))
        
        # Apply center-crop preprocessing (matches training exactly)
        preprocessed = center_crop_frame(frame, target_size=224, bgr_to_rgb=True)
        
        # Classify gesture
        gesture, confidence = self.classify_gesture(preprocessed)

        # --- Dynamic gesture (LSTM-ViT) ---
        if self.enable_dynamic_gesture and self.dynamic_gesture_model is not None:
            # Add current frame to buffer (convert to PIL)
            pil_frame = Image.fromarray(preprocessed)
            self.frame_buffer.append(pil_frame)
            # Only run LSTM if buffer is full
            if len(self.frame_buffer) == self.lstm_seq_len:
                result = self.dynamic_gesture_model.predict(list(self.frame_buffer))
                self.dynamic_gesture_result = result['description']
                self.dynamic_gesture_conf = result['confidence']
        
        # Determine if we should show prediction based on configuration
        is_high_conf = confidence >= self.confidence_threshold
        should_display_prediction = self.show_all_predictions or is_high_conf
        
        # Choose color based on confidence
        bbox_color = COLOR_BBOX if is_high_conf else COLOR_BBOX_LOW
        
        # Always draw crop region to show system is active
        cv2.rectangle(frame, (x1, y1), (x2, y2), bbox_color, 3)
        
        if not should_display_prediction:
            # Skip prediction labels when filtering is enabled and confidence is low
            # (crop region is already drawn to show system is active)
            return frame
        
        # ==============================================================
        # PRIMARY LABEL: Top-left of frame (always visible)
        # ==============================================================
        primary_label = f"GESTURE: {gesture.upper()}"
        conf_label = f"CONFIDENCE: {confidence:.1%}"

        # Large, bold text for primary label
        primary_font_scale = 1.2
        primary_thickness = 3
        conf_font_scale = 0.9
        conf_thickness = 2

        # Position at top-left with padding
        label_x = 20
        label_y = 60

        # Draw background panel for primary labels
        panel_height = 140 if self.enable_dynamic_gesture else 100
        cv2.rectangle(
            frame,
            (10, 40),
            (w - 10, 40 + panel_height),
            COLOR_BG,
            -1
        )

        # Draw primary label (gesture name)
        cv2.putText(
            frame,
            primary_label,
            (label_x, label_y),
            cv2.FONT_HERSHEY_DUPLEX,
            primary_font_scale,
            bbox_color,  # Use confidence color
            primary_thickness,
        )

        # Draw confidence
        cv2.putText(
            frame,
            conf_label,
            (label_x, label_y + 40),
            cv2.FONT_HERSHEY_DUPLEX,
            conf_font_scale,
            COLOR_TEXT,
            conf_thickness,
        )

        # Add status indicator
        status = "✓ HIGH CONFIDENCE" if is_high_conf else "⚠ LOW CONFIDENCE"
        status_color = COLOR_BBOX if is_high_conf else COLOR_BBOX_LOW
        cv2.putText(
            frame,
            status,
            (label_x, label_y + 75),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            status_color,
            2,
        )

        # --- Dynamic gesture label ---
        if self.enable_dynamic_gesture and self.dynamic_gesture_result:
            dyn_label = f"DYNAMIC: {self.dynamic_gesture_result}"
            dyn_conf = f"DYN CONF: {self.dynamic_gesture_conf:.1%}"
            cv2.putText(
                frame,
                dyn_label,
                (label_x, label_y + 110),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.8,
                (0, 255, 255),
                2,
            )
            cv2.putText(
                frame,
                dyn_conf,
                (label_x, label_y + 135),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (0, 255, 255),
                2,
            )
        
        # ==============================================================
        # SECONDARY LABEL: Above crop box (if space available)
        # ==============================================================
        if y1 > 50:  # Only if there's space above crop box
            compact_label = f"{gesture}: {confidence:.0%}"
            cv2.putText(
                frame,
                compact_label,
                (x1 + 10, y1 - 15),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                bbox_color,
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
        logger.info("\n" + "="*70)
        logger.info("LIVE PREDICTIONS (updated every frame):")
        logger.info("="*70)

        prev_time = time.time()
        frame_count = 0
        log_interval = 10  # Log every N frames

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
                
                # Log predictions to console periodically
                frame_count += 1
                if frame_count % log_interval == 0 and self.last_prediction:
                    conf_status = "HIGH" if self.last_confidence >= self.confidence_threshold else "LOW"
                    logger.info(
                        f"[Frame {frame_count:04d}] Gesture: {self.last_prediction.upper():20s} | "
                        f"Confidence: {self.last_confidence:6.1%} | Status: {conf_status}"
                    )

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
        "--no-show-all-predictions",
        dest="show_all_predictions",
        action="store_false",
        default=True,
        help="Only show predictions when confidence exceeds threshold (cleaner UX). By default, all predictions are shown.",
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
        show_all_predictions=args.show_all_predictions,
    )

    demo.run()


if __name__ == "__main__":
    main()
