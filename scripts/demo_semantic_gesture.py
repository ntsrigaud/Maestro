#!/usr/bin/env python3
"""
Semantic Gesture Recognition Demo
Real-time gesture recognition using semantic understanding (zero-shot capable)
"""

import argparse
import sys
import time
from pathlib import Path
import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image
import timm
from sentence_transformers import SentenceTransformer

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

try:
    import mediapipe as mp
except ImportError:
    print("Error: MediaPipe not installed. Install with: pip install mediapipe")
    sys.exit(1)


class GestureEncoder(nn.Module):
    """Visual encoder for gesture images (ViT-B/16)"""

    def __init__(self, embed_dim=512):
        super().__init__()
        self.vit = timm.create_model("vit_base_patch16_224", pretrained=False)
        self.vit.head = nn.Identity()

        self.projection = nn.Sequential(
            nn.Linear(768, embed_dim), nn.ReLU(), nn.Linear(embed_dim, embed_dim)
        )

    def forward(self, images):
        features = self.vit(images)
        embeddings = self.projection(features)
        embeddings = F.normalize(embeddings, p=2, dim=1)
        return embeddings


class TextEncoder(nn.Module):
    """Text encoder for command descriptions"""

    def __init__(self, embed_dim=512, model_name="all-MiniLM-L6-v2"):
        super().__init__()
        self.sentence_encoder = SentenceTransformer(model_name)

        for param in self.sentence_encoder.parameters():
            param.requires_grad = False

        text_dim = self.sentence_encoder.get_sentence_embedding_dimension()
        self.projection = nn.Sequential(
            nn.Linear(text_dim, embed_dim), nn.ReLU(), nn.Linear(embed_dim, embed_dim)
        )

    def forward(self, texts):
        with torch.no_grad():
            text_features = self.sentence_encoder.encode(
                texts, convert_to_tensor=True, show_progress_bar=False
            )

        text_features = text_features.clone()
        embeddings = self.projection(text_features)
        embeddings = F.normalize(embeddings, p=2, dim=1)
        return embeddings


class SemanticGestureRecognizer:
    """Real-time semantic gesture recognition"""

    # Default gesture commands (can be customized)
    DEFAULT_COMMANDS = [
        # Approval/Positive
        "thumbs up",
        "like",
        "approve",
        "good job",
        "agree",
        # Disapproval/Negative
        "thumbs down",
        "dislike",
        "disapprove",
        "bad",
        "disagree",
        # Stop/Attention
        "stop",
        "wait",
        "halt",
        "attention",
        "pause",
        # OK/Confirm
        "ok",
        "okay",
        "confirm",
        "perfect",
        "all good",
        # Peace/Victory
        "peace",
        "victory",
        "two",
        "v sign",
        # Point/Select
        "point",
        "select",
        "choose",
        "indicate",
        "show",
        # Call/Phone
        "call me",
        "phone",
        "telephone",
        "make a call",
        # Rock/Metal
        "rock on",
        "metal",
        "devil horns",
        "awesome",
        # Fist/Power
        "fist",
        "power",
        "strength",
        "solidarity",
        # Palm/Hand
        "palm",
        "hand",
        "open hand",
        "show hand",
        # One/Single
        "one",
        "single",
        "first",
        "number one",
        # Three/Triple
        "three",
        "triple",
        "third",
        "number three",
        # Four/Quad
        "four",
        "quad",
        "fourth",
        "number four",
        # Mute/Quiet
        "mute",
        "quiet",
        "silence",
        "shh",
        # Volume
        "volume up",
        "increase volume",
        "louder",
        "volume down",
        "decrease volume",
        "quieter",
        # Navigation
        "swipe left",
        "go back",
        "previous",
        "swipe right",
        "go forward",
        "next",
        # No gesture
        "no gesture",
        "nothing",
        "empty hand",
    ]

    def __init__(
        self, checkpoint_path, device="cuda", confidence_threshold=0.3, commands=None
    ):
        self.device = torch.device(device if torch.cuda.is_available() else "cpu")
        self.confidence_threshold = confidence_threshold

        print(f"Using device: {self.device}")

        # Load models
        print("Loading semantic gesture recognition model...")
        self.gesture_encoder = GestureEncoder(embed_dim=512).to(self.device)
        self.text_encoder = TextEncoder(embed_dim=512).to(self.device)

        # Load checkpoint
        checkpoint = torch.load(checkpoint_path, map_location=self.device)
        self.gesture_encoder.load_state_dict(checkpoint["gesture_encoder"])
        self.text_encoder.load_state_dict(checkpoint["text_encoder"])

        self.gesture_encoder.eval()
        self.text_encoder.eval()

        print(
            f"✓ Model loaded (epoch {checkpoint.get('epoch', 'N/A')+1 if checkpoint.get('epoch') is not None else 'N/A'})"
        )

        # Setup commands
        self.commands = commands if commands else self.DEFAULT_COMMANDS
        print(f"✓ Using {len(self.commands)} gesture commands")

        # Precompute text embeddings (for speed)
        print("Precomputing text embeddings...")
        with torch.no_grad():
            self.text_embeds = self.text_encoder(self.commands).cpu()
        print("✓ Text embeddings ready")

        # Setup MediaPipe hand detection
        self.mp_hands = mp.solutions.hands
        self.hands = self.mp_hands.Hands(
            static_image_mode=False,
            max_num_hands=1,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5,
        )
        self.mp_draw = mp.solutions.drawing_utils

        # Image transforms
        self.transform = self._create_transform()

        # Performance tracking
        self.fps_history = []
        self.last_time = time.time()

    def _create_transform(self):
        """Create image preprocessing transform"""
        from torchvision import transforms

        return transforms.Compose(
            [
                transforms.Resize((224, 224)),
                transforms.ToTensor(),
                transforms.Normalize(
                    mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
                ),
            ]
        )

    def extract_hand_region(self, frame, hand_landmarks):
        """Extract hand region from frame"""
        h, w, _ = frame.shape

        # Get bounding box
        x_coords = [lm.x for lm in hand_landmarks.landmark]
        y_coords = [lm.y for lm in hand_landmarks.landmark]

        x_min, x_max = min(x_coords), max(x_coords)
        y_min, y_max = min(y_coords), max(y_coords)

        # Add padding
        padding = 0.2
        width = x_max - x_min
        height = y_max - y_min

        x_min = max(0, x_min - padding * width)
        x_max = min(1, x_max + padding * width)
        y_min = max(0, y_min - padding * height)
        y_max = min(1, y_max + padding * height)

        # Convert to pixel coordinates
        x1, y1 = int(x_min * w), int(y_min * h)
        x2, y2 = int(x_max * w), int(y_max * h)

        # Extract region
        hand_region = frame[y1:y2, x1:x2]

        return hand_region, (x1, y1, x2, y2)

    def recognize_gesture(self, hand_image):
        """Recognize gesture using semantic similarity"""
        if hand_image is None or hand_image.size == 0:
            return None, 0.0, []

        # Preprocess image
        pil_image = Image.fromarray(cv2.cvtColor(hand_image, cv2.COLOR_BGR2RGB))
        img_tensor = self.transform(pil_image).unsqueeze(0).to(self.device)

        # Get visual embedding
        with torch.no_grad():
            visual_embed = self.gesture_encoder(img_tensor).cpu()

        # Compute similarities with all commands
        similarities = torch.matmul(visual_embed, self.text_embeds.T).squeeze()

        # Get top-5 predictions
        top5_scores, top5_indices = similarities.topk(5)

        top5_predictions = [
            (self.commands[idx], score.item())
            for idx, score in zip(top5_indices, top5_scores)
        ]

        # Best prediction
        best_command = self.commands[top5_indices[0]]
        best_score = top5_scores[0].item()

        return best_command, best_score, top5_predictions

    def update_fps(self):
        """Update FPS calculation"""
        current_time = time.time()
        fps = 1.0 / (current_time - self.last_time)
        self.last_time = current_time

        self.fps_history.append(fps)
        if len(self.fps_history) > 30:
            self.fps_history.pop(0)

        return np.mean(self.fps_history)

    def draw_results(self, frame, hand_bbox, gesture, confidence, top5, fps):
        """Draw recognition results on frame"""
        h, w, _ = frame.shape

        # Draw hand bounding box
        if hand_bbox:
            x1, y1, x2, y2 = hand_bbox
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)

        # Draw semi-transparent panel
        panel_height = 200
        overlay = frame.copy()
        cv2.rectangle(overlay, (0, 0), (w, panel_height), (0, 0, 0), -1)
        cv2.addWeighted(overlay, 0.7, frame, 0.3, 0, frame)

        # Draw main prediction
        if gesture:
            # Confidence color (green to red)
            if confidence >= 0.7:
                color = (0, 255, 0)  # Green
            elif confidence >= 0.5:
                color = (0, 255, 255)  # Yellow
            elif confidence >= 0.3:
                color = (0, 165, 255)  # Orange
            else:
                color = (0, 0, 255)  # Red

            # Main gesture text
            cv2.putText(
                frame,
                f"Gesture: {gesture}",
                (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.8,
                color,
                2,
            )

            # Confidence bar
            bar_width = int(confidence * 300)
            cv2.rectangle(frame, (10, 45), (310, 65), (50, 50, 50), -1)
            cv2.rectangle(frame, (10, 45), (10 + bar_width, 65), color, -1)
            cv2.putText(
                frame,
                f"{confidence:.1%}",
                (320, 62),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                (255, 255, 255),
                1,
            )

            # Top-5 predictions
            y_offset = 85
            cv2.putText(
                frame,
                "Top-5 Matches:",
                (10, y_offset),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                (200, 200, 200),
                1,
            )

            for i, (cmd, score) in enumerate(top5, 1):
                y_offset += 20
                # Truncate long commands
                display_cmd = cmd if len(cmd) <= 25 else cmd[:22] + "..."
                text = f"{i}. {display_cmd}: {score:.3f}"
                cv2.putText(
                    frame,
                    text,
                    (15, y_offset),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.4,
                    (180, 180, 180),
                    1,
                )
        else:
            cv2.putText(
                frame,
                "No hand detected",
                (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.8,
                (0, 0, 255),
                2,
            )

        # FPS counter
        cv2.putText(
            frame,
            f"FPS: {fps:.1f}",
            (w - 120, 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (255, 255, 0),
            2,
        )

        # Instructions
        cv2.putText(
            frame,
            "Press Q or ESC to quit",
            (w - 250, h - 10),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (200, 200, 200),
            1,
        )

        return frame

    def run(self, camera_id=0, show_landmarks=True):
        """Run real-time gesture recognition demo"""
        print("\n" + "=" * 70)
        print("SEMANTIC GESTURE RECOGNITION DEMO")
        print("=" * 70)
        print(f"Camera: {camera_id}")
        print(f"Confidence threshold: {self.confidence_threshold}")
        print(f"Available commands: {len(self.commands)}")
        print(f"Device: {self.device}")
        print("\nPress Q or ESC to quit")
        print("=" * 70 + "\n")

        cap = cv2.VideoCapture(camera_id)

        if not cap.isOpened():
            print(f"Error: Could not open camera {camera_id}")
            return

        # Set camera properties
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
        cap.set(cv2.CAP_PROP_FPS, 30)

        print("✓ Camera initialized. Starting recognition...")

        try:
            while True:
                ret, frame = cap.read()
                if not ret:
                    print("Error: Failed to grab frame")
                    break

                # Flip frame horizontally for mirror effect
                frame = cv2.flip(frame, 1)

                # Convert to RGB for MediaPipe
                rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

                # Detect hands
                results = self.hands.process(rgb_frame)

                gesture = None
                confidence = 0.0
                top5 = []
                hand_bbox = None

                if results.multi_hand_landmarks:
                    # Get first hand
                    hand_landmarks = results.multi_hand_landmarks[0]

                    # Draw landmarks
                    if show_landmarks:
                        self.mp_draw.draw_landmarks(
                            frame, hand_landmarks, self.mp_hands.HAND_CONNECTIONS
                        )

                    # Extract hand region and recognize
                    try:
                        hand_region, hand_bbox = self.extract_hand_region(
                            frame, hand_landmarks
                        )
                        gesture, confidence, top5 = self.recognize_gesture(hand_region)

                        # Filter by confidence threshold
                        if confidence < self.confidence_threshold:
                            gesture = None
                            confidence = 0.0
                            top5 = []
                    except Exception as e:
                        print(f"Recognition error: {e}")

                # Update FPS
                fps = self.update_fps()

                # Draw results
                frame = self.draw_results(
                    frame, hand_bbox, gesture, confidence, top5, fps
                )

                # Display
                cv2.imshow("Semantic Gesture Recognition", frame)

                # Check for quit
                key = cv2.waitKey(1) & 0xFF
                if key == ord("q") or key == 27:  # Q or ESC
                    break

        except KeyboardInterrupt:
            print("\n\nInterrupted by user")

        finally:
            cap.release()
            cv2.destroyAllWindows()
            self.hands.close()
            print("\n✓ Demo finished")


def main():
    parser = argparse.ArgumentParser(
        description="Semantic Gesture Recognition Demo",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Basic usage
  python demo_semantic_gesture.py
  
  # Specify checkpoint
  python demo_semantic_gesture.py --checkpoint path/to/semantic_model.pth
  
  # Custom confidence threshold
  python demo_semantic_gesture.py --confidence 0.5
  
  # Use different camera
  python demo_semantic_gesture.py --camera 1
  
  # Custom commands
  python demo_semantic_gesture.py --commands "thumbs up" "thumbs down" "peace"
        """,
    )

    parser.add_argument(
        "--checkpoint",
        type=str,
        default="notebooks/checkpoints/semantic_model.pth",
        help="Path to semantic model checkpoint",
    )
    parser.add_argument(
        "--camera", type=int, default=0, help="Camera device ID (default: 0)"
    )
    parser.add_argument(
        "--confidence",
        type=float,
        default=0.3,
        help="Confidence threshold (0.0-1.0, default: 0.3)",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda",
        choices=["cuda", "cpu"],
        help="Device to use (default: cuda)",
    )
    parser.add_argument(
        "--no-landmarks", action="store_true", help="Hide hand landmarks overlay"
    )
    parser.add_argument(
        "--commands", nargs="+", help="Custom gesture commands (space-separated)"
    )

    args = parser.parse_args()

    # Check if checkpoint exists
    checkpoint_path = Path(args.checkpoint)
    if not checkpoint_path.exists():
        print(f"Error: Checkpoint not found: {checkpoint_path}")
        print("\nTo download the model, run:")
        print("  python scripts/download_semantic_model.py")
        sys.exit(1)

    # Create recognizer
    recognizer = SemanticGestureRecognizer(
        checkpoint_path=str(checkpoint_path),
        device=args.device,
        confidence_threshold=args.confidence,
        commands=args.commands,
    )

    # Run demo
    recognizer.run(camera_id=args.camera, show_landmarks=not args.no_landmarks)


if __name__ == "__main__":
    main()
