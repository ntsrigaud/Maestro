#!/usr/bin/env python3
"""
Dynamic Gesture Meaning Demo

Tracks the evolution of ViT gesture predictions over time and outputs a dynamic gesture meaning in real time.
No LSTM training required—uses a buffer of recent predictions to infer gesture dynamics.
"""
import sys
import time
from pathlib import Path
from collections import deque, Counter
import numpy as np
import cv2
from PIL import Image
import torch
import timm
from torchvision import transforms
from src.input_manager import InputManager
from src.recognition_engine.preprocessing import center_crop_frame, get_crop_coordinates
from src.shared.logger import get_logger

default_model_dir = Path(__file__).parent.parent / "checkpoints"
default_gesture_names = [
    'call', 'dislike', 'fist', 'four', 'like', 'mute', 'ok', 'one',
    'palm', 'peace', 'peace_inverted', 'rock', 'stop', 'stop_inverted',
    'three', 'three2', 'two_up', 'two_up_inverted'
]
logger = get_logger(__name__)

class DynamicGestureMeaningDemo:
    def __init__(self, checkpoint_path, camera_index=0, buffer_size=16, device=None):
        self.checkpoint_path = Path(checkpoint_path)
        self.camera_index = camera_index
        self.buffer_size = buffer_size
        self.device = torch.device(device) if device else (
            torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu'))
        self.input_mgr = None
        self.model = None
        self.transform = None
        self.gesture_classes = default_gesture_names
        self.pred_buffer = deque(maxlen=buffer_size)
        self.fps_history = deque(maxlen=30)

    def initialize(self):
        logger.info(f"Initializing dynamic gesture meaning demo...")
        self.input_mgr = InputManager(camera_index=self.camera_index)
        self.input_mgr._convert_to_rgb = False
        if not self.input_mgr.start_continuous_capture():
            raise RuntimeError(f"Failed to open camera {self.camera_index}.")
        logger.info("✓ Camera ready")
        checkpoint = torch.load(self.checkpoint_path, map_location=self.device)
        self.model = timm.create_model(
            "vit_base_patch16_224", pretrained=False, num_classes=len(self.gesture_classes)
        )
        if "model_state_dict" in checkpoint:
            self.model.load_state_dict(checkpoint["model_state_dict"])
        else:
            self.model.load_state_dict(checkpoint)
        self.model = self.model.to(self.device)
        self.model.eval()
        self.transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])
        logger.info("✓ Model loaded and ready")

    def classify_gesture(self, frame_rgb):
        pil_img = Image.fromarray(frame_rgb)
        img_tensor = self.transform(pil_img).unsqueeze(0).to(self.device)
        with torch.no_grad():
            output = self.model(img_tensor)
            probs = torch.softmax(output, dim=1)[0]
        pred_idx = output.argmax(dim=1).item()
        confidence = probs[pred_idx].item()
        gesture = self.gesture_classes[pred_idx]
        return gesture, confidence

    def infer_dynamic_meaning(self):
        # Simple rule-based dynamic meaning from buffer
        if not self.pred_buffer:
            return "No gesture detected"
        counts = Counter(self.pred_buffer)
        most_common, freq = counts.most_common(1)[0]
        if freq > len(self.pred_buffer) * 0.7:
            return f"Holding: {most_common}"
        if len(set(self.pred_buffer)) > 1:
            # Detect transitions
            if self.pred_buffer[0] != self.pred_buffer[-1]:
                return f"Transition: {self.pred_buffer[0]} → {self.pred_buffer[-1]}"
            else:
                return f"Changing gesture"
        return f"Gesture: {most_common}"

    def run(self):
        self.initialize()
        logger.info("Starting dynamic gesture meaning demo. Press 'q' to quit.")
        prev_time = time.time()
        while True:
            frame_data = self.input_mgr.get_latest_frame()
            if frame_data is None or frame_data.frame is None:
                time.sleep(0.05)
                continue
            frame = frame_data.frame
            h, w = frame.shape[:2]
            x1, y1, x2, y2 = get_crop_coordinates((h, w))
            preprocessed = center_crop_frame(frame, target_size=224, bgr_to_rgb=True)
            gesture, confidence = self.classify_gesture(preprocessed)
            self.pred_buffer.append(gesture)
            meaning = self.infer_dynamic_meaning()
            # Draw crop box
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            # Draw meaning
            cv2.rectangle(frame, (10, 40), (w - 10, 110), (0, 0, 0), -1)
            cv2.putText(frame, f"Dynamic: {meaning}", (20, 80), cv2.FONT_HERSHEY_DUPLEX, 1.0, (0, 255, 255), 2)
            # FPS
            curr_time = time.time()
            fps = 1.0 / (curr_time - prev_time) if curr_time != prev_time else 0
            self.fps_history.append(fps)
            avg_fps = np.mean(self.fps_history) if self.fps_history else 0
            prev_time = curr_time
            cv2.putText(frame, f"FPS: {avg_fps:.1f}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255,255,255), 2)
            cv2.imshow("Dynamic Gesture Meaning", frame)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break
        self.input_mgr.stop_capture()
        cv2.destroyAllWindows()
        logger.info("Demo stopped.")

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Dynamic Gesture Meaning Demo")
    parser.add_argument("--checkpoint", type=str, default=str(default_model_dir / "best_model.pth"), help="Path to ViT model checkpoint")
    parser.add_argument("--camera", type=int, default=0, help="Camera device index")
    parser.add_argument("--buffer", type=int, default=16, help="Number of frames for dynamic analysis")
    parser.add_argument("--device", type=str, default=None, help="Device: cuda, cpu, etc.")
    args = parser.parse_args()
    demo = DynamicGestureMeaningDemo(args.checkpoint, camera_index=args.camera, buffer_size=args.buffer, device=args.device)
    demo.run()
