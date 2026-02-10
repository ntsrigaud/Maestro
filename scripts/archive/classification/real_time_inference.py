"""
Real-Time Gesture Recognition with ViT and LSTM (CPU-only, stable)

Pipeline:
1. Webcam frames
2. ViT backbone (768-dim embeddings)
3. LSTM temporal classifier

This version is strictly CPU-only and matches training architecture.
"""

import os

os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
os.environ["TF_ENABLE_METAL"] = "0"
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"

import sys
import time
import argparse
import shutil
from pathlib import Path
from collections import deque

import cv2
import numpy as np
from PIL import Image

import torch
import timm
from torchvision import transforms

import tensorflow as tf

project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.input_manager import InputManager
from src.shared.logger import get_logger

logger = get_logger(__name__)

DEFAULT_VIT_HF_REPO = "neilrigaud/hagrid-vit-gesture"
DEFAULT_LSTM_HF_REPO = "hslay13/vit_lstm_jester"
DEFAULT_MODEL_DIR = project_root / "checkpoints"

FRAME_COUNT = 16
FRAME_SIZE = (224, 224)

COLOR_TEXT = (0, 255, 0)
COLOR_BG = (0, 0, 0)

GESTURE_CLASSES = [
    "Doing other things",
    "Drumming Fingers",
    "No gesture",
    "Pulling Hand In",
    "Pulling Two Fingers In",
    "Pushing Hand Away",
    "Pushing Two Fingers Away",
    "Rolling Hand Backward",
    "Rolling Hand Forward",
    "Shaking Hand",
    "Sliding Two Fingers Down",
    "Sliding Two Fingers Left",
    "Sliding Two Fingers Right",
    "Sliding Two Fingers Up",
    "Stop Sign",
    "Swiping Down",
    "Swiping Left",
    "Swiping Right",
    "Swiping Up",
    "Thumb Down",
    "Thumb Up",
    "Turning Hand Clockwise",
    "Turning Hand Counterclockwise",
    "Zooming In With Full Hand",
    "Zooming In With Two Fingers",
    "Zooming Out With Full Hand",
    "Zooming Out With Two Fingers",
]


def download_model_from_hf(repo_id: str, filename: str, cache_dir: Path) -> Path:
    cache_dir.mkdir(parents=True, exist_ok=True)
    model_path = cache_dir / filename

    if model_path.exists():
        logger.info(f"Model found in cache: {model_path}")
        return model_path

    logger.info(f"Downloading {repo_id}/{filename}")

    from huggingface_hub import hf_hub_download

    downloaded = hf_hub_download(
        repo_id=repo_id,
        filename=filename,
        cache_dir=cache_dir,
    )

    shutil.copy(downloaded, model_path)
    return model_path


class RealTimeInference:
    def __init__(
        self,
        vit_checkpoint: str | None,
        lstm_checkpoint: str | None,
        vit_hf_repo: str,
        lstm_hf_repo: str,
        camera_index: int = 0,
    ):
        # Force CPU for Torch
        self.device = torch.device("cpu")
        torch.set_num_threads(1)

        # Resolve checkpoints
        self.vit_checkpoint = self._resolve(
            vit_checkpoint, vit_hf_repo, "best_model.pth"
        )
        self.lstm_checkpoint = self._resolve(
            lstm_checkpoint, lstm_hf_repo, "best_lstm_model.keras"
        )

        self.camera_index = camera_index
        self.frames = deque(maxlen=FRAME_COUNT)
        self.last_prediction = "Initializing"

    def _resolve(self, path, repo, filename):
        if path:
            p = Path(path)
            if p.exists():
                return p
        return download_model_from_hf(repo, filename, DEFAULT_MODEL_DIR)

    # Initialization
    def initialize(self):
        logger.info("Initializing (CPU-only)...")

        # Camera
        self.input_manager = InputManager(camera_index=self.camera_index)
        if not self.input_manager.start_continuous_capture():
            raise RuntimeError("Camera initialization failed")

        # ViT backbone ONLY (768-dim output)
        self.vit = timm.create_model(
            "vit_base_patch16_224",
            pretrained=False,
            num_classes=0,
        )

        checkpoint = torch.load(self.vit_checkpoint, map_location="cpu")
        state_dict = checkpoint.get("model_state_dict", checkpoint)
        self.vit.load_state_dict(state_dict, strict=False)
        self.vit.eval()

        # TensorFlow CPU only
        tf.config.set_visible_devices([], "GPU")
        self.lstm = tf.keras.models.load_model(self.lstm_checkpoint)

        # Preprocessing
        self.transform = transforms.Compose(
            [
                transforms.Resize(FRAME_SIZE),
                transforms.ToTensor(),
                transforms.Normalize(
                    mean=[0.485, 0.456, 0.406],
                    std=[0.229, 0.224, 0.225],
                ),
            ]
        )

        logger.info("Initialization complete.")

    # Inference helpers
    def extract_embeddings(self, frames):
        images = [Image.fromarray(cv2.cvtColor(f, cv2.COLOR_BGR2RGB)) for f in frames]

        batch = torch.stack([self.transform(img) for img in images])

        with torch.no_grad():
            embeddings = self.vit(batch)

        return embeddings.numpy()  # (16, 768)

    def predict(self, embeddings):
        preds = self.lstm.predict(embeddings[None, ...], verbose=0)
        return GESTURE_CLASSES[int(np.argmax(preds))]

    # Main loop
    def run(self):
        self.initialize()
        logger.info("Running inference. Press 'q' to quit.")

        while True:
            frame_data = self.input_manager.get_latest_frame()
            if frame_data is None or frame_data.frame is None:
                continue

            frame = frame_data.frame
            self.frames.append(frame.copy())

            if len(self.frames) == FRAME_COUNT:
                emb = self.extract_embeddings(list(self.frames))
                self.last_prediction = self.predict(emb)

            cv2.putText(
                frame,
                f"Prediction: {self.last_prediction}",
                (10, 40),
                cv2.FONT_HERSHEY_SIMPLEX,
                1,
                COLOR_TEXT,
                2,
            )

            cv2.imshow("Gesture Recognition", frame)

            if cv2.waitKey(1) & 0xFF == ord("q"):
                break

        self.cleanup()

    def cleanup(self):
        self.input_manager.stop_capture()
        cv2.destroyAllWindows()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--vit-checkpoint", type=str, default=None)
    parser.add_argument("--lstm-checkpoint", type=str, default=None)
    parser.add_argument("--vit-hf-repo", type=str, default=DEFAULT_VIT_HF_REPO)
    parser.add_argument("--lstm-hf-repo", type=str, default=DEFAULT_LSTM_HF_REPO)
    parser.add_argument("--camera", type=int, default=0)

    args = parser.parse_args()

    app = RealTimeInference(
        vit_checkpoint=args.vit_checkpoint,
        lstm_checkpoint=args.lstm_checkpoint,
        vit_hf_repo=args.vit_hf_repo,
        lstm_hf_repo=args.lstm_hf_repo,
        camera_index=args.camera,
    )

    app.run()


if __name__ == "__main__":
    main()
