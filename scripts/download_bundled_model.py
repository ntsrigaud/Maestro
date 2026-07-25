#!/usr/bin/env python3
"""
Downloads the production inference model from Hugging Face Hub, and the
MediaPipe hand landmarker model, so that they can be bundled into the
PyInstaller application for offline execution.
"""

import urllib.request
from pathlib import Path

import yaml
from huggingface_hub import hf_hub_download


def main():
    root_dir = Path(__file__).resolve().parent.parent
    config_path = root_dir / "configs" / "model_source.yaml"

    with open(config_path, encoding="utf-8") as f:
        config = yaml.safe_load(f)

    repo_id = config["repo_id"]
    filename = config["filename"]
    revision = config.get("revision", "main")

    models_dir = root_dir / "models"
    models_dir.mkdir(exist_ok=True)

    # 1. Download the PyTorch sequence model from HuggingFace
    print(f"Downloading {filename} from {repo_id} (revision: {revision})...")
    local_path = hf_hub_download(
        repo_id=repo_id,
        filename=filename,
        revision=revision,
        local_dir=models_dir,
    )
    print(f"PyTorch model downloaded successfully to {local_path}")

    # 2. Download the MediaPipe Hand Landmarker model
    mp_dir = models_dir / "mediapipe"
    mp_dir.mkdir(exist_ok=True)
    mp_filename = "hand_landmarker.task"
    mp_local_path = mp_dir / mp_filename
    mp_url = "https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/1/hand_landmarker.task"
    
    print(f"Downloading {mp_filename} from MediaPipe...")
    urllib.request.urlretrieve(mp_url, mp_local_path)
    print(f"MediaPipe model downloaded successfully to {mp_local_path}")


if __name__ == "__main__":
    main()
