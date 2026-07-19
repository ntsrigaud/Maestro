#!/usr/bin/env python3
"""
Downloads the production inference model from Hugging Face Hub so that it can be
bundled into the PyInstaller application for offline execution.
"""

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

    print(f"Downloading {filename} from {repo_id} (revision: {revision})...")

    local_path = hf_hub_download(
        repo_id=repo_id,
        filename=filename,
        revision=revision,
        local_dir=models_dir,  # Download directly to models/ instead of cache
    )

    print(f"Model downloaded successfully to {local_path}")


if __name__ == "__main__":
    main()
