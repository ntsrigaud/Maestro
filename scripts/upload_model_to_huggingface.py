#!/usr/bin/env python3
"""
Upload trained ViT gesture recognition model to Hugging Face Hub.

This script:
1. Loads the best trained model and metadata
2. Packages it with proper configuration
3. Uploads to Hugging Face Hub with documentation

Usage:
    python upload_model_to_huggingface.py --repo-id <username/repo-name> [--token <hf-token>]

Requirements:
    pip install huggingface_hub
"""

import argparse
import json
import sys
from pathlib import Path

import torch
import timm
from huggingface_hub import HfApi, create_repo, upload_file

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.append(str(project_root))


def load_model_and_metadata(checkpoint_dir: Path):
    """Load trained model and its metadata."""
    checkpoint_path = checkpoint_dir / "best_model.pth"
    metadata_path = checkpoint_dir / "model_metadata.json"

    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Model checkpoint not found: {checkpoint_path}")

    if not metadata_path.exists():
        raise FileNotFoundError(f"Metadata not found: {metadata_path}")

    # Load metadata
    with open(metadata_path, "r") as f:
        metadata = json.load(f)

    print(f"✓ Loaded model metadata:")
    print(f"  Version: {metadata['version']}")
    print(f"  Accuracy: {metadata['best_val_acc']:.4f}")
    print(f"  F1 Score: {metadata['best_val_f1']:.4f}")
    print(f"  Classes: {metadata['num_classes']}")

    return checkpoint_path, metadata


def create_model_card(metadata: dict) -> str:
    """Generate model card in Markdown format."""

    class_list = "\n".join([f"- {cls}" for cls in metadata["class_names"]])

    # Extract preprocessing and architecture details
    preprocessing = metadata.get("preprocessing", {})
    architecture = metadata.get("architecture", {})
    training_config = metadata.get("training_config", {})

    model_card = f"""---
language: en
license: mit
tags:
- vision
- image-classification
- gesture-recognition
- hand-gesture
- vision-transformer
- vit
- hagrid
datasets:
- HaGRID
metrics:
- accuracy
- f1
library_name: timm
---

# Hand Gesture Recognition - ViT-B/16 (Clean Baseline)

## Model Description

This Vision Transformer (ViT-B/16) model is fine-tuned on the HaGRID dataset for real-time hand gesture recognition. This is a **clean baseline implementation** with no experimental augmentation or architectural modifications.

**Model Version:** {metadata.get('version', '1.0.0')}  
**Architecture:** ViT-B/16 (224×224 input)  
**Parameters:** ~86M  
**Framework:** PyTorch + timm

## Performance Metrics

| Metric | Value |
|--------|-------|
| Test Accuracy | {metadata.get('test_accuracy', 0.0):.4f} ({metadata.get('test_accuracy', 0.0)*100:.2f}%) |
| Validation Accuracy | {metadata.get('best_val_acc', 0.0):.4f} ({metadata.get('best_val_acc', 0.0)*100:.2f}%) |
| Validation F1 Score | {metadata.get('best_val_f1', 0.0):.4f} ({metadata.get('best_val_f1', 0.0)*100:.2f}%) |
| Training Accuracy | {metadata.get('train_acc', 0.0):.4f} ({metadata.get('train_acc', 0.0)*100:.2f}%) |
| Confusion Matrix Diagonality | {metadata.get('confusion_matrix_diagonality', 0.0):.4f} |

## Training Details

### Dataset
- **Name:** {metadata['dataset']['name']}
- **Source:** [{metadata['dataset']['source']}]({metadata['dataset']['source']})
- **Train samples:** {metadata['dataset']['train_size']:,}
- **Validation samples:** {metadata['dataset']['val_size']:,}
- **Test samples:** {metadata['dataset']['test_size']:,}

### Training Configuration
- **Batch Size:** {training_config.get('batch_size', 32)}
- **Learning Rate:** {training_config.get('learning_rate', 1e-4)}
- **Weight Decay:** {training_config.get('weight_decay', 0.01)}
- **Optimizer:** AdamW
- **Scheduler:** CosineAnnealingLR
- **Epochs Trained:** {training_config.get('epochs_trained', 0)}/{training_config.get('total_epochs', 0)}
- **Label Smoothing:** {training_config.get('label_smoothing', 0.0)} (disabled)

### Data Processing
- **Preprocessing:** Single deterministic pipeline (no augmentation)
- **Resize:** {preprocessing.get('resize', 224)}×{preprocessing.get('resize', 224)}
- **Normalization:** ImageNet statistics (mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
- **Augmentation:** {preprocessing.get('augmentation', 'none')} (clean baseline principle)
- **Test-Time Augmentation:** {preprocessing.get('tta', False)} (disabled)

### Architecture Details
- **Type:** {architecture.get('type', 'standard')} ViT-B/16 (no modifications)
- **Classification Head:** {architecture.get('head', 'single_linear')} (768 → {metadata['num_classes']} classes)
- **Hidden Dimensions:** {architecture.get('hidden_dim', 'N/A')}
- **Dropout:** {architecture.get('dropout', 0.0)}

## Gesture Classes ({metadata['num_classes']} total)

{class_list}

## Usage

### Installation
```bash
pip install torch timm pillow opencv-python
```

### Quick Start
```python
import torch
import timm
from PIL import Image
from torchvision import transforms

# Load model
model = timm.create_model('vit_base_patch16_224', pretrained=False, num_classes={metadata['num_classes']})
checkpoint = torch.load('best_model.pth', map_location='cpu')

# Handle both full checkpoint and state_dict formats
if 'model_state_dict' in checkpoint:
    model.load_state_dict(checkpoint['model_state_dict'])
else:
    model.load_state_dict(checkpoint)

model.eval()

# Prepare transform (MUST match training exactly)
transform = transforms.Compose([
    transforms.Resize(({metadata['img_size']}, {metadata['img_size']})),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

# Inference
image = Image.open('hand_gesture.jpg').convert('RGB')
input_tensor = transform(image).unsqueeze(0)

with torch.no_grad():
    output = model(input_tensor)
    probabilities = torch.softmax(output, dim=1)
    predicted_class = output.argmax(dim=1).item()
    confidence = probabilities[0, predicted_class].item()

class_names = {metadata['class_names']}
print(f"Predicted: {{class_names[predicted_class]}} ({{confidence:.2%}})")
```

### Download from Hugging Face
```python
from huggingface_hub import hf_hub_download

model_path = hf_hub_download(
    repo_id="YOUR_USERNAME/YOUR_REPO",
    filename="best_model.pth"
)
```

## Model Architecture

Based on Vision Transformer (ViT-B/16) with standard architecture:
- **Input:** 224×224 RGB image
- **Patch Size:** 16×16 (196 patches)
- **Embedding Dim:** 768
- **Layers:** 12 transformer blocks
- **Attention Heads:** 12 per layer
- **MLP Ratio:** 4
- **Classification Head:** Single linear layer (768 → {metadata['num_classes']})

**Note:** This is a clean baseline with no architectural modifications, custom heads, or ensemble methods.

## Limitations

- **Single preprocessing pipeline:** No augmentation during training or inference
- **Standard architecture:** No custom attention mechanisms or ensemble methods
- **Dataset bias:** Trained on HaGRID which primarily contains Western gestures
- **Performance constraints:**
  - Requires good lighting conditions
  - Sensitive to heavy occlusion (>40% of hand)
  - Best results with clear, centered hand poses
  - Minimum resolution: ~100px for reliable detection
- **Hardware:** Real-time performance requires GPU (15-30 FPS on modern GPUs)

## Clean Baseline Principles

This model adheres to strict baseline principles:
1. **No data augmentation** - Single deterministic transform pipeline
2. **No test-time augmentation** - Direct inference only
3. **Standard architecture** - No custom heads or modifications
4. **Reproducible** - Fixed random seeds and deterministic operations
5. **Minimal complexity** - Simplest possible training configuration

These principles ensure:
- Clear cause-and-effect relationships
- Reproducible results
- Fair comparison baselines
- Academic rigor

## Reproducibility

All hyperparameters and configurations are documented in the training notebook:
- `notebooks/finetune_vit_hagrid.ipynb`

To reproduce:
```bash
# 1. Download dataset
python scripts/download_hagrid_kaggle.py

# 2. Train model
jupyter notebook notebooks/finetune_vit_hagrid.ipynb

# 3. Upload to HF
python scripts/upload_model_to_huggingface.py --repo-id username/repo-name
```

## Citation

```bibtex
@misc{{hagrid-vit-clean-baseline-2025,
  title={{Hand Gesture Recognition with Vision Transformer (Clean Baseline)}},
  author={{CSU Capstone Project}},
  year={{2025}},
  publisher={{Hugging Face}},
  howpublished={{\\url{{https://huggingface.co/neilrigaud/hagrid-vit-gesture}}}}
}}
```

## License

MIT License

## Acknowledgments

- **Dataset:** HaGRID (Hand Gesture Recognition Image Dataset)
- **Base Model:** ViT-B/16 pretrained on ImageNet-21k (timm library)
- **Framework:** PyTorch, timm, torchvision
"""

    return model_card


def upload_to_huggingface(
    checkpoint_path: Path,
    metadata: dict,
    repo_id: str,
    token: str = None,
    private: bool = False,
):
    """Upload model to Hugging Face Hub."""

    print(f"\n{'='*70}")
    print(f"UPLOADING MODEL TO HUGGING FACE")
    print(f"{'='*70}\n")

    api = HfApi()

    # Create repository
    print(f"[1/4] Creating repository: {repo_id}")
    try:
        create_repo(
            repo_id=repo_id,
            token=token,
            private=private,
            repo_type="model",
            exist_ok=True,
        )
        print(f"✓ Repository created/exists: {repo_id}")
    except Exception as e:
        print(f"✗ Failed to create repository: {e}")
        return False

    # Upload model weights
    print(f"\n[2/4] Uploading model weights...")
    try:
        api.upload_file(
            path_or_fileobj=str(checkpoint_path),
            path_in_repo="best_model.pth",
            repo_id=repo_id,
            token=token,
        )
        print(f"✓ Model weights uploaded")
    except Exception as e:
        print(f"✗ Failed to upload model: {e}")
        return False

    # Upload metadata
    print(f"\n[3/4] Uploading metadata...")
    metadata_path = checkpoint_path.parent / "model_metadata.json"
    try:
        api.upload_file(
            path_or_fileobj=str(metadata_path),
            path_in_repo="model_metadata.json",
            repo_id=repo_id,
            token=token,
        )
        print(f"✓ Metadata uploaded")
    except Exception as e:
        print(f"✗ Failed to upload metadata: {e}")
        return False

    # Create and upload model card
    print(f"\n[4/4] Creating model card...")
    model_card = create_model_card(metadata)
    model_card_path = checkpoint_path.parent / "README.md"

    with open(model_card_path, "w") as f:
        f.write(model_card)

    try:
        api.upload_file(
            path_or_fileobj=str(model_card_path),
            path_in_repo="README.md",
            repo_id=repo_id,
            token=token,
        )
        print(f"✓ Model card uploaded")
    except Exception as e:
        print(f"✗ Failed to upload model card: {e}")
        return False

    print(f"\n{'='*70}")
    print(f"SUCCESS")
    print(f"{'='*70}")
    print(f"\nModel uploaded to: https://huggingface.co/{repo_id}")
    print(f"\nTo use this model:")
    print(f"  from huggingface_hub import hf_hub_download")
    print(
        f"  model_path = hf_hub_download(repo_id='{repo_id}', filename='best_model.pth')"
    )
    print(f"{'='*70}\n")

    return True


def main():
    parser = argparse.ArgumentParser(
        description="Upload trained gesture recognition model to Hugging Face Hub"
    )
    parser.add_argument(
        "--repo-id",
        type=str,
        required=True,
        help="Hugging Face repository ID (format: username/repo-name)",
    )
    parser.add_argument(
        "--token",
        type=str,
        default=None,
        help="Hugging Face API token (or set HF_TOKEN environment variable)",
    )
    parser.add_argument(
        "--checkpoint-dir",
        type=str,
        default="notebooks/checkpoints",
        help="Directory containing model checkpoint and metadata",
    )
    parser.add_argument(
        "--private", action="store_true", help="Create private repository"
    )

    args = parser.parse_args()

    # Resolve checkpoint directory
    checkpoint_dir = project_root / args.checkpoint_dir
    if not checkpoint_dir.exists():
        print(f"✗ Checkpoint directory not found: {checkpoint_dir}")
        print(f"  Please train the model first using finetune_vit_hagrid.ipynb")
        sys.exit(1)

    # Load model and metadata
    try:
        checkpoint_path, metadata = load_model_and_metadata(checkpoint_dir)
    except Exception as e:
        print(f"✗ Failed to load model: {e}")
        sys.exit(1)

    # Upload to Hugging Face
    success = upload_to_huggingface(
        checkpoint_path=checkpoint_path,
        metadata=metadata,
        repo_id=args.repo_id,
        token=args.token,
        private=args.private,
    )

    if not success:
        sys.exit(1)


if __name__ == "__main__":
    main()
