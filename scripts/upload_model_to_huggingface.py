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
    checkpoint_path = checkpoint_dir / 'best_model.pth'
    metadata_path = checkpoint_dir / 'model_metadata.json'
    
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Model checkpoint not found: {checkpoint_path}")
    
    if not metadata_path.exists():
        raise FileNotFoundError(f"Metadata not found: {metadata_path}")
    
    # Load metadata
    with open(metadata_path, 'r') as f:
        metadata = json.load(f)
    
    print(f"✓ Loaded model metadata:")
    print(f"  Version: {metadata['version']}")
    print(f"  Accuracy: {metadata['best_val_acc']:.4f}")
    print(f"  F1 Score: {metadata['best_val_f1']:.4f}")
    print(f"  Classes: {metadata['num_classes']}")
    
    return checkpoint_path, metadata


def create_model_card(metadata: dict) -> str:
    """Generate model card in Markdown format."""
    
    class_list = '\n'.join([f"- {cls}" for cls in metadata['class_names']])
    
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

# Hand Gesture Recognition - ViT-B/16 (HaGRID Fine-tuned)

## Model Description

This Vision Transformer (ViT-B/16) model is fine-tuned on the HaGRID dataset for real-time hand gesture recognition. It achieves **{metadata['best_val_acc']*100:.2f}% accuracy** on 18 gesture classes.

**Model Version:** {metadata['version']}  
**Architecture:** ViT-B/16 (224×224 input)  
**Parameters:** ~86M  
**Framework:** PyTorch + timm

## Performance Metrics

| Metric | Value |
|--------|-------|
| Validation Accuracy | {metadata['best_val_acc']:.4f} ({metadata['best_val_acc']*100:.2f}%) |
| Validation F1 Score | {metadata['best_val_f1']:.4f} ({metadata['best_val_f1']*100:.2f}%) |
| Training Accuracy | {metadata['train_acc']:.4f} ({metadata['train_acc']*100:.2f}%) |

## Training Details

### Dataset
- **Name:** {metadata['dataset']['name']}
- **Source:** [{metadata['dataset']['source']}]({metadata['dataset']['source']})
- **Train samples:** {metadata['dataset']['train_size']}
- **Validation samples:** {metadata['dataset']['val_size']}
- **Test samples:** {metadata['dataset']['test_size']}

### Training Configuration
- **Batch Size:** {metadata['training_config']['batch_size']}
- **Learning Rate:** {metadata['training_config']['learning_rate']}
- **Weight Decay:** {metadata['training_config']['weight_decay']}
- **Label Smoothing:** {metadata['training_config']['label_smoothing']}
- **Epochs Trained:** {metadata['training_config']['epochs_trained']}/{metadata['training_config']['total_epochs']}

### Augmentation Strategy
Enhanced augmentation for improved robustness:
- Geometric: RandomRotation, RandomAffine, RandomPerspective
- Photometric: ColorJitter with hue adjustment
- Blur: GaussianBlur for motion simulation
- Occlusion: RandomErasing for partial hand occlusion
- Test-Time Augmentation (TTA) for inference boost

## Gesture Classes

{class_list}

## Usage

### Installation
```bash
pip install torch timm pillow
```

### Inference
```python
import torch
import timm
from PIL import Image
from torchvision import transforms

# Load model
model = timm.create_model('vit_base_patch16_224', pretrained=False, num_classes={metadata['num_classes']})
model.load_state_dict(torch.load('best_model.pth'))
model.eval()

# Prepare image
transform = transforms.Compose([
    transforms.Resize(({metadata['img_size']}, {metadata['img_size']})),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

image = Image.open('hand_gesture.jpg').convert('RGB')
input_tensor = transform(image).unsqueeze(0)

# Predict
with torch.no_grad():
    output = model(input_tensor)
    probabilities = torch.softmax(output, dim=1)
    predicted_class = output.argmax(dim=1).item()
    confidence = probabilities[0, predicted_class].item()

class_names = {metadata['class_names']}
print(f"Predicted: {{class_names[predicted_class]}} ({{confidence:.2%}})")
```

### Test-Time Augmentation (Recommended)
For improved accuracy, use TTA during inference:
```python
tta_transforms = [
    transform,  # Original
    transforms.Compose([  # Horizontal flip
        transforms.Resize(({metadata['img_size']}, {metadata['img_size']})),
        transforms.RandomHorizontalFlip(p=1.0),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
]

outputs = []
for tta_t in tta_transforms:
    input_tensor = tta_t(image).unsqueeze(0)
    with torch.no_grad():
        outputs.append(model(input_tensor))

output_avg = torch.stack(outputs).mean(dim=0)
predicted_class = output_avg.argmax(dim=1).item()
```

## Model Architecture

Based on Vision Transformer (ViT-B/16):
- **Input:** 224×224 RGB image
- **Patch Size:** 16×16
- **Embedding Dim:** 768
- **Layers:** 12 transformer blocks
- **Attention Heads:** 12
- **MLP Ratio:** 4
- **Classification Head:** Linear layer to {metadata['num_classes']} classes

## Limitations and Biases

- Trained on HaGRID dataset which primarily contains Western gestures
- Performance may degrade with:
  - Extreme lighting conditions
  - Heavy occlusion (>40% of hand)
  - Non-standard hand poses
  - Very blurry or low-resolution images (<100px)
- Model assumes single hand in frame
- Real-time performance requires GPU (15-30 FPS on modern GPUs)

## Training Improvements

This model incorporates several improvements over baseline ViT fine-tuning:
1. **Label smoothing** (0.1) for better generalization
2. **Enhanced augmentation** for robustness
3. **Gradient clipping** for stable training
4. **Warmup + Cosine annealing** learning rate schedule
5. **Early stopping** to prevent overfitting
6. **Mixed precision training** for efficiency
7. **Test-time augmentation** for improved accuracy

## Citation

```bibtex
@misc{{hagrid-vit-gesture-2025,
  title={{Hand Gesture Recognition with Vision Transformer}},
  author={{CSU Capstone Project}},
  year={{2025}},
  publisher={{Hugging Face}},
  howpublished={{\\url{{https://huggingface.co/YOUR_USERNAME/YOUR_REPO}}}}
}}
```

## License

MIT License - See LICENSE file for details.

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
    private: bool = False
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
            exist_ok=True
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
    metadata_path = checkpoint_path.parent / 'model_metadata.json'
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
    model_card_path = checkpoint_path.parent / 'README.md'
    
    with open(model_card_path, 'w') as f:
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
    print(f"  model_path = hf_hub_download(repo_id='{repo_id}', filename='best_model.pth')")
    print(f"{'='*70}\n")
    
    return True


def main():
    parser = argparse.ArgumentParser(
        description='Upload trained gesture recognition model to Hugging Face Hub'
    )
    parser.add_argument(
        '--repo-id',
        type=str,
        required=True,
        help='Hugging Face repository ID (format: username/repo-name)'
    )
    parser.add_argument(
        '--token',
        type=str,
        default=None,
        help='Hugging Face API token (or set HF_TOKEN environment variable)'
    )
    parser.add_argument(
        '--checkpoint-dir',
        type=str,
        default='notebooks/checkpoints',
        help='Directory containing model checkpoint and metadata'
    )
    parser.add_argument(
        '--private',
        action='store_true',
        help='Create private repository'
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
        private=args.private
    )
    
    if not success:
        sys.exit(1)


if __name__ == '__main__':
    main()
