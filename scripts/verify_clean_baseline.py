#!/usr/bin/env python3
"""
Clean Baseline Sanity Check

Verifies that the clean baseline training setup is correct before training.
Checks:
1. Dataset path and structure
2. Class names and ordering
3. Sample counts per class
4. Forward pass with sample batch
5. Label-to-index mapping consistency
6. Transform pipeline correctness
"""

import sys
from pathlib import Path

project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

import torch
import timm
from torchvision import transforms
from PIL import Image
import numpy as np


# Expected gesture classes (alphabetically sorted)
EXPECTED_CLASSES = [
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


def check_dataset():
    """Verify dataset structure and classes."""
    print("=" * 70)
    print("DATASET VERIFICATION")
    print("=" * 70)

    dataset_root = Path("./data/hagrid-sample-30k-384p/hagrid_30k")

    if not dataset_root.exists():
        print(f"❌ Dataset not found at: {dataset_root}")
        print(
            f"\nDownload from: https://www.kaggle.com/datasets/innominate817/hagrid-sample-30k-384p"
        )
        return False

    print(f"✓ Dataset root exists: {dataset_root}")

    # Find gesture directories
    gesture_dirs = sorted(
        [d for d in dataset_root.iterdir() if d.is_dir() and not d.name.startswith(".")]
    )

    # Extract class names
    class_names = sorted([d.name.replace("train_val_", "") for d in gesture_dirs])

    print(f"\nFound {len(class_names)} classes:")
    for cls in class_names:
        print(f"  - {cls}")

    # Verify classes match expected
    if class_names != EXPECTED_CLASSES:
        print(f"\n❌ Class mismatch!")
        print(f"Expected: {EXPECTED_CLASSES}")
        print(f"Found: {class_names}")
        return False

    print(f"\n✓ Class names match expected (alphabetically sorted)")

    # Count samples per class
    print(f"\nSample counts:")
    total_samples = 0
    for gesture_dir in gesture_dirs:
        cls_name = gesture_dir.name.replace("train_val_", "")
        jpg_count = len(list(gesture_dir.glob("*.jpg")))
        png_count = len(list(gesture_dir.glob("*.png")))
        count = jpg_count + png_count
        total_samples += count
        print(f"  {cls_name}: {count}")

    print(f"\nTotal samples: {total_samples}")

    if total_samples < 1000:
        print(f"❌ Too few samples: {total_samples}")
        return False

    print(f"✓ Sample count reasonable")

    return True


def check_transform():
    """Verify transform pipeline."""
    print("\n" + "=" * 70)
    print("TRANSFORM VERIFICATION")
    print("=" * 70)

    transform = transforms.Compose(
        [
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
    )

    print("Transform pipeline:")
    print(transform)

    # Test with dummy image
    dummy_img = Image.new("RGB", (384, 384), color="red")
    transformed = transform(dummy_img)

    print(f"\nTest transform:")
    print(f"  Input: PIL Image (384, 384, 3)")
    print(f"  Output: {transformed.shape}")
    print(f"  Expected: torch.Size([3, 224, 224])")

    if transformed.shape != torch.Size([3, 224, 224]):
        print(f"❌ Shape mismatch")
        return False

    print(f"✓ Transform output correct")

    # Check normalization
    print(f"\n  Value range: [{transformed.min():.3f}, {transformed.max():.3f}]")
    print(f"  Expected: approximately [-2.5, 2.5] (ImageNet normalized)")

    return True


def check_model():
    """Verify model architecture."""
    print("\n" + "=" * 70)
    print("MODEL VERIFICATION")
    print("=" * 70)

    NUM_CLASSES = len(EXPECTED_CLASSES)

    # Load ViT-B/16
    print("Loading ViT-B/16...")
    model = timm.create_model(
        "vit_base_patch16_224", pretrained=False, num_classes=NUM_CLASSES
    )

    print(f"✓ Model created")
    print(f"  Architecture: ViT-B/16")
    print(f"  Output classes: {NUM_CLASSES}")
    print(f"  Parameters: {sum(p.numel() for p in model.parameters()) / 1e6:.1f}M")

    # Test forward pass
    print(f"\nTesting forward pass...")
    model.eval()

    batch_size = 4
    dummy_input = torch.randn(batch_size, 3, 224, 224)

    with torch.no_grad():
        output = model(dummy_input)

    print(f"  Input shape: {dummy_input.shape}")
    print(f"  Output shape: {output.shape}")
    print(f"  Expected output: ({batch_size}, {NUM_CLASSES})")

    if output.shape != (batch_size, NUM_CLASSES):
        print(f"❌ Output shape mismatch")
        return False

    print(f"✓ Forward pass successful")

    # Check output is valid logits
    print(f"\n  Logit range: [{output.min():.3f}, {output.max():.3f}]")
    print(f"  Mean: {output.mean():.3f}, Std: {output.std():.3f}")

    # Test softmax
    probs = torch.softmax(output, dim=1)
    print(f"\n  Softmax sum: {probs.sum(dim=1).tolist()} (should be all 1.0)")

    if not torch.allclose(probs.sum(dim=1), torch.ones(batch_size)):
        print(f"❌ Softmax normalization failed")
        return False

    print(f"✓ Softmax normalization correct")

    return True


def check_label_ordering():
    """Verify label-to-index mapping consistency."""
    print("\n" + "=" * 70)
    print("LABEL ORDERING VERIFICATION")
    print("=" * 70)

    class_to_idx = {cls: i for i, cls in enumerate(EXPECTED_CLASSES)}
    idx_to_class = {i: cls for cls, i in class_to_idx.items()}

    print("Class to Index mapping:")
    for cls, idx in sorted(class_to_idx.items()):
        print(f"  {cls}: {idx}")

    # Verify bidirectional consistency
    for cls, idx in class_to_idx.items():
        if idx_to_class[idx] != cls:
            print(f"❌ Mapping inconsistency for {cls}")
            return False

    print(f"\n✓ Label ordering consistent (alphabetical)")

    return True


def main():
    """Run all sanity checks."""
    print("\n" + "=" * 70)
    print("CLEAN BASELINE SANITY CHECK")
    print("=" * 70)
    print("\nVerifying setup before training...\n")

    checks = [
        ("Dataset", check_dataset),
        ("Transform", check_transform),
        ("Model", check_model),
        ("Label Ordering", check_label_ordering),
    ]

    results = []
    for name, check_fn in checks:
        try:
            result = check_fn()
            results.append((name, result))
        except Exception as e:
            print(f"\n❌ {name} check failed with exception: {e}")
            results.append((name, False))

    # Summary
    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)

    all_passed = True
    for name, passed in results:
        status = "✓ PASS" if passed else "❌ FAIL"
        print(f"  {status}: {name}")
        if not passed:
            all_passed = False

    print()
    if all_passed:
        print("✓ All checks passed! Ready to train.")
        print("\nNext steps:")
        print(
            "  1. Open: jupyter notebook notebooks/finetune_vit_hagrid_CLEAN_BASELINE.ipynb"
        )
        print("  2. Run all cells")
        print("  3. Verify test accuracy >85%")
        return 0
    else:
        print("❌ Some checks failed. Fix issues before training.")
        return 1


if __name__ == "__main__":
    sys.exit(main())
