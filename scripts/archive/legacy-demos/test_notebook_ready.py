#!/usr/bin/env python3
"""
Test that the training notebook will correctly detect and use 24 classes.
This simulates what the notebook will do when you run it.
"""

from pathlib import Path
from torchvision.datasets import ImageFolder

print("\n" + "="*70)
print("TRAINING NOTEBOOK PRE-FLIGHT CHECK")
print("="*70 + "\n")

# Get project root dynamically
project_root = Path(__file__).parent.parent
DATASET_ROOT = project_root / 'notebooks/data/hagrid_subset'

print(f"Loading dataset from {DATASET_ROOT}...")

# Load each split (just like the notebook does)
train_path = DATASET_ROOT / 'train'
val_path = DATASET_ROOT / 'val'
test_path = DATASET_ROOT / 'test'

# Create temporary ImageFolder to get class names (exactly as notebook does)
temp_dataset = ImageFolder(train_path)
CLASS_NAMES = temp_dataset.classes
CLASS_TO_IDX = temp_dataset.class_to_idx
NUM_CLASSES = len(CLASS_NAMES)

print(f"\n✓ Dataset structure:")
print(f"  Train: {train_path}")
print(f"  Validation: {val_path}")
print(f"  Test: {test_path}")

print(f"\n✓ Classes ({NUM_CLASSES}):")
for cls_name in sorted(CLASS_NAMES):
    idx = CLASS_TO_IDX[cls_name]
    # Highlight the new class
    marker = " ← NEW!" if cls_name == 'no_gesture2' else ""
    print(f"  {cls_name:<20} (index: {idx:>2}){marker}")

# Verify no_gesture2 is present
assert 'no_gesture2' in CLASS_NAMES, "❌ ERROR: no_gesture2 not found in class names!"
assert NUM_CLASSES == 24, f"❌ ERROR: Expected 24 classes, got {NUM_CLASSES}"

print(f"\n✓ Assertions passed:")
print(f"  - no_gesture2 is present in class list")
print(f"  - Total classes = 24")

# Count samples
print(f"\n✓ Sample counts:")
for split_name, split_path in [('Train', train_path), ('Val', val_path), ('Test', test_path)]:
    ds = ImageFolder(split_path)
    ng2_count = sum(1 for _, lbl in ds.samples if ds.classes[lbl] == 'no_gesture2')
    print(f"  {split_name:5s}: {len(ds):>6,} total samples, {ng2_count:>5,} no_gesture2 samples")

print("\n" + "="*70)
print("✅ ALL CHECKS PASSED - Notebook ready for training with 24 classes!")
print("="*70 + "\n")

print("Next steps:")
print("  1. Open: notebooks/finetune_vit_hagrid.ipynb")
print("  2. Run all cells sequentially")
print("  3. Training will automatically use 24 classes (including no_gesture2)")
print("  4. Expect ~3% longer training time due to additional samples\n")
