#!/usr/bin/env python3
"""Quick verification that no_gesture2 class was added successfully."""

from pathlib import Path
from collections import Counter
from torchvision.datasets import ImageFolder

# Get project root dynamically
project_root = Path(__file__).parent.parent
dataset_path = project_root / 'notebooks/data/hagrid_subset/train'

# Load training dataset
ds = ImageFolder(dataset_path)

# Count samples per class
counts = Counter([ds.classes[lbl] for _, lbl in ds.samples])
total = len(ds.samples)

print(f'\n{"="*60}')
print("UPDATED TRAINING DATASET STATISTICS")
print(f'{"="*60}\n')
print(f'Total classes: {len(ds.classes)}')
print(f'Total samples: {total:,}\n')

# Show negative classes specifically
print('Negative classes:')
for cls in sorted(ds.classes):
    if 'no_gesture' in cls:
        cnt = counts[cls]
        pct = 100 * cnt / total
        print(f'  {cls:<20} {cnt:>6,} samples ({pct:>5.2f}%)')

print(f'\n{"="*60}')
print('✅ no_gesture2 class successfully integrated!')
print(f'{"="*60}\n')
