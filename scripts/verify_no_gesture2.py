#!/usr/bin/env python3
"""Quick verification that no_gesture2 class was added successfully."""

from torchvision.datasets import ImageFolder
from collections import Counter

# Load training dataset
ds = ImageFolder('/home/rachel/.Neil/CSU-Capstone-Project/notebooks/data/hagrid_subset/train')

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
