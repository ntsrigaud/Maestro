#!/usr/bin/env python3
"""
Build no_gesture2 Class from External Images (Simplified)
=========================================================

This script creates a robust negative class by randomly sampling images from MS COCO
or similar datasets. Since we pass raw images directly to ViT (no hand cropping),
we simply need diverse images to serve as negative examples.

Usage:
    python scripts/build_no_gesture2_simple.py \
        --source_dir ./notebooks/data/coco_val2017 \
        --output_dir ./notebooks/data/hagrid_subset \
        --target_count 7000 \
        --train_ratio 0.7 \
        --val_ratio 0.15 \
        --test_ratio 0.15

Author: CSU Capstone Team
Date: 2026-01-23
"""

import argparse
import cv2
import numpy as np
from pathlib import Path
import shutil
import json
from typing import List, Dict, Tuple
from tqdm import tqdm
import random


class QualityFilter:
    """Filter images by resolution and aspect ratio."""
    
    def __init__(self, min_size: int = 160, max_aspect: float = 3.0):
        self.min_size = min_size
        self.max_aspect = max_aspect
    
    def is_valid(self, image_path: Path) -> bool:
        """Check if image meets quality criteria."""
        try:
            img = cv2.imread(str(image_path))
            if img is None:
                return False
            
            h, w = img.shape[:2]
            
            # Check minimum size (lowered to 160 to accept CelebA's 178×218 images)
            # Images will be resized to 224×224 during training anyway
            if min(h, w) < self.min_size:
                return False
            
            # Check aspect ratio
            aspect = max(h, w) / min(h, w)
            if aspect > self.max_aspect:
                return False
            
            return True
        except Exception:
            return False


def sample_valid_images(
    source_dir: Path,
    target_count: int,
    quality_filter: QualityFilter,
    seed: int = 42
) -> List[Path]:
    """
    Randomly sample valid images from source directory.
    
    Args:
        source_dir: Directory containing source images
        target_count: Number of images to collect
        quality_filter: QualityFilter instance
        seed: Random seed for reproducibility
    
    Returns:
        List of valid image paths
    """
    random.seed(seed)
    
    # Get all images
    image_extensions = {'.jpg', '.jpeg', '.png'}
    all_images = [
        p for p in source_dir.rglob('*')
        if p.suffix.lower() in image_extensions and p.is_file()
    ]
    
    print(f"Found {len(all_images)} total images in source directory")
    random.shuffle(all_images)
    
    valid_images = []
    
    print(f"Sampling valid images (target: {target_count})...")
    
    for img_path in tqdm(all_images, desc="Filtering by quality"):
        if len(valid_images) >= target_count:
            break
        
        # Check quality criteria
        if quality_filter.is_valid(img_path):
            valid_images.append(img_path)
    
    return valid_images


def distribute_to_splits(
    image_paths: List[Path],
    output_dir: Path,
    train_ratio: float,
    val_ratio: float,
    test_ratio: float,
    seed: int = 42
) -> Dict[str, List[Path]]:
    """
    Distribute images to train/val/test splits.
    
    Args:
        image_paths: List of source image paths
        output_dir: Root output directory
        train_ratio: Training split ratio
        val_ratio: Validation split ratio
        test_ratio: Test split ratio
        seed: Random seed
    
    Returns:
        Dictionary mapping split names to output paths
    """
    random.seed(seed)
    random.shuffle(image_paths)
    
    # Calculate split sizes
    total = len(image_paths)
    train_size = int(total * train_ratio)
    val_size = int(total * val_ratio)
    
    # Split images
    train_images = image_paths[:train_size]
    val_images = image_paths[train_size:train_size + val_size]
    test_images = image_paths[train_size + val_size:]
    
    splits = {
        'train': train_images,
        'val': val_images,
        'test': test_images
    }
    
    # Create output directories and copy files
    output_paths = {}
    
    for split_name, split_images in splits.items():
        # Create directory
        split_dir = output_dir / split_name / 'no_gesture2'
        split_dir.mkdir(parents=True, exist_ok=True)
        
        output_paths[split_name] = []
        
        # Copy images
        print(f"\nCopying {len(split_images)} images to {split_name}...")
        for i, src_path in enumerate(tqdm(split_images, desc=f"{split_name}")):
            # Use original filename with index to avoid conflicts
            dst_filename = f"no_gesture2_{i:06d}{src_path.suffix}"
            dst_path = split_dir / dst_filename
            
            shutil.copy2(src_path, dst_path)
            output_paths[split_name].append(dst_path)
    
    return output_paths


def main():
    parser = argparse.ArgumentParser(
        description="Build no_gesture2 class from external images (simplified - no hand detection)"
    )
    parser.add_argument(
        '--source_dir',
        type=Path,
        required=True,
        help='Directory containing source images (e.g., COCO val2017)'
    )
    parser.add_argument(
        '--output_dir',
        type=Path,
        required=True,
        help='Root directory of HaGRID dataset (with train/val/test subdirs)'
    )
    parser.add_argument(
        '--target_count',
        type=int,
        default=7000,
        help='Total number of no_gesture2 images to create'
    )
    parser.add_argument(
        '--train_ratio',
        type=float,
        default=0.7,
        help='Training split ratio'
    )
    parser.add_argument(
        '--val_ratio',
        type=float,
        default=0.15,
        help='Validation split ratio'
    )
    parser.add_argument(
        '--test_ratio',
        type=float,
        default=0.15,
        help='Test split ratio'
    )
    parser.add_argument(
        '--min_size',
        type=int,
        default=160,
        help='Minimum image dimension (default 160 for CelebA compatibility)'
    )
    parser.add_argument(
        '--seed',
        type=int,
        default=42,
        help='Random seed'
    )
    
    args = parser.parse_args()
    
    # Validate paths
    if not args.source_dir.exists():
        raise FileNotFoundError(f"Source directory not found: {args.source_dir}")
    
    if not args.output_dir.exists():
        raise FileNotFoundError(f"Output directory not found: {args.output_dir}")
    
    # Validate split ratios
    total_ratio = args.train_ratio + args.val_ratio + args.test_ratio
    if abs(total_ratio - 1.0) > 0.01:
        raise ValueError(f"Split ratios must sum to 1.0 (got {total_ratio})")
    
    # Initialize filter
    quality_filter = QualityFilter(min_size=args.min_size)
    
    # Sample images
    print(f"\n=== Building no_gesture2 Class ===")
    print(f"Source: {args.source_dir}")
    print(f"Output: {args.output_dir}")
    print(f"Target: {args.target_count} images")
    print(f"Splits: train={args.train_ratio}, val={args.val_ratio}, test={args.test_ratio}")
    print(f"Quality: min_size={args.min_size}px\n")
    
    valid_images = sample_valid_images(
        args.source_dir,
        args.target_count,
        quality_filter,
        args.seed
    )
    
    if len(valid_images) < args.target_count:
        print(f"\nWARNING: Only found {len(valid_images)} valid images "
              f"(target: {args.target_count})")
        print("Consider lowering --target_count or providing more source images")
    else:
        print(f"\n✓ Collected {len(valid_images)} valid images")
    
    # Distribute to splits
    split_paths = distribute_to_splits(
        valid_images,
        args.output_dir,
        args.train_ratio,
        args.val_ratio,
        args.test_ratio,
        args.seed
    )
    
    # Save manifest
    manifest = {
        'class_name': 'no_gesture2',
        'source_dir': str(args.source_dir),
        'total_images': len(valid_images),
        'splits': {k: len(v) for k, v in split_paths.items()},
        'split_ratios': {
            'train': args.train_ratio,
            'val': args.val_ratio,
            'test': args.test_ratio
        },
        'quality_criteria': {
            'min_size': args.min_size,
        },
        'seed': args.seed,
        'created_at': str(Path(__file__).parent)
    }
    
    manifest_path = args.output_dir / 'no_gesture2_manifest.json'
    with open(manifest_path, 'w') as f:
        json.dump(manifest, f, indent=2)
    
    print(f"\n✓ Manifest saved: {manifest_path}")
    print(f"\n=== Summary ===")
    for split, paths in split_paths.items():
        print(f"{split:5s}: {len(paths):5d} images")
    print(f"{'Total':5s}: {len(valid_images):5d} images")
    print("\n✓ no_gesture2 class ready for training!")
    print("\nNext steps:")
    print("1. Verify the new class appears in train/val/test directories")
    print("2. Re-run your training notebook with 24 classes instead of 23")


if __name__ == '__main__':
    main()
