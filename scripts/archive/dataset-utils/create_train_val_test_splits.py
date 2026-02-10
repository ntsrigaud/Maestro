#!/usr/bin/env python3
"""
Create Train/Val/Test Splits from Existing Test Directory

Takes the existing test directory and splits it into proper train/val/test
splits with stratified sampling to maintain class balance.
"""

import shutil
import random
from pathlib import Path
from collections import defaultdict
import argparse


def create_splits(source_dir, output_dir, train_ratio=0.70, val_ratio=0.15, 
                 test_ratio=0.15, seed=42, use_symlinks=False):
    """
    Split dataset into train/val/test with stratified sampling.
    
    Args:
        source_dir: Path to source directory with class subdirectories
        output_dir: Path to output directory
        train_ratio: Proportion for training (default 0.70)
        val_ratio: Proportion for validation (default 0.15)
        test_ratio: Proportion for testing (default 0.15)
        seed: Random seed for reproducibility
        use_symlinks: Use symlinks instead of copying (saves space)
    """
    random.seed(seed)
    
    source_path = Path(source_dir)
    output_path = Path(output_dir)
    
    # Validate ratios
    assert abs((train_ratio + val_ratio + test_ratio) - 1.0) < 0.01, \
        f"Ratios must sum to 1.0, got {train_ratio + val_ratio + test_ratio}"
    
    # Create output directories
    splits = ['train', 'val', 'test']
    for split in splits:
        (output_path / split).mkdir(parents=True, exist_ok=True)
    
    # Collect all images per class
    class_images = defaultdict(list)
    
    print(f"Scanning {source_path}...")
    for class_dir in sorted(source_path.iterdir()):
        if not class_dir.is_dir():
            continue
        
        class_name = class_dir.name
        
        # Find all image files
        img_extensions = {'.jpg', '.jpeg', '.png', '.JPG', '.JPEG', '.PNG'}
        images = [f for f in class_dir.iterdir() 
                 if f.is_file() and f.suffix in img_extensions]
        
        class_images[class_name] = images
        print(f"  {class_name}: {len(images)} images")
    
    # Split each class
    print(f"\nCreating splits...")
    stats = {split: defaultdict(int) for split in splits}
    
    for class_name, images in class_images.items():
        # Shuffle
        random.shuffle(images)
        
        # Calculate split indices
        n_total = len(images)
        n_train = int(n_total * train_ratio)
        n_val = int(n_total * val_ratio)
        
        # Split
        train_images = images[:n_train]
        val_images = images[n_train:n_train + n_val]
        test_images = images[n_train + n_val:]
        
        # Create class directories in each split
        for split in splits:
            (output_path / split / class_name).mkdir(parents=True, exist_ok=True)
        
        # Copy or link files
        for split, split_images in [('train', train_images), 
                                     ('val', val_images), 
                                     ('test', test_images)]:
            for img_path in split_images:
                dest_path = output_path / split / class_name / img_path.name
                
                if use_symlinks:
                    # Create symlink (relative path)
                    dest_path.symlink_to(img_path.resolve())
                else:
                    # Copy file
                    shutil.copy2(img_path, dest_path)
                
                stats[split][class_name] += 1
    
    # Print statistics
    print(f"\n{'='*60}")
    print("SPLIT STATISTICS")
    print(f"{'='*60}")
    
    for split in splits:
        total = sum(stats[split].values())
        print(f"\n{split.upper()}: {total} images")
        print(f"{'Class':<20} {'Count':>10} {'Percentage':>12}")
        print(f"{'-'*44}")
        for class_name in sorted(stats[split].keys()):
            count = stats[split][class_name]
            pct = 100 * count / total if total > 0 else 0
            print(f"{class_name:<20} {count:>10} {pct:>11.2f}%")
    
    print(f"\n{'='*60}")
    print(f"✓ Dataset split complete!")
    print(f"  Output: {output_path}")
    print(f"  Method: {'Symlinks' if use_symlinks else 'Copied files'}")
    print(f"{'='*60}")


def main():
    parser = argparse.ArgumentParser(
        description="Create train/val/test splits from existing directory"
    )
    parser.add_argument(
        "--source",
        type=str,
        default="data/hagrid_subset_cropped/test",
        help="Source directory with class subdirectories"
    )
    parser.add_argument(
        "--output",
        type=str,
        default="data/hagrid_subset",
        help="Output directory for splits"
    )
    parser.add_argument(
        "--train-ratio",
        type=float,
        default=0.70,
        help="Training split ratio (default: 0.70)"
    )
    parser.add_argument(
        "--val-ratio",
        type=float,
        default=0.15,
        help="Validation split ratio (default: 0.15)"
    )
    parser.add_argument(
        "--test-ratio",
        type=float,
        default=0.15,
        help="Test split ratio (default: 0.15)"
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed (default: 42)"
    )
    parser.add_argument(
        "--symlink",
        action="store_true",
        help="Use symlinks instead of copying"
    )
    
    args = parser.parse_args()
    
    create_splits(
        source_dir=args.source,
        output_dir=args.output,
        train_ratio=args.train_ratio,
        val_ratio=args.val_ratio,
        test_ratio=args.test_ratio,
        seed=args.seed,
        use_symlinks=args.symlink
    )


if __name__ == '__main__':
    main()
