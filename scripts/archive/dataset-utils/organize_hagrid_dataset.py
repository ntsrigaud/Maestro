#!/usr/bin/env python3
"""
Organize HaGRID Sample 30K Dataset for Semantic Learning

This script reorganizes the downloaded HaGRID Sample 30K dataset from the
Kaggle format (train_val_* directories with JSON annotations) into the
standard train/val/test split structure required by semantic_gesture_learning.ipynb.

Directory structure transformation:
    Before:
        hagrid-sample-30k-384p/
            hagrid-sample-30k-384p/
                ann_train_val/
                    call.json
                    dislike.json
                    ...
                hagrid_30k/
                    train_val_call/
                        image1.jpg
                        image2.jpg
                    train_val_dislike/
                    ...

    After:
        data/
            hagrid_30k/
                train/
                    call/
                        image1.jpg
                        ...
                    dislike/
                        ...
                val/
                    call/
                    dislike/
                    ...
                test/
                    call/
                    dislike/
                    ...

Usage:
    python scripts/organize_hagrid_dataset.py

    Or with custom paths:
    python scripts/organize_hagrid_dataset.py --input ./hagrid-sample-30k-384p --output ./data
"""

import shutil
from pathlib import Path
import argparse
import random
from tqdm import tqdm


def parse_args():
    parser = argparse.ArgumentParser(
        description="Organize HaGRID Sample 30K dataset into train/val/test splits"
    )
    parser.add_argument(
        "--input",
        type=str,
        default="./notebooks/hagrid-sample-30k-384p/hagrid-sample-30k-384p",
        help="Path to downloaded hagrid-sample-30k-384p dataset (default: ./notebooks/hagrid-sample-30k-384p/hagrid-sample-30k-384p)",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="./notebooks/data",
        help="Output directory for organized dataset (default: ./notebooks/data)",
    )
    parser.add_argument(
        "--train-ratio",
        type=float,
        default=0.7,
        help="Ratio of data for training (default: 0.7)",
    )
    parser.add_argument(
        "--val-ratio",
        type=float,
        default=0.15,
        help="Ratio of data for validation (default: 0.15)",
    )
    parser.add_argument(
        "--test-ratio",
        type=float,
        default=0.15,
        help="Ratio of data for testing (default: 0.15)",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for reproducible splits (default: 42)",
    )
    parser.add_argument(
        "--symlink",
        action="store_true",
        help="Create symbolic links instead of copying files (saves disk space)",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Show what would be done without actually copying files",
    )

    return parser.parse_args()


def get_gesture_classes(input_path):
    """Get list of gesture classes from train_val_* directories"""
    hagrid_path = Path(input_path) / "hagrid_30k"

    classes = []
    for dir_path in sorted(hagrid_path.iterdir()):
        if dir_path.is_dir() and dir_path.name.startswith("train_val_"):
            class_name = dir_path.name.replace("train_val_", "")
            classes.append(class_name)

    return classes


def split_dataset(image_files, train_ratio, val_ratio, test_ratio, seed):
    """Split image files into train/val/test sets"""
    random.seed(seed)

    # Shuffle
    image_files = list(image_files)
    random.shuffle(image_files)

    # Calculate split indices
    total = len(image_files)
    train_end = int(total * train_ratio)
    val_end = train_end + int(total * val_ratio)

    train_files = image_files[:train_end]
    val_files = image_files[train_end:val_end]
    test_files = image_files[val_end:]

    return train_files, val_files, test_files


def organize_dataset(args):
    """Main function to organize the dataset"""
    input_path = Path(args.input)
    output_path = Path(args.output) / "hagrid_30k"

    # Validate input
    if not input_path.exists():
        print(f"❌ Error: Input path does not exist: {input_path}")
        return False

    hagrid_path = input_path / "hagrid_30k"
    if not hagrid_path.exists():
        print(f"❌ Error: hagrid_30k directory not found in {input_path}")
        return False

    # Validate split ratios
    total_ratio = args.train_ratio + args.val_ratio + args.test_ratio
    if abs(total_ratio - 1.0) > 0.01:
        print(f"❌ Error: Split ratios must sum to 1.0 (got {total_ratio})")
        return False

    print("=" * 70)
    print("HAGRID DATASET ORGANIZATION")
    print("=" * 70)
    print(f"Input path:  {input_path}")
    print(f"Output path: {output_path}")
    print(
        f"Split ratios: Train={args.train_ratio:.0%}, Val={args.val_ratio:.0%}, Test={args.test_ratio:.0%}"
    )
    print(f"Method: {'Symbolic links' if args.symlink else 'Copy files'}")
    print(f"Dry run: {'Yes' if args.dry_run else 'No'}")
    print("=" * 70)

    # Get gesture classes
    classes = get_gesture_classes(input_path)
    print(f"\n✓ Found {len(classes)} gesture classes:")
    print(f"  {', '.join(classes)}")

    # Create output directory structure
    if not args.dry_run:
        for split in ["train", "val", "test"]:
            for class_name in classes:
                output_dir = output_path / split / class_name
                output_dir.mkdir(parents=True, exist_ok=True)

    # Process each gesture class
    print(f"\n{'=' * 70}")
    print("ORGANIZING FILES")
    print("=" * 70)

    total_stats = {"train": 0, "val": 0, "test": 0}

    for class_name in classes:
        print(f"\nProcessing '{class_name}'...")

        # Get all image files for this class
        source_dir = hagrid_path / f"train_val_{class_name}"
        image_files = list(source_dir.glob("*.jpg"))

        if not image_files:
            print(f"  ⚠️  No images found in {source_dir}")
            continue

        # Split into train/val/test
        train_files, val_files, test_files = split_dataset(
            image_files, args.train_ratio, args.val_ratio, args.test_ratio, args.seed
        )

        print(f"  Total: {len(image_files)} images")
        print(
            f"  Train: {len(train_files)} | Val: {len(val_files)} | Test: {len(test_files)}"
        )

        # Copy/link files
        splits = {"train": train_files, "val": val_files, "test": test_files}

        for split_name, files in splits.items():
            if args.dry_run:
                total_stats[split_name] += len(files)
                continue

            dest_dir = output_path / split_name / class_name

            for file_path in tqdm(
                files, desc=f"  {split_name.capitalize()}", leave=False
            ):
                dest_path = dest_dir / file_path.name

                if args.symlink:
                    # Create symbolic link
                    if not dest_path.exists():
                        dest_path.symlink_to(file_path.absolute())
                else:
                    # Copy file
                    if not dest_path.exists():
                        shutil.copy2(file_path, dest_path)

                total_stats[split_name] += 1

    # Print summary
    print(f"\n{'=' * 70}")
    print("SUMMARY")
    print("=" * 70)
    print(f"Total images organized:")
    print(f"  Train: {total_stats['train']:,} images ({args.train_ratio:.0%})")
    print(f"  Val:   {total_stats['val']:,} images ({args.val_ratio:.0%})")
    print(f"  Test:  {total_stats['test']:,} images ({args.test_ratio:.0%})")
    print(f"  Total: {sum(total_stats.values()):,} images")

    if args.dry_run:
        print("\n⚠️  Dry run completed. No files were actually copied.")
        print("   Remove --dry-run flag to perform the actual organization.")
    else:
        print(f"\n✓ Dataset successfully organized!")
        print(f"✓ Output location: {output_path}")
        print(f"\nYou can now use this dataset with semantic_gesture_learning.ipynb")
        print(f"The notebook will automatically detect the dataset at: {output_path}")

    print("=" * 70)

    return True


def main():
    args = parse_args()

    try:
        success = organize_dataset(args)
        if success:
            return 0
        else:
            return 1
    except KeyboardInterrupt:
        print("\n\n⚠️  Organization cancelled by user")
        return 130
    except Exception as e:
        print(f"\n❌ Error: {e}")
        import traceback

        traceback.print_exc()
        return 1


if __name__ == "__main__":
    exit(main())
