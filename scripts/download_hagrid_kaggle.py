#!/usr/bin/env python3
"""
HaGRID Kaggle Subset Downloader

Uses Kaggle's pre-made HaGRID subset (~1.5GB, 30K images, 384px resolution).
This is the FASTEST and most storage-efficient method.

Prerequisites:
    1. Install Kaggle: pip install kaggle
    2. Setup API key: https://www.kaggle.com/docs/api
       - Go to kaggle.com → Account → Create New API Token
       - Save kaggle.json to ~/.kaggle/

Usage:
    python scripts/download_hagrid_kaggle.py --save_path ./data/hagrid_subset
"""

import argparse
import shutil
from pathlib import Path
import pandas as pd
from tqdm import tqdm


def check_kaggle():
    """Check if Kaggle CLI is installed and configured."""
    try:
        import kaggle

        return True
    except ImportError:
        print("❌ Kaggle library not installed.")
        print("   Install with: pip install kaggle")
        return False
    except OSError:
        print("❌ Kaggle API key not configured.")
        print("   Setup instructions:")
        print("   1. Go to https://www.kaggle.com/account")
        print("   2. Click 'Create New API Token'")
        print("   3. Save kaggle.json to ~/.kaggle/")
        print("   4. chmod 600 ~/.kaggle/kaggle.json")
        return False


def download_kaggle_subset(
    save_path: Path, dataset_name: str = "capitanov/hagrid-sample-30k-384p"
):
    """Download pre-made HaGRID subset from Kaggle."""
    if not check_kaggle():
        return False

    print(f"\n📥 Downloading HaGRID subset from Kaggle...")
    print(f"   Dataset: {dataset_name}")
    print(f"   Size: ~1.5 GB (30,000 images, 384px)")

    save_path.mkdir(parents=True, exist_ok=True)

    # Download using Kaggle CLI
    import kaggle

    try:
        print("   Downloading... (this may take 5-15 minutes)")
        kaggle.api.dataset_download_files(
            dataset_name, path=save_path, unzip=True, quiet=False
        )
    except Exception as e:
        print(f"❌ Download failed: {e}")
        return False

    print("\n✅ Download complete!")
    print(f"   Location: {save_path}")

    # The Kaggle dataset may have a different structure, let's inspect
    print("\n📁 Dataset structure:")
    for item in save_path.rglob("*"):
        if item.is_dir():
            print(f"   📂 {item.relative_to(save_path)}/")

    return True


def create_csv_from_folders(data_root: Path, output_csv: Path):
    """Create annotations.csv from folder structure."""
    print("\n📝 Creating annotations.csv...")

    items = []

    # Look for images in folder structure
    for img_file in data_root.rglob("*.jpg"):
        # Try to infer label from folder structure
        # Common patterns: data/train/gesture_name/image.jpg or data/gesture_name/image.jpg
        parts = img_file.relative_to(data_root).parts

        if len(parts) >= 2:
            # Assume last folder before image is the label
            label = (
                parts[-2]
                if parts[-2] not in ["train", "val", "test"]
                else parts[-3] if len(parts) >= 3 else "unknown"
            )
            split = parts[-2] if parts[-2] in ["train", "val", "test"] else "train"
        else:
            label = "unknown"
            split = "train"

        items.append({"image_path": str(img_file), "label": label, "split": split})

    df = pd.DataFrame(items)
    df.to_csv(output_csv, index=False)

    print(f"✅ Created {output_csv}")
    print(f"   Total images: {len(df)}")
    print(f"   Classes: {df['label'].nunique()}")
    print(f"\nClass distribution:")
    print(df["label"].value_counts())

    return df


def filter_and_reorganize(
    data_root: Path, target_gestures: list, samples_per_class: int
):
    """Filter to specific gestures and limit samples."""
    print(
        f"\n🔧 Filtering to {len(target_gestures)} gestures, {samples_per_class} samples each..."
    )

    output_root = data_root / "filtered"
    output_root.mkdir(exist_ok=True)

    collected = {}

    for gesture in target_gestures:
        # Find all images for this gesture
        gesture_images = list(data_root.rglob(f"*/{gesture}/*.jpg")) + list(
            data_root.rglob(f"{gesture}/*.jpg")
        )

        if not gesture_images:
            print(f"   ⚠️  No images found for '{gesture}'")
            continue

        # Sample up to samples_per_class
        import random

        random.seed(42)
        selected = random.sample(
            gesture_images, min(samples_per_class, len(gesture_images))
        )

        # Copy to filtered folder
        out_dir = output_root / "train" / gesture
        out_dir.mkdir(parents=True, exist_ok=True)

        collected[gesture] = []
        for i, src in enumerate(tqdm(selected, desc=f"Copying {gesture}")):
            dst = out_dir / f"{gesture}_{i:04d}.jpg"
            shutil.copy2(src, dst)
            collected[gesture].append(
                {"image_path": str(dst), "label": gesture, "split": "train"}
            )

    # Create CSV
    all_items = []
    for items in collected.values():
        all_items.extend(items)

    df = pd.DataFrame(all_items)
    csv_path = output_root / "annotations.csv"
    df.to_csv(csv_path, index=False)

    print(f"\n✅ Filtered dataset created!")
    print(f"   Location: {output_root}")
    print(f"   Total images: {len(df)}")
    print(f"   Annotations: {csv_path}")

    return output_root


def main():
    parser = argparse.ArgumentParser(description="Download HaGRID subset from Kaggle")
    parser.add_argument(
        "--save_path", type=str, default="./data/hagrid_subset", help="Save path"
    )
    parser.add_argument(
        "--gestures",
        nargs="+",
        default=["call", "dislike", "fist", "like", "ok", "palm", "peace", "stop"],
        help="Gestures to keep (after download)",
    )
    parser.add_argument(
        "--samples_per_class", type=int, default=500, help="Max samples per class"
    )
    parser.add_argument(
        "--skip_filter",
        action="store_true",
        help="Skip filtering step (keep all downloaded data)",
    )

    args = parser.parse_args()

    save_path = Path(args.save_path)

    print("=" * 60)
    print("HaGRID Kaggle Subset Downloader")
    print("=" * 60)
    print("This will download a pre-made subset from Kaggle:")
    print("  - Size: ~1.5 GB")
    print("  - Images: ~30,000")
    print("  - Resolution: 384px")
    print("  - Classes: All 18 original gestures")
    print("=" * 60)

    # Download
    success = download_kaggle_subset(
        save_path, dataset_name="innominate817/hagrid-classification-512p"
    )

    if not success:
        print("\n❌ Download failed. Alternative methods:")
        print("\n1️⃣  Manual download from Kaggle:")
        print(
            "   - Go to: https://www.kaggle.com/datasets/capitanov/hagrid-sample-30k-384p"
        )
        print("   - Click 'Download' button")
        print(f"   - Extract to: {save_path}")
        print("\n2️⃣  Use even smaller subset (200 images/class):")
        print(
            "   - Download from: https://www.kaggle.com/datasets/innominate817/hagrid-classification-512p"
        )
        print("   - Much smaller: ~500 MB")
        return

    # Try to create CSV from downloaded data
    csv_path = save_path / "annotations.csv"
    if not csv_path.exists():
        create_csv_from_folders(save_path, csv_path)

    # Filter if requested
    if not args.skip_filter and args.gestures:
        filter_and_reorganize(save_path, args.gestures, args.samples_per_class)
        print(f"\n💡 Tip: Update notebook to use: '{save_path}/filtered'")


if __name__ == "__main__":
    main()
