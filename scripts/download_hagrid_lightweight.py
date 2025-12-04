#!/usr/bin/env python3
"""
HaGRID Lightweight Subset Downloader (HuggingFace Datasets)

This script uses HuggingFace's datasets library with streaming to download
ONLY the subset you need, without downloading massive ZIP files.

Usage:
    python scripts/download_hagrid_lightweight.py \
        --save_path ./data/hagrid_subset \
        --samples_per_class 200 \
        --gestures call dislike fist like ok palm peace point rock

Expected size: ~500MB for 200 samples/class (10 classes)
              ~1.5GB for 500 samples/class (10 classes)
"""

import argparse
import os
from pathlib import Path
from collections import defaultdict

from PIL import Image
from tqdm import tqdm
import pandas as pd


def download_hagrid_subset_hf(save_path: Path, gestures: list, samples_per_class: int, seed: int = 42):
    """
    Download HaGRID subset using HuggingFace datasets (streaming mode).
    This avoids downloading the entire dataset.
    """
    try:
        from datasets import load_dataset
    except ImportError:
        print("❌ Error: 'datasets' library not installed.")
        print("Install with: pip install datasets")
        return False

    save_path.mkdir(parents=True, exist_ok=True)
    
    print("📥 Loading HaGRID dataset from HuggingFace (streaming mode)...")
    print("   This will download only the images you need, not the entire dataset.")
    
    # Load dataset with streaming
    try:
        dataset = load_dataset("HuBERT/hagrid", split="train", streaming=True)
    except Exception as e:
        print(f"❌ Error loading dataset: {e}")
        print("\n⚠️  HaGRID may not be available on HuggingFace.")
        print("   Falling back to manual download method...")
        return False
    
    # Collect samples per class
    collected = defaultdict(list)
    target_gestures = set(gestures)
    
    print(f"\n🔍 Collecting {samples_per_class} samples per class...")
    
    total_needed = len(gestures) * samples_per_class
    pbar = tqdm(total=total_needed, desc="Downloading images")
    
    for example in dataset:
        # Check if we have all we need
        if all(len(samples) >= samples_per_class for gesture, samples in collected.items() if gesture in target_gestures):
            break
        
        label = example.get('label') or example.get('gesture')
        
        if label in target_gestures and len(collected[label]) < samples_per_class:
            # Save image
            split = 'train'  # You can modify to include val/test based on some logic
            out_dir = save_path / split / label
            out_dir.mkdir(parents=True, exist_ok=True)
            
            img_idx = len(collected[label])
            img_path = out_dir / f"{label}_{img_idx:05d}.jpg"
            
            # Save image
            img = example['image']
            img.save(img_path)
            
            collected[label].append({
                'image_path': str(img_path),
                'label': label,
                'split': split
            })
            
            pbar.update(1)
    
    pbar.close()
    
    # Create CSV
    all_items = []
    for label, items in collected.items():
        all_items.extend(items)
    
    df = pd.DataFrame(all_items)
    csv_path = save_path / 'annotations.csv'
    df.to_csv(csv_path, index=False)
    
    print(f"\n✅ Download complete!")
    print(f"   Total images: {len(df)}")
    print(f"   Classes: {len(collected)}")
    print(f"   Saved to: {save_path}")
    print(f"   Annotations: {csv_path}")
    
    return True


def download_manual_subset(save_path: Path, gestures: list, samples_per_class: int):
    """
    Fallback: Download pre-cropped HaGRID subset from alternative source.
    Uses the 512px version which is much smaller.
    """
    print("\n📦 Alternative Method: Download 512px subset")
    print("=" * 60)
    print("The HaGRID team provides a pre-resized 512px version (119GB for ALL classes).")
    print("To keep size under 3GB, we recommend:")
    print()
    print("Option 1: Use fewer gestures (5-6 core gestures)")
    print("Option 2: Use fewer samples (100-200 per class)")
    print("Option 3: Use a pre-made subset from Kaggle/other sources")
    print()
    print("Recommended minimal setup:")
    print("  - 5 gestures: fist, palm, peace, thumbs_up (like), thumbs_down (dislike)")
    print("  - 200 samples per class")
    print("  - Estimated size: ~400-600 MB")
    print()
    
    # Check if user wants to proceed with Kaggle dataset
    print("📌 Quick Solution: Use Kaggle's HaGRID Sampled Dataset")
    print("   1. Install Kaggle CLI: pip install kaggle")
    print("   2. Setup API key: https://www.kaggle.com/docs/api")
    print("   3. Download subset: kaggle datasets download -d capitanov/hagrid-sample-30k-384p")
    print()
    print("   This dataset contains 30K images (~1.5GB) across all classes.")
    

def main():
    parser = argparse.ArgumentParser(description="Download HaGRID subset efficiently")
    parser.add_argument("--save_path", type=str, default="./data/hagrid_subset", help="Save path")
    parser.add_argument("--gestures", nargs="+", default=["fist", "palm", "peace", "like", "dislike"], 
                        help="Gestures to download")
    parser.add_argument("--samples_per_class", type=int, default=200, 
                        help="Samples per class (200 recommended for <1GB)")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    
    args = parser.parse_args()
    
    save_path = Path(args.save_path)
    
    print("=" * 60)
    print("HaGRID Lightweight Subset Downloader")
    print("=" * 60)
    print(f"Gestures: {', '.join(args.gestures)}")
    print(f"Samples per class: {args.samples_per_class}")
    print(f"Estimated size: ~{len(args.gestures) * args.samples_per_class * 0.06:.1f} MB")
    print("=" * 60)
    
    # Try HuggingFace method first
    success = download_hagrid_subset_hf(save_path, args.gestures, args.samples_per_class, args.seed)
    
    if not success:
        # Show alternative methods
        download_manual_subset(save_path, args.gestures, args.samples_per_class)


if __name__ == "__main__":
    main()
