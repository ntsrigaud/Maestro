#!/usr/bin/env python3
"""
Download public domain images for no_gesture2 class (hands out-of-frame)

This script downloads person/scene images from public datasets suitable
for creating a "hands absent" negative class.

Usage:
    python scripts/download_no_gesture2_images.py \
        --output ./notebooks/data/coco_person_subset \
        --max-samples 6000
"""

import argparse
import urllib.request
import json
from pathlib import Path
from tqdm import tqdm


def download_unsplash_lite_sample(output_dir, max_samples=6000):
    """
    Download sample images from Unsplash Lite Dataset (CC0 license)
    Alternative: Use any collection of person/scene images without visible hands.
    
    For production use, download from:
    - MS COCO 2017 (person category): https://cocodataset.org/
    - Open Images V7 (person): https://storage.googleapis.com/openimages/web/index.html
    - Unsplash Lite: https://unsplash.com/data
    """
    print(f"="*70)
    print("DOWNLOAD SOURCE IMAGES FOR NO_GESTURE2")
    print(f"="*70)
    print(f"\nTarget: {max_samples} images")
    print(f"Output: {output_dir}")
    print("\nNOTE: This script provides guidance for downloading images.")
    print("You have several options:\n")
    
    print("Option 1: MS COCO 2017 (Recommended)")
    print("-" * 70)
    print("  Dataset: MS COCO 2017 Train Images")
    print("  Size: ~19GB (118k images)")
    print("  License: CC BY 4.0")
    print("  URL: http://images.cocodataset.org/zips/train2017.zip")
    print("  Commands:")
    print(f"    mkdir -p {output_dir}")
    print(f"    cd {output_dir}")
    print("    wget http://images.cocodataset.org/zips/train2017.zip")
    print("    unzip train2017.zip")
    print("    # This gives you train2017/ with 118k images\n")
    
    print("Option 2: MS COCO 2017 Val (Smaller)")
    print("-" * 70)
    print("  Dataset: MS COCO 2017 Validation Images")
    print("  Size: ~1GB (5k images)")
    print("  License: CC BY 4.0")
    print("  URL: http://images.cocodataset.org/zips/val2017.zip")
    print("  Commands:")
    print(f"    mkdir -p {output_dir}")
    print(f"    cd {output_dir}")
    print("    wget http://images.cocodataset.org/zips/val2017.zip")
    print("    unzip val2017.zip")
    print("    # This gives you val2017/ with 5k images\n")
    
    print("Option 3: Use your own image collection")
    print("-" * 70)
    print("  - Any collection of person/scene images")
    print("  - Ensure minimal hand visibility")
    print("  - At least 6000 images recommended")
    print(f"  - Place in: {output_dir}\n")
    
    print("\nAfter downloading, run:")
    print("  python scripts/build_no_gesture2_class.py \\")
    print(f"    --source {output_dir}/train2017 \\  # or val2017")
    print("    --output ./notebooks/data/hagrid_subset \\")
    print("    --max-samples 5000")
    
    return None


def auto_download_coco_val(output_dir, max_samples=6000):
    """Automated download of COCO val2017 (smaller dataset)."""
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    url = "http://images.cocodataset.org/zips/val2017.zip"
    zip_path = output_path / "val2017.zip"
    extract_dir = output_path / "val2017"
    
    if extract_dir.exists() and len(list(extract_dir.glob('*.jpg'))) > 100:
        print(f"✓ Dataset already exists at {extract_dir}")
        print(f"  Found {len(list(extract_dir.glob('*.jpg')))} images")
        return str(extract_dir)
    
    print(f"\nDownloading COCO val2017 (~1GB)...")
    print(f"From: {url}")
    print(f"To: {zip_path}")
    print("\nThis may take several minutes...\n")
    
    try:
        # Download with progress bar
        urllib.request.urlretrieve(url, zip_path)
        print(f"\n✓ Downloaded to {zip_path}")
        
        # Extract
        print(f"\nExtracting...")
        import zipfile
        with zipfile.ZipFile(zip_path, 'r') as zip_ref:
            zip_ref.extractall(output_path)
        
        print(f"✓ Extracted to {extract_dir}")
        
        # Clean up zip
        zip_path.unlink()
        print(f"✓ Removed {zip_path}")
        
        # Count images
        image_count = len(list(extract_dir.glob('*.jpg')))
        print(f"\n✓ Ready: {image_count} images in {extract_dir}")
        
        return str(extract_dir)
        
    except Exception as e:
        print(f"\n❌ Download failed: {e}")
        print("\nPlease download manually:")
        print(f"  wget {url}")
        print(f"  unzip val2017.zip")
        return None


def main():
    parser = argparse.ArgumentParser(
        description="Download source images for no_gesture2 class"
    )
    parser.add_argument('--output', type=str, 
                       default='./notebooks/data/coco_person_subset',
                       help='Output directory for downloaded images')
    parser.add_argument('--max-samples', type=int, default=6000,
                       help='Target number of images')
    parser.add_argument('--auto', action='store_true',
                       help='Automatically download COCO val2017 (~1GB)')
    
    args = parser.parse_args()
    
    if args.auto:
        source_dir = auto_download_coco_val(args.output, args.max_samples)
        if source_dir:
            print(f"\n" + "="*70)
            print("DOWNLOAD COMPLETE")
            print("="*70)
            print(f"\nNext step: Run filtering script")
            print(f"\n  python scripts/build_no_gesture2_class.py \\")
            print(f"    --source {source_dir} \\")
            print(f"    --output ./notebooks/data/hagrid_subset \\")
            print(f"    --max-samples 5000 \\")
            print(f"    --confidence 0.3")
    else:
        download_unsplash_lite_sample(args.output, args.max_samples)


if __name__ == '__main__':
    main()
