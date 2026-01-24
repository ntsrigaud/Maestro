#!/usr/bin/env python3
"""
Download CelebA dataset for no_gesture2 class.

The CelebA dataset contains face images which are perfect for the no_gesture2 class
because they represent users sitting in front of their computer with no hands visible.
"""

import os
import sys
from pathlib import Path
import zipfile
import subprocess

def download_celeba(output_dir):
    """Download CelebA dataset using Kaggle API."""
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    print(f"\n{'='*70}")
    print("DOWNLOADING CELEBA DATASET")
    print(f"{'='*70}\n")
    
    # Check if Kaggle credentials exist
    kaggle_json = Path.home() / '.kaggle' / 'kaggle.json'
    if not kaggle_json.exists():
        print("❌ Kaggle API credentials not found!")
        print("\nTo download CelebA, you need to:")
        print("1. Go to https://www.kaggle.com/settings/account")
        print("2. Click 'Create New API Token' to download kaggle.json")
        print("3. Place it at: ~/.kaggle/kaggle.json")
        print("4. Run: chmod 600 ~/.kaggle/kaggle.json")
        print("\nAlternatively, manually download from:")
        print("https://www.kaggle.com/datasets/jessicali9530/celeba-dataset")
        print(f"And extract to: {output_path}/")
        return False
    
    print("✓ Kaggle credentials found")
    
    # Download using Kaggle API
    print(f"\nDownloading to: {output_path}")
    print("This may take 10-20 minutes (CelebA is ~1.4GB)...\n")
    
    try:
        # Change to output directory
        os.chdir(output_path)
        
        # Download dataset
        result = subprocess.run(
            ['kaggle', 'datasets', 'download', '-d', 'jessicali9530/celeba-dataset'],
            check=True,
            capture_output=True,
            text=True
        )
        
        print("✓ Download complete")
        
        # Find the downloaded zip file
        zip_files = list(output_path.glob('*.zip'))
        if not zip_files:
            print("❌ No zip file found after download")
            return False
        
        zip_file = zip_files[0]
        print(f"\nExtracting {zip_file.name}...")
        
        # Extract
        with zipfile.ZipFile(zip_file, 'r') as zip_ref:
            zip_ref.extractall(output_path)
        
        print("✓ Extraction complete")
        
        # Clean up zip
        zip_file.unlink()
        print(f"✓ Removed {zip_file.name}")
        
        # Find the img_align_celeba directory
        img_dir = output_path / 'img_align_celeba'
        if img_dir.exists():
            img_count = len(list(img_dir.glob('*.jpg')))
            print(f"\n✓ Found {img_count:,} images in {img_dir}")
            return str(img_dir)
        else:
            # Look for any directory with images
            for subdir in output_path.iterdir():
                if subdir.is_dir():
                    imgs = list(subdir.glob('*.jpg')) + list(subdir.glob('*.png'))
                    if len(imgs) > 1000:
                        print(f"\n✓ Found {len(imgs):,} images in {subdir}")
                        return str(subdir)
            
            print("\n⚠️ Could not find image directory")
            print("Please check the extracted contents manually")
            return False
        
    except subprocess.CalledProcessError as e:
        print(f"\n❌ Download failed: {e}")
        print(f"Error output: {e.stderr}")
        return False
    except Exception as e:
        print(f"\n❌ Error: {e}")
        return False


if __name__ == '__main__':
    output_dir = '/home/rachel/.Neil/CSU-Capstone-Project/notebooks/data/celeba'
    result = download_celeba(output_dir)
    
    if result:
        print(f"\n{'='*70}")
        print("✅ DOWNLOAD COMPLETE")
        print(f"{'='*70}\n")
        print("Next step:")
        print(f"  python scripts/build_no_gesture2_simple.py \\")
        print(f"    --source_dir {result} \\")
        print(f"    --output_dir /home/rachel/.Neil/CSU-Capstone-Project/notebooks/data/hagrid_subset \\")
        print(f"    --target_count 7000")
        sys.exit(0)
    else:
        sys.exit(1)
