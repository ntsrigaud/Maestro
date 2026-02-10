#!/usr/bin/env python3
"""
Download MS COCO Person Subset for no_gesture2 Class

Downloads a subset of MS COCO 2017 validation images containing people.
These will be filtered for hand presence to create the no_gesture2 negative class.

Usage:
    python scripts/download_coco_person_subset.py --max-images 10000

Requirements:
    pip install requests tqdm pillow
"""

import argparse
import json
import urllib.request
from pathlib import Path
from tqdm import tqdm
import time


def download_coco_annotations(output_dir):
    """Download COCO 2017 validation annotations."""
    annotations_url = "http://images.cocodataset.org/annotations/annotations_trainval2017.zip"
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    annotations_zip = output_path / "annotations_trainval2017.zip"
    
    if annotations_zip.exists():
        print(f"✓ Annotations already downloaded: {annotations_zip}")
        return annotations_zip
    
    print(f"Downloading COCO 2017 annotations...")
    print(f"URL: {annotations_url}")
    print(f"Destination: {annotations_zip}")
    print("This may take several minutes (252 MB)...")
    
    urllib.request.urlretrieve(annotations_url, annotations_zip)
    print(f"✓ Downloaded: {annotations_zip}")
    
    return annotations_zip


def extract_annotations(annotations_zip, output_dir):
    """Extract annotations from zip file."""
    import zipfile
    
    output_path = Path(output_dir)
    annotations_dir = output_path / "annotations"
    
    if annotations_dir.exists():
        print(f"✓ Annotations already extracted: {annotations_dir}")
        return annotations_dir
    
    print(f"Extracting annotations...")
    with zipfile.ZipFile(annotations_zip, 'r') as zip_ref:
        zip_ref.extractall(output_path)
    
    print(f"✓ Extracted to: {annotations_dir}")
    return annotations_dir


def load_person_images(annotations_file, max_images):
    """Load image IDs that contain person annotations."""
    print(f"\nLoading annotations from {annotations_file}...")
    
    with open(annotations_file, 'r') as f:
        coco_data = json.load(f)
    
    # Find person category ID
    person_category_id = None
    for category in coco_data['categories']:
        if category['name'] == 'person':
            person_category_id = category['id']
            break
    
    if person_category_id is None:
        raise ValueError("Person category not found in COCO annotations")
    
    print(f"✓ Person category ID: {person_category_id}")
    
    # Find images with person annotations
    image_ids_with_person = set()
    for annotation in coco_data['annotations']:
        if annotation['category_id'] == person_category_id:
            image_ids_with_person.add(annotation['image_id'])
    
    print(f"✓ Found {len(image_ids_with_person)} images with person annotations")
    
    # Map image IDs to filenames
    image_id_to_filename = {}
    for image in coco_data['images']:
        if image['id'] in image_ids_with_person:
            image_id_to_filename[image['id']] = image['file_name']
    
    # Limit to max_images
    selected_filenames = list(image_id_to_filename.values())[:max_images]
    
    print(f"✓ Selected {len(selected_filenames)} images for download")
    
    return selected_filenames


def download_images(filenames, output_dir, max_retries=3):
    """Download COCO images."""
    base_url = "http://images.cocodataset.org/val2017/"
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    print(f"\nDownloading {len(filenames)} images...")
    print(f"Destination: {output_path}")
    
    downloaded = 0
    failed = []
    
    for filename in tqdm(filenames, desc="Downloading"):
        dest_file = output_path / filename
        
        # Skip if already exists
        if dest_file.exists():
            downloaded += 1
            continue
        
        # Download with retries
        success = False
        for attempt in range(max_retries):
            try:
                url = base_url + filename
                urllib.request.urlretrieve(url, dest_file)
                downloaded += 1
                success = True
                break
            except Exception as e:
                if attempt == max_retries - 1:
                    failed.append((filename, str(e)))
                else:
                    time.sleep(1)  # Wait before retry
        
        if not success:
            # Clean up partial download
            if dest_file.exists():
                dest_file.unlink()
    
    print(f"\n✓ Downloaded: {downloaded}/{len(filenames)} images")
    
    if failed:
        print(f"⚠️  Failed: {len(failed)} images")
        print("First 5 failures:")
        for filename, error in failed[:5]:
            print(f"  {filename}: {error}")
    
    return downloaded, failed


def main():
    parser = argparse.ArgumentParser(
        description="Download MS COCO person subset for no_gesture2 class"
    )
    
    parser.add_argument('--max-images', type=int, default=10000,
                       help='Maximum number of images to download (default: 10000)')
    parser.add_argument('--output', type=str, default='./data/coco_person_subset',
                       help='Output directory (default: ./data/coco_person_subset)')
    
    args = parser.parse_args()
    
    print("="*60)
    print("DOWNLOAD MS COCO PERSON SUBSET")
    print("="*60)
    print(f"Max images: {args.max_images}")
    print(f"Output: {args.output}")
    print("="*60)
    
    output_path = Path(args.output)
    
    # Step 1: Download annotations
    print("\nStep 1: Download annotations")
    annotations_zip = download_coco_annotations(output_path.parent)
    
    # Step 2: Extract annotations
    print("\nStep 2: Extract annotations")
    annotations_dir = extract_annotations(annotations_zip, output_path.parent)
    
    # Step 3: Load person image IDs
    print("\nStep 3: Find images with people")
    annotations_file = annotations_dir / "instances_val2017.json"
    if not annotations_file.exists():
        raise FileNotFoundError(f"Annotations not found: {annotations_file}")
    
    filenames = load_person_images(annotations_file, args.max_images)
    
    # Step 4: Download images
    print("\nStep 4: Download images")
    downloaded, failed = download_images(filenames, output_path)
    
    # Summary
    print("\n" + "="*60)
    print("DOWNLOAD COMPLETE")
    print("="*60)
    print(f"✓ Downloaded: {downloaded} images")
    print(f"  Location: {output_path}")
    
    if failed:
        print(f"\n⚠️  {len(failed)} images failed to download")
    
    print("\nNext step:")
    print(f"  python scripts/build_no_gesture2_class.py \\")
    print(f"    --source {output_path} \\")
    print(f"    --output data/hagrid_subset \\")
    print(f"    --max-samples 5000 \\")
    print(f"    --source-name 'MS COCO 2017 Val' \\")
    print(f"    --license 'CC BY 4.0' \\")
    print(f"    --attribution 'Microsoft COCO' \\")
    print(f"    --url 'https://cocodataset.org'")


if __name__ == '__main__':
    main()
