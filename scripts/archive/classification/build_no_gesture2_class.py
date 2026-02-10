#!/usr/bin/env python3
"""
Build no_gesture2 Class from External Dataset

Filters images from an external source (MS COCO, Wider Face, etc.) to create
a "hands out-of-frame" negative class for gesture recognition.

Usage:
    python scripts/build_no_gesture2_class.py \
        --source ./data/coco_person_subset \
        --output ./data/hagrid_subset \
        --max-samples 5000 \
        --confidence 0.3

Requirements:
    pip install mediapipe opencv-python tqdm scikit-learn
"""

import argparse
import json
import random
import shutil
from pathlib import Path
from collections import Counter
from datetime import datetime

import cv2
from tqdm import tqdm
from sklearn.model_selection import train_test_split

# MediaPipe for hand detection
try:
    import mediapipe as mp
    MP_AVAILABLE = True
except ImportError:
    MP_AVAILABLE = False
    print("Warning: MediaPipe not available. Install with: pip install mediapipe")


class HandDetector:
    """Wrapper for MediaPipe Hands detector."""
    
    def __init__(self, confidence_threshold=0.3):
        if not MP_AVAILABLE:
            raise ImportError("MediaPipe required for hand detection")
        
        self.mp_hands = mp.solutions.hands
        self.hands = self.mp_hands.Hands(
            static_image_mode=True,
            max_num_hands=2,
            min_detection_confidence=confidence_threshold
        )
        self.confidence_threshold = confidence_threshold
    
    def has_hands(self, image_path):
        """
        Check if image contains visible hands.
        
        Returns:
            bool: True if hands detected, False otherwise
            None: If image is invalid/corrupted
        """
        try:
            image = cv2.imread(str(image_path))
            if image is None:
                return None
            
            # Convert BGR to RGB
            image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            
            # Process image
            results = self.hands.process(image_rgb)
            
            # Check for hand landmarks
            if results.multi_hand_landmarks:
                return True
            return False
            
        except Exception as e:
            print(f"Error processing {image_path}: {e}")
            return None
    
    def __del__(self):
        """Clean up resources."""
        if hasattr(self, 'hands'):
            self.hands.close()


class QualityFilter:
    """Apply quality checks to images."""
    
    def __init__(self, min_width=224, min_height=224, 
                 min_aspect_ratio=0.5, max_aspect_ratio=2.0):
        self.min_width = min_width
        self.min_height = min_height
        self.min_aspect_ratio = min_aspect_ratio
        self.max_aspect_ratio = max_aspect_ratio
    
    def passes_quality_check(self, image_path):
        """
        Check if image meets quality criteria.
        
        Returns:
            tuple: (passes: bool, reason: str or None)
        """
        try:
            image = cv2.imread(str(image_path))
            if image is None:
                return False, "invalid_file"
            
            height, width = image.shape[:2]
            
            # Check minimum resolution
            if width < self.min_width or height < self.min_height:
                return False, f"resolution_too_low_{width}x{height}"
            
            # Check aspect ratio
            aspect_ratio = width / height
            if aspect_ratio < self.min_aspect_ratio or aspect_ratio > self.max_aspect_ratio:
                return False, f"aspect_ratio_{aspect_ratio:.2f}"
            
            return True, None
            
        except Exception as e:
            return False, f"error_{str(e)[:20]}"


def filter_images(source_dir, max_samples, hand_detector, quality_filter, seed=42):
    """
    Filter images from source directory.
    
    Args:
        source_dir: Path to source images
        max_samples: Maximum number of images to accept
        hand_detector: HandDetector instance
        quality_filter: QualityFilter instance
        seed: Random seed
    
    Returns:
        tuple: (accepted_images, rejection_log)
    """
    random.seed(seed)
    
    source_path = Path(source_dir)
    
    # Find all image files
    img_extensions = {'.jpg', '.jpeg', '.png', '.JPG', '.JPEG', '.PNG'}
    image_files = []
    for ext in img_extensions:
        image_files.extend(source_path.rglob(f'*{ext}'))
    
    print(f"Found {len(image_files)} images in {source_path}")
    
    # Shuffle for random sampling
    random.shuffle(image_files)
    
    accepted = []
    rejection_log = []
    
    pbar = tqdm(image_files, desc="Filtering images")
    for img_path in pbar:
        if len(accepted) >= max_samples:
            break
        
        # Quality check
        passes_quality, quality_reason = quality_filter.passes_quality_check(img_path)
        if not passes_quality:
            rejection_log.append({
                'path': str(img_path),
                'reason': f'quality_{quality_reason}'
            })
            continue
        
        # Hand detection
        has_hands = hand_detector.has_hands(img_path)
        
        if has_hands is None:
            rejection_log.append({
                'path': str(img_path),
                'reason': 'hand_detection_failed'
            })
        elif has_hands:
            rejection_log.append({
                'path': str(img_path),
                'reason': 'hands_detected'
            })
        else:
            accepted.append(img_path)
            pbar.set_postfix({'accepted': len(accepted)})
    
    return accepted, rejection_log


def create_no_gesture2_class(
    accepted_images,
    output_root,
    train_ratio=0.70,
    val_ratio=0.15,
    test_ratio=0.15,
    seed=42
):
    """
    Create no_gesture2 class directories and split images.
    
    Args:
        accepted_images: List of Path objects to filtered images
        output_root: Root directory for dataset (e.g., data/hagrid_subset)
        train_ratio: Proportion for training
        val_ratio: Proportion for validation
        test_ratio: Proportion for testing
        seed: Random seed
    """
    output_path = Path(output_root)
    
    # Validate ratios
    assert abs((train_ratio + val_ratio + test_ratio) - 1.0) < 0.01, \
        f"Ratios must sum to 1.0, got {train_ratio + val_ratio + test_ratio}"
    
    # Split images
    train_imgs, temp_imgs = train_test_split(
        accepted_images, test_size=(1 - train_ratio), random_state=seed
    )
    val_imgs, test_imgs = train_test_split(
        temp_imgs, 
        test_size=(test_ratio / (val_ratio + test_ratio)), 
        random_state=seed
    )
    
    print(f"\nCreating no_gesture2 class...")
    
    # Create directories and copy images
    for split, images in [('train', train_imgs), ('val', val_imgs), ('test', test_imgs)]:
        split_dir = output_path / split / 'no_gesture2'
        split_dir.mkdir(parents=True, exist_ok=True)
        
        print(f"  Copying {len(images)} images to {split}/...")
        for img_path in tqdm(images, desc=f"{split}", leave=False):
            dest_path = split_dir / img_path.name
            
            # Handle name collisions
            if dest_path.exists():
                stem = img_path.stem
                suffix = img_path.suffix
                counter = 1
                while dest_path.exists():
                    dest_path = split_dir / f"{stem}_{counter}{suffix}"
                    counter += 1
            
            shutil.copy2(img_path, dest_path)
    
    print(f"\n✓ no_gesture2 class created successfully")
    print(f"  Train: {len(train_imgs)} images")
    print(f"  Val: {len(val_imgs)} images")
    print(f"  Test: {len(test_imgs)} images")
    print(f"  Total: {len(accepted_images)} images")


def save_provenance(args, accepted_count, rejection_log, output_path):
    """Save provenance metadata."""
    
    rejection_reasons = Counter([r['reason'] for r in rejection_log])
    total_processed = accepted_count + len(rejection_log)
    
    provenance = {
        'class': 'no_gesture2',
        'description': 'Hands out-of-frame negative examples',
        'source_dataset': args.source_name,
        'source_path': str(args.source),
        'license': args.license,
        'attribution': args.attribution,
        'url': args.url,
        'date_created': datetime.now().isoformat(),
        'filtering': {
            'hand_detector': 'MediaPipe Hands',
            'confidence_threshold': args.confidence,
            'total_processed': total_processed,
            'accepted': accepted_count,
            'rejected': len(rejection_log),
            'rejection_rate': len(rejection_log) / total_processed if total_processed > 0 else 0,
            'rejection_reasons': dict(rejection_reasons)
        },
        'quality_checks': {
            'min_resolution': f"{args.min_width}x{args.min_height}",
            'aspect_ratio_range': [args.min_aspect, args.max_aspect]
        },
        'splits': {
            'train_ratio': args.train_ratio,
            'val_ratio': args.val_ratio,
            'test_ratio': args.test_ratio,
            'seed': args.seed
        }
    }
    
    provenance_path = output_path / 'no_gesture2_provenance.json'
    with open(provenance_path, 'w') as f:
        json.dump(provenance, f, indent=2)
    
    print(f"\n✓ Saved provenance to {provenance_path}")
    
    # Save rejection log (first 1000 entries to avoid huge files)
    rejection_log_path = output_path / 'no_gesture2_rejection_log.json'
    with open(rejection_log_path, 'w') as f:
        json.dump({
            'total_rejections': len(rejection_log),
            'reasons': dict(rejection_reasons),
            'sample_rejections': rejection_log[:1000]
        }, f, indent=2)
    
    print(f"✓ Saved rejection log to {rejection_log_path}")


def main():
    parser = argparse.ArgumentParser(
        description="Build no_gesture2 class from external dataset"
    )
    
    # Required arguments
    parser.add_argument('--source', type=str, required=True,
                       help='Source directory with images')
    parser.add_argument('--output', type=str, required=True,
                       help='Output dataset root (e.g., data/hagrid_subset)')
    
    # Filtering parameters
    parser.add_argument('--max-samples', type=int, default=5000,
                       help='Maximum number of images to accept (default: 5000)')
    parser.add_argument('--confidence', type=float, default=0.3,
                       help='Hand detection confidence threshold (default: 0.3)')
    
    # Quality parameters
    parser.add_argument('--min-width', type=int, default=224,
                       help='Minimum image width (default: 224)')
    parser.add_argument('--min-height', type=int, default=224,
                       help='Minimum image height (default: 224)')
    parser.add_argument('--min-aspect', type=float, default=0.5,
                       help='Minimum aspect ratio (default: 0.5)')
    parser.add_argument('--max-aspect', type=float, default=2.0,
                       help='Maximum aspect ratio (default: 2.0)')
    
    # Split parameters
    parser.add_argument('--train-ratio', type=float, default=0.70,
                       help='Training split ratio (default: 0.70)')
    parser.add_argument('--val-ratio', type=float, default=0.15,
                       help='Validation split ratio (default: 0.15)')
    parser.add_argument('--test-ratio', type=float, default=0.15,
                       help='Test split ratio (default: 0.15)')
    parser.add_argument('--seed', type=int, default=42,
                       help='Random seed (default: 42)')
    
    # Provenance metadata
    parser.add_argument('--source-name', type=str, default='Unknown',
                       help='Name of source dataset (e.g., "MS COCO 2017")')
    parser.add_argument('--license', type=str, default='Unknown',
                       help='License of source dataset')
    parser.add_argument('--attribution', type=str, default='Unknown',
                       help='Attribution string for source dataset')
    parser.add_argument('--url', type=str, default='',
                       help='URL for source dataset')
    
    args = parser.parse_args()
    
    print("="*60)
    print("BUILD NO_GESTURE2 CLASS")
    print("="*60)
    print(f"Source: {args.source}")
    print(f"Output: {args.output}")
    print(f"Target samples: {args.max_samples}")
    print(f"Hand detection confidence: {args.confidence}")
    print("="*60)
    
    # Initialize components
    hand_detector = HandDetector(confidence_threshold=args.confidence)
    quality_filter = QualityFilter(
        min_width=args.min_width,
        min_height=args.min_height,
        min_aspect_ratio=args.min_aspect,
        max_aspect_ratio=args.max_aspect
    )
    
    # Filter images
    print("\nPhase 1: Filtering images...")
    accepted_images, rejection_log = filter_images(
        args.source,
        args.max_samples,
        hand_detector,
        quality_filter,
        args.seed
    )
    
    if len(accepted_images) == 0:
        print("\n❌ No images passed filtering. Check source directory and thresholds.")
        return
    
    print(f"\n✓ Accepted {len(accepted_images)} / {len(accepted_images) + len(rejection_log)} images")
    print(f"  Rejection rate: {100 * len(rejection_log) / (len(accepted_images) + len(rejection_log)):.1f}%")
    
    # Create class
    print("\nPhase 2: Creating no_gesture2 class...")
    create_no_gesture2_class(
        accepted_images,
        args.output,
        args.train_ratio,
        args.val_ratio,
        args.test_ratio,
        args.seed
    )
    
    # Save provenance
    print("\nPhase 3: Saving provenance...")
    save_provenance(args, len(accepted_images), rejection_log, Path(args.output))
    
    print("\n" + "="*60)
    print("✓ BUILD COMPLETE")
    print("="*60)
    print(f"\nNext steps:")
    print(f"1. Manually inspect samples in {args.output}/train/no_gesture2/")
    print(f"2. Update notebook to retrain with 24 classes")
    print(f"3. Compare metrics before/after adding no_gesture2")


if __name__ == '__main__':
    main()
