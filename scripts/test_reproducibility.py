#!/usr/bin/env python3
"""
Reproducibility Tests for Production Deployment Audit

⚠️  PARTIALLY OBSOLETE - MediaPipe Removed ⚠️

Test 1 (cropped test set) is OBSOLETE as MediaPipe has been removed from the pipeline.
The distribution shift has been eliminated by removing hand detection entirely.

Test 2 (full-frame baseline) remains valid for validation purposes.

This script is kept for historical reference. For current validation, see:
- scripts/demo_gesture_recognition.py (production demo with center-crop)
- notebooks/finetune_vit_hagrid_CLEAN_BASELINE.ipynb (training pipeline)

Original purpose:
This script implemented controlled experiments to validate the distribution shift hypothesis:
1. ❌ OBSOLETE: Create cropped test set (simulate production pipeline on test data)
2. ✅ VALID: Measure accuracy on full-frame test sets
3. ✅ VALID: Capture live webcam frames for offline evaluation
4. ✅ VALID: Compare confidence distributions across datasets

Result: Distribution shift confirmed (25.19% vs 92.49%). 
Solution: MediaPipe removed, center-crop preprocessing implemented.
"""

import sys
import time
import argparse
from pathlib import Path
from collections import defaultdict, Counter
from typing import Dict, List, Tuple

import cv2
import numpy as np
from PIL import Image
from tqdm import tqdm

import torch
import timm
from torchvision import transforms
from torchvision.datasets import ImageFolder
from torch.utils.data import DataLoader
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

# Import from installed package (after pip install -e .)
from input_manager import InputManager
from shared.logger import get_logger

# NOTE: HandDetector/HandCropper removed - Test 1 is now obsolete
# To use Test 1, you would need to reinstall MediaPipe and restore the hand_detection module
# However, this is NOT recommended as the architectural decision removed MediaPipe entirely.

logger = get_logger(__name__)

# Gesture classes (must match training)
GESTURE_CLASSES = [
    "call", "dislike", "fist", "four", "like", "mute", "ok", "one",
    "palm", "peace", "peace_inverted", "rock", "stop", "stop_inverted",
    "three", "three2", "two_up", "two_up_inverted",
]


# ============================================================================
# TEST 1: CREATE CROPPED TEST SET (OBSOLETE - MediaPipe removed)
# ============================================================================
# This function used HandDetector and HandCropper which have been removed.
# Kept for historical reference only. DO NOT USE.
# ============================================================================

def create_cropped_test_set(
    source_dir: Path,
    output_dir: Path,
    detector=None,  # OBSOLETE: HandDetector removed
    cropper=None,   # OBSOLETE: HandCropper removed
    max_images_per_class: int = None
) -> Dict[str, int]:
    """
    Create a test set of cropped hands from full-frame images.
    
    This simulates the production pipeline on test data to measure
    accuracy drop due to distribution shift.
    
    Args:
        source_dir: Directory with full-frame test images (ImageFolder structure)
        output_dir: Directory to save cropped images
        detector: Hand detector instance
        cropper: Hand cropper instance
        max_images_per_class: Limit per class (None = process all)
    
    Returns:
        Dictionary with statistics per class
    """
    logger.info("="*70)
    logger.info("TEST 1: Creating Cropped Test Set")
    logger.info("="*70)
    logger.info(f"Source: {source_dir}")
    logger.info(f"Output: {output_dir}")
    
    stats = defaultdict(lambda: {'processed': 0, 'saved': 0, 'no_hands': 0, 'errors': 0})
    
    # Process each class directory
    for class_dir in sorted(source_dir.iterdir()):
        if not class_dir.is_dir():
            continue
        
        class_name = class_dir.name
        if class_name not in GESTURE_CLASSES:
            logger.warning(f"Skipping unknown class: {class_name}")
            continue
        
        # Create output directory
        output_class_dir = output_dir / class_name
        output_class_dir.mkdir(parents=True, exist_ok=True)
        
        # Get all images
        image_files = list(class_dir.glob('*.jpg')) + list(class_dir.glob('*.png'))
        if max_images_per_class:
            image_files = image_files[:max_images_per_class]
        
        logger.info(f"\nProcessing class '{class_name}': {len(image_files)} images")
        
        # Process each image
        for img_path in tqdm(image_files, desc=f"  {class_name}"):
            try:
                stats[class_name]['processed'] += 1
                
                # Read image
                frame = cv2.imread(str(img_path))
                if frame is None:
                    logger.warning(f"Failed to read {img_path}")
                    stats[class_name]['errors'] += 1
                    continue
                
                # Detect hands
                detections = detector.detect(frame)
                if len(detections) == 0:
                    stats[class_name]['no_hands'] += 1
                    continue
                
                # Crop first hand (matches production)
                hand_roi = cropper.crop_hand(frame, detections[0])
                if hand_roi is None or hand_roi.image is None:
                    stats[class_name]['errors'] += 1
                    continue
                
                # Save cropped hand (RGB format)
                save_path = output_class_dir / img_path.name
                Image.fromarray(hand_roi.image).save(save_path)
                stats[class_name]['saved'] += 1
                
            except Exception as e:
                logger.error(f"Error processing {img_path}: {e}")
                stats[class_name]['errors'] += 1
    
    # Print statistics
    logger.info("\n" + "="*70)
    logger.info("CROPPING STATISTICS")
    logger.info("="*70)
    
    total_processed = 0
    total_saved = 0
    total_no_hands = 0
    total_errors = 0
    
    logger.info(f"{'Class':<20} {'Processed':>10} {'Saved':>10} {'No Hands':>10} {'Errors':>10}")
    logger.info("-"*70)
    
    for class_name in sorted(stats.keys()):
        s = stats[class_name]
        logger.info(f"{class_name:<20} {s['processed']:>10} {s['saved']:>10} "
                   f"{s['no_hands']:>10} {s['errors']:>10}")
        total_processed += s['processed']
        total_saved += s['saved']
        total_no_hands += s['no_hands']
        total_errors += s['errors']
    
    logger.info("-"*70)
    logger.info(f"{'TOTAL':<20} {total_processed:>10} {total_saved:>10} "
               f"{total_no_hands:>10} {total_errors:>10}")
    logger.info(f"\nSuccess rate: {100*total_saved/total_processed:.1f}%")
    logger.info(f"No hand detected: {100*total_no_hands/total_processed:.1f}%")
    logger.info(f"Errors: {100*total_errors/total_processed:.1f}%")
    
    return dict(stats)


def evaluate_model(
    model: torch.nn.Module,
    dataloader: DataLoader,
    device: torch.device,
    dataset_name: str
) -> Tuple[float, np.ndarray, np.ndarray, List[float]]:
    """
    Evaluate model on a dataset.
    
    Args:
        model: PyTorch model
        dataloader: DataLoader for evaluation
        device: Device to run on
        dataset_name: Name for logging
    
    Returns:
        Tuple of (accuracy, all_labels, all_preds, all_confidences)
    """
    logger.info(f"\nEvaluating on {dataset_name}...")
    
    model.eval()
    all_preds = []
    all_labels = []
    all_confidences = []
    
    with torch.no_grad():
        for images, labels in tqdm(dataloader, desc=f"  Evaluating"):
            images = images.to(device)
            outputs = model(images)
            probs = torch.softmax(outputs, dim=1)
            
            preds = outputs.argmax(dim=1).cpu().numpy()
            confidences = probs.max(dim=1)[0].cpu().numpy()
            
            all_preds.extend(preds)
            all_labels.extend(labels.numpy())
            all_confidences.extend(confidences)
    
    all_preds = np.array(all_preds)
    all_labels = np.array(all_labels)
    all_confidences = np.array(all_confidences)
    
    accuracy = accuracy_score(all_labels, all_preds)
    
    return accuracy, all_labels, all_preds, all_confidences


def print_evaluation_results(
    dataset_name: str,
    accuracy: float,
    labels: np.ndarray,
    preds: np.ndarray,
    confidences: np.ndarray,
    class_names: List[str]
):
    """Print detailed evaluation results."""
    logger.info("\n" + "="*70)
    logger.info(f"{dataset_name.upper()} EVALUATION RESULTS")
    logger.info("="*70)
    logger.info(f"Accuracy: {accuracy*100:.2f}%")
    
    # Confidence statistics
    logger.info(f"\nConfidence Statistics:")
    logger.info(f"  Mean:   {confidences.mean():.3f}")
    logger.info(f"  Std:    {confidences.std():.3f}")
    logger.info(f"  Median: {np.median(confidences):.3f}")
    logger.info(f"  Min:    {confidences.min():.3f}")
    logger.info(f"  Max:    {confidences.max():.3f}")
    logger.info(f"  P10:    {np.percentile(confidences, 10):.3f}")
    logger.info(f"  P90:    {np.percentile(confidences, 90):.3f}")
    
    # Per-class accuracy
    logger.info(f"\nPer-Class Accuracy:")
    for i, class_name in enumerate(class_names):
        class_mask = labels == i
        if class_mask.sum() == 0:
            continue
        class_acc = (preds[class_mask] == labels[class_mask]).mean()
        class_conf = confidences[class_mask].mean()
        logger.info(f"  {class_name:<20} Acc: {class_acc*100:>5.1f}%  Conf: {class_conf:.3f}")
    
    # Confusion matrix diagonality
    cm = confusion_matrix(labels, preds)
    diagonality = 100 * np.trace(cm) / cm.sum()
    logger.info(f"\nConfusion Matrix Diagonality: {diagonality:.2f}%")
    
    # Classification report
    logger.info(f"\nClassification Report:")
    print(classification_report(labels, preds, target_names=class_names, 
                                digits=3, zero_division=0))


def compare_results(
    baseline_name: str,
    baseline_acc: float,
    baseline_conf: np.ndarray,
    test_name: str,
    test_acc: float,
    test_conf: np.ndarray
):
    """Compare two evaluation results."""
    logger.info("\n" + "="*70)
    logger.info("COMPARISON")
    logger.info("="*70)
    
    acc_drop = baseline_acc - test_acc
    acc_drop_pct = 100 * acc_drop / baseline_acc if baseline_acc > 0 else 0
    
    logger.info(f"{baseline_name} Accuracy: {baseline_acc*100:.2f}%")
    logger.info(f"{test_name} Accuracy:    {test_acc*100:.2f}%")
    logger.info(f"Accuracy Drop:           {acc_drop*100:.2f}% ({acc_drop_pct:.1f}% relative)")
    
    conf_drop = baseline_conf.mean() - test_conf.mean()
    
    logger.info(f"\n{baseline_name} Mean Confidence: {baseline_conf.mean():.3f}")
    logger.info(f"{test_name} Mean Confidence:    {test_conf.mean():.3f}")
    logger.info(f"Confidence Drop:                {conf_drop:.3f}")
    
    # Interpretation
    logger.info("\n" + "="*70)
    logger.info("INTERPRETATION")
    logger.info("="*70)
    
    if acc_drop > 0.2:  # >20% absolute drop
        logger.info("🔴 CRITICAL: Severe accuracy drop indicates distribution shift")
        logger.info("   → Model trained on wrong input distribution")
        logger.info("   → Retrain on cropped hands (Priority #1)")
    elif acc_drop > 0.1:  # >10% absolute drop
        logger.info("🟡 MEDIUM: Significant accuracy drop")
        logger.info("   → Distribution shift is a factor")
        logger.info("   → Consider retraining on matched distribution")
    else:
        logger.info("🟢 LOW: Minimal accuracy drop")
        logger.info("   → Distribution shift is not primary issue")
        logger.info("   → Investigate other factors (temporal, confidence)")


def capture_webcam_gestures(
    output_dir: Path,
    checkpoint_path: Path,
    frames_per_gesture: int = 30,
    camera_index: int = 0
) -> Dict[str, int]:
    """
    Capture webcam frames for each gesture.
    
    Args:
        output_dir: Directory to save frames
        checkpoint_path: Model checkpoint (for detector/cropper config)
        frames_per_gesture: Number of frames to capture per gesture
        camera_index: Camera device index
    
    Returns:
        Dictionary with frame counts per gesture
    """
    logger.info("\n" + "="*70)
    logger.info("TEST 2: Capturing Webcam Frames")
    logger.info("="*70)
    logger.info(f"Output: {output_dir}")
    logger.info(f"Frames per gesture: {frames_per_gesture}")
    
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Initialize components
    input_mgr = InputManager(camera_index=camera_index)
    detector = HandDetector()
    cropper = HandCropper(
        target_size=(224, 224),
        padding=0.2,
        square_crop=True,
        normalize=False,
        bgr_to_rgb=True,
    )
    
    # Start camera
    if not input_mgr.start_continuous_capture():
        logger.error("Failed to open camera")
        return {}
    
    logger.info("Camera ready")
    
    stats = {}
    
    try:
        for gesture in GESTURE_CLASSES:
            logger.info(f"\n{'='*70}")
            logger.info(f"GESTURE: {gesture}")
            logger.info(f"{'='*70}")
            logger.info("Position your hand to show this gesture")
            logger.info("Press SPACE when ready, or 's' to skip")
            
            # Wait for user confirmation
            while True:
                frame_data = input_mgr.get_latest_frame()
                if frame_data is not None:
                    frame = frame_data.frame
                    cv2.putText(frame, f"Show gesture: {gesture}", (10, 30),
                              cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
                    cv2.putText(frame, "Press SPACE to start, 's' to skip", (10, 70),
                              cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
                    cv2.imshow("Webcam Capture", frame)
                
                key = cv2.waitKey(1) & 0xFF
                if key == ord(' '):
                    break
                elif key == ord('s'):
                    logger.info(f"Skipped {gesture}")
                    stats[gesture] = 0
                    break
            
            if stats.get(gesture) == 0:
                continue
            
            # Create gesture directory
            gesture_dir = output_dir / gesture
            gesture_dir.mkdir(exist_ok=True)
            
            # Capture frames
            frames_captured = 0
            logger.info(f"Capturing {frames_per_gesture} frames...")
            
            while frames_captured < frames_per_gesture:
                frame_data = input_mgr.get_latest_frame()
                if frame_data is None:
                    continue
                
                frame = frame_data.frame
                
                # Detect hands
                detections = detector.detect(frame)
                
                if len(detections) > 0:
                    # Crop hand
                    hand_roi = cropper.crop_hand(frame, detections[0])
                    
                    if hand_roi is not None and hand_roi.image is not None:
                        # Save frame
                        save_path = gesture_dir / f"frame_{frames_captured:04d}.jpg"
                        Image.fromarray(hand_roi.image).save(save_path)
                        frames_captured += 1
                        
                        # Visual feedback
                        x, y, w, h = detections[0].bbox
                        cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
                
                # Display progress
                cv2.putText(frame, f"{gesture}: {frames_captured}/{frames_per_gesture}", 
                          (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
                cv2.imshow("Webcam Capture", frame)
                cv2.waitKey(1)
            
            stats[gesture] = frames_captured
            logger.info(f"✓ Captured {frames_captured} frames for {gesture}")
    
    finally:
        input_mgr.stop_capture()
        cv2.destroyAllWindows()
    
    # Print summary
    logger.info("\n" + "="*70)
    logger.info("CAPTURE SUMMARY")
    logger.info("="*70)
    total_frames = sum(stats.values())
    logger.info(f"Total frames captured: {total_frames}")
    for gesture, count in stats.items():
        logger.info(f"  {gesture:<20} {count:>5} frames")
    
    return stats


def main():
    parser = argparse.ArgumentParser(
        description="Reproducibility tests for production deployment audit"
    )
    parser.add_argument(
        '--test',
        type=str,
        choices=['1', '2', 'all'],
        default='all',
        help="Which test to run: 1 (cropped test set), 2 (webcam capture), all"
    )
    parser.add_argument(
        '--checkpoint',
        type=str,
        default='./checkpoints/best_model.pth',
        help="Path to model checkpoint"
    )
    parser.add_argument(
        '--test-dir',
        type=str,
        default='./notebooks/data/hagrid_subset/test',
        help="Path to full-frame test set"
    )
    parser.add_argument(
        '--cropped-dir',
        type=str,
        default='./data/hagrid_subset_cropped/test',
        help="Path to save/load cropped test set"
    )
    parser.add_argument(
        '--webcam-dir',
        type=str,
        default='./data/webcam_test_frames',
        help="Path to save webcam frames"
    )
    parser.add_argument(
        '--max-images',
        type=int,
        default=None,
        help="Max images per class to process (for quick testing)"
    )
    parser.add_argument(
        '--batch-size',
        type=int,
        default=32,
        help="Batch size for evaluation"
    )
    parser.add_argument(
        '--device',
        type=str,
        choices=['cuda', 'mps', 'cpu', 'auto'],
        default='auto',
        help="Device to use"
    )
    
    args = parser.parse_args()
    
    # Set device
    if args.device == 'auto':
        if torch.backends.mps.is_available():
            device = torch.device('mps')
        elif torch.cuda.is_available():
            device = torch.device('cuda')
        else:
            device = torch.device('cpu')
    else:
        device = torch.device(args.device)
    
    logger.info(f"Using device: {device}")
    
    # Paths
    checkpoint_path = Path(args.checkpoint)
    test_dir = Path(args.test_dir)
    cropped_dir = Path(args.cropped_dir)
    webcam_dir = Path(args.webcam_dir)
    
    # Verify checkpoint exists
    if not checkpoint_path.exists():
        logger.error(f"Checkpoint not found: {checkpoint_path}")
        logger.error("Train model first: notebooks/finetune_vit_hagrid.ipynb")
        return
    
    # Load model
    logger.info(f"Loading model from {checkpoint_path}")
    
    # Load checkpoint first to get number of classes
    checkpoint = torch.load(checkpoint_path, map_location=device)
    
    # Detect number of classes from checkpoint
    if 'model_state_dict' in checkpoint:
        state_dict = checkpoint['model_state_dict']
        # Also load class names if available
        if 'class_names' in checkpoint:
            GESTURE_CLASSES[:] = checkpoint['class_names']
            logger.info(f"Loaded {len(GESTURE_CLASSES)} class names from checkpoint")
    else:
        state_dict = checkpoint
    
    # Get num_classes from head.weight shape
    num_classes = state_dict['head.weight'].shape[0]
    logger.info(f"Detected {num_classes} classes from checkpoint")
    
    # Create model with correct number of classes
    model = timm.create_model('vit_base_patch16_224', pretrained=False, 
                             num_classes=num_classes)
    model.load_state_dict(state_dict)
    
    model = model.to(device)
    model.eval()
    logger.info("✓ Model loaded")
    
    # Transform (must match training)
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                           std=[0.229, 0.224, 0.225])
    ])
    
    # Test 1: Create and evaluate cropped test set
    if args.test in ['1', 'all']:
        # Initialize detector and cropper
        detector = HandDetector()
        cropper = HandCropper(
            target_size=(224, 224),
            padding=0.2,
            square_crop=True,
            normalize=False,
            bgr_to_rgb=True,
        )
        
        # Create cropped test set if it doesn't exist
        if not cropped_dir.exists():
            if not test_dir.exists():
                logger.error(f"Test directory not found: {test_dir}")
                logger.error("Cannot run Test 1 without full-frame test data")
            else:
                create_cropped_test_set(
                    source_dir=test_dir,
                    output_dir=cropped_dir,
                    detector=detector,
                    cropper=cropper,
                    max_images_per_class=args.max_images
                )
        else:
            logger.info(f"Cropped test set already exists: {cropped_dir}")
        
        # Evaluate on both datasets
        if test_dir.exists() and cropped_dir.exists():
            # Full-frame test set
            full_dataset = ImageFolder(test_dir, transform=transform)
            full_loader = DataLoader(full_dataset, batch_size=args.batch_size, 
                                   shuffle=False, num_workers=0)
            
            full_acc, full_labels, full_preds, full_conf = evaluate_model(
                model, full_loader, device, "Full-Frame Test Set"
            )
            
            print_evaluation_results(
                "Full-Frame Test Set", full_acc, full_labels, 
                full_preds, full_conf, GESTURE_CLASSES
            )
            
            # Cropped test set
            cropped_dataset = ImageFolder(cropped_dir, transform=transform)
            cropped_loader = DataLoader(cropped_dataset, batch_size=args.batch_size,
                                      shuffle=False, num_workers=0)
            
            cropped_acc, cropped_labels, cropped_preds, cropped_conf = evaluate_model(
                model, cropped_loader, device, "Cropped Test Set"
            )
            
            print_evaluation_results(
                "Cropped Test Set", cropped_acc, cropped_labels,
                cropped_preds, cropped_conf, GESTURE_CLASSES
            )
            
            # Compare results
            compare_results(
                "Full-Frame", full_acc, full_conf,
                "Cropped", cropped_acc, cropped_conf
            )
    
    # Test 2: Capture and evaluate webcam frames
    if args.test in ['2', 'all']:
        # Capture frames
        if not webcam_dir.exists() or len(list(webcam_dir.glob('*/*.jpg'))) == 0:
            capture_stats = capture_webcam_gestures(
                output_dir=webcam_dir,
                checkpoint_path=checkpoint_path,
                frames_per_gesture=30
            )
        else:
            logger.info(f"Webcam frames already exist: {webcam_dir}")
        
        # Evaluate on webcam frames
        if webcam_dir.exists() and len(list(webcam_dir.glob('*/*.jpg'))) > 0:
            webcam_dataset = ImageFolder(webcam_dir, transform=transform)
            webcam_loader = DataLoader(webcam_dataset, batch_size=args.batch_size,
                                      shuffle=False, num_workers=0)
            
            webcam_acc, webcam_labels, webcam_preds, webcam_conf = evaluate_model(
                model, webcam_loader, device, "Webcam Frames (Offline)"
            )
            
            print_evaluation_results(
                "Webcam Frames (Offline)", webcam_acc, webcam_labels,
                webcam_preds, webcam_conf, GESTURE_CLASSES
            )
    
    logger.info("\n" + "="*70)
    logger.info("REPRODUCIBILITY TESTS COMPLETE")
    logger.info("="*70)
    logger.info("See PRODUCTION_DEPLOYMENT_AUDIT.md for interpretation and next steps")


if __name__ == '__main__':
    main()
