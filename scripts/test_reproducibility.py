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

# HaGRID gesture classes (alphabetically sorted as in training)
# This is the standard 18-class set from the full HaGRID dataset
HAGRID_GESTURE_NAMES = [
    'call', 'dislike', 'fist', 'four', 'like', 'mute', 'ok', 'one',
    'palm', 'peace', 'peace_inverted', 'rock', 'stop', 'stop_inverted',
    'three', 'three2', 'two_up', 'two_up_inverted'
]


def load_gesture_classes(checkpoint_path: Path) -> List[str]:
    """
    Load gesture class names dynamically from checkpoint.
    
    Args:
        checkpoint_path: Path to model checkpoint
        
    Returns:
        List of class names in training order
        
    Raises:
        ValueError: If classes cannot be determined
    """
    checkpoint = torch.load(checkpoint_path, map_location='cpu')
    
    # Try checkpoint metadata first
    if 'class_names' in checkpoint:
        classes = checkpoint['class_names']
        logger.info(f"Loaded {len(classes)} classes from checkpoint metadata")
        return classes
    
    # Fallback: use HaGRID gesture names if class count matches
    state_dict = checkpoint.get('model_state_dict', checkpoint)
    if 'head.weight' in state_dict:
        num_classes = state_dict['head.weight'].shape[0]
        
        # If model has 18 classes, use the standard HaGRID gesture names
        if num_classes == len(HAGRID_GESTURE_NAMES):
            preview_count = 3
            logger.warning(
                f"No class_names in checkpoint. Using standard HaGRID gesture names "
                f"({num_classes} classes). Ensure your model was trained with HaGRID "
                f"in alphabetical order: {', '.join(HAGRID_GESTURE_NAMES[:preview_count])}..."
            )
            return HAGRID_GESTURE_NAMES
        
        # For non-standard class counts, use generic labels
        logger.warning(
            f"No class_names in checkpoint. Inferred {num_classes} classes. "
            f"Class count doesn't match standard HaGRID (18 classes). "
            f"Using generic labels (class_0, class_1, ...). "
            f"Consider re-saving checkpoint with 'class_names' metadata."
        )
        return [f"class_{i}" for i in range(num_classes)]
    
    raise ValueError("Cannot determine classes from checkpoint")


# ============================================================================
# TEST 1: REMOVED - MediaPipe Dependency Eliminated
# ============================================================================
# Test 1 (create_cropped_test_set) has been removed as MediaPipe HandDetector
# and HandCropper classes no longer exist in the codebase. The distribution
# shift issue has been resolved by removing hand detection and using center-crop
# preprocessing instead.
#
# For current validation, use Test 2 (webcam capture) or see:
# - scripts/demo_gesture_recognition.py (production demo with center-crop)
# - notebooks/finetune_vit_hagrid_CLEAN_BASELINE.ipynb (training pipeline)
# ============================================================================


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
    gesture_classes: List[str],
    frames_per_gesture: int = 30,
    camera_index: int = 0
) -> Dict[str, int]:
    """
    Capture webcam frames for each gesture using center-crop preprocessing.
    
    NOTE: Updated to use center-crop approach (no hand detection).
    Captures full center-cropped frames matching the production pipeline.
    
    Args:
        output_dir: Directory to save frames
        checkpoint_path: Model checkpoint (unused, kept for compatibility)
        gesture_classes: List of gesture class names to capture
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
    
    # Initialize components (center-crop preprocessing, no hand detection)
    input_mgr = InputManager(camera_index=camera_index)
    
    # Import preprocessing functions
    from recognition_engine.preprocessing import center_crop_frame, get_crop_coordinates
    
    # Start camera
    if not input_mgr.start_continuous_capture():
        logger.error("Failed to open camera")
        return {}
    
    logger.info("Camera ready")
    
    stats = {}
    
    try:
        for gesture in gesture_classes:
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
                
                # Apply center-crop preprocessing (matches production pipeline)
                preprocessed = center_crop_frame(frame, target_size=224, bgr_to_rgb=True)
                
                # Save preprocessed frame
                save_path = gesture_dir / f"frame_{frames_captured:04d}.jpg"
                Image.fromarray(preprocessed).save(save_path)
                frames_captured += 1
                
                # Visual feedback - show crop region on display frame
                h, w = frame.shape[:2]
                x1, y1, x2, y2 = get_crop_coordinates((h, w))
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                
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
        description="Reproducibility tests for production deployment audit. "
                    "Runs Test 2: webcam capture and evaluation. "
                    "(Test 1 removed - MediaPipe dependency eliminated)"
    )
    parser.add_argument(
        '--checkpoint',
        type=str,
        default='./checkpoints/best_model.pth',
        help="Path to model checkpoint"
    )
    parser.add_argument(
        '--webcam-dir',
        type=str,
        default='./data/webcam_test_frames',
        help="Path to save webcam frames"
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
    
    # Load gesture classes dynamically from checkpoint
    gesture_classes = load_gesture_classes(checkpoint_path)
    logger.info(f"Using {len(gesture_classes)} gesture classes")
    
    # Detect number of classes from checkpoint
    if 'model_state_dict' in checkpoint:
        state_dict = checkpoint['model_state_dict']
    else:
        state_dict = checkpoint
    
    # Get num_classes from head.weight shape
    num_classes = state_dict['head.weight'].shape[0]
    logger.info(f"Detected {num_classes} classes from checkpoint")
    
    # Verify consistency
    if len(gesture_classes) != num_classes:
        logger.warning(
            f"Class count mismatch: {len(gesture_classes)} names vs {num_classes} in model. "
            f"This may cause incorrect evaluation."
        )
    
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
    
    # Test 2: Capture and evaluate webcam frames
    # Capture frames if they don't exist
    webcam_frames_exist = webcam_dir.exists() and any(webcam_dir.glob('*/*.jpg'))
    
    if not webcam_frames_exist:
        capture_stats = capture_webcam_gestures(
            output_dir=webcam_dir,
            checkpoint_path=checkpoint_path,
            gesture_classes=gesture_classes,
            frames_per_gesture=30
        )
    else:
        logger.info(f"Webcam frames already exist: {webcam_dir}")
    
    # Evaluate on webcam frames if directory exists
    if webcam_dir.exists():
        webcam_dataset = ImageFolder(webcam_dir, transform=transform)
        webcam_loader = DataLoader(webcam_dataset, batch_size=args.batch_size,
                                  shuffle=False, num_workers=0)
        
        webcam_acc, webcam_labels, webcam_preds, webcam_conf = evaluate_model(
            model, webcam_loader, device, "Webcam Frames (Offline)"
        )
        
        print_evaluation_results(
            "Webcam Frames (Offline)", webcam_acc, webcam_labels,
            webcam_preds, webcam_conf, gesture_classes
        )
    
    logger.info("\n" + "="*70)
    logger.info("REPRODUCIBILITY TESTS COMPLETE")
    logger.info("="*70)
    logger.info("See PRODUCTION_DEPLOYMENT_AUDIT.md for interpretation and next steps")


if __name__ == '__main__':
    main()
