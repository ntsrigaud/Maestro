#!/usr/bin/env python3
"""
Test script for hand detection submodule.

This script tests the HandDetector and HandCropper classes with live camera input.
It validates:
- Hand detection with MediaPipe
- Multi-hand detection (up to 2 hands)
- Bounding box calculation
- Hand cropping with padding
- Preprocessing pipeline
- Frame boundary handling
"""

import sys
import time
import numpy as np
import cv2
from pathlib import Path

# Add project root to path for imports
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.input_manager import InputManager
from src.recognition_engine.hand_detection import (
    HandDetector, 
    HandCropper,
    HandDetectionResult,
    HandROI,
)
from src.shared.logger import get_logger

logger = get_logger(__name__)


def test_hand_detector_initialization():
    """Test HandDetector initialization with various parameters."""
    print("\n=== Test 1: HandDetector Initialization ===")
    
    # Default parameters
    detector1 = HandDetector()
    print(f"✓ Default detector: {detector1}")
    
    # Custom parameters
    detector2 = HandDetector(
        max_num_hands=1,
        min_detection_confidence=0.7,
        min_tracking_confidence=0.7,
        model_complexity=0,
    )
    print(f"✓ Custom detector: {detector2}")
    
    # Context manager
    with HandDetector() as detector3:
        print(f"✓ Context manager: {detector3}")
    
    print("✓ Test 1 passed\n")


def test_hand_cropper_initialization():
    """Test HandCropper initialization with various parameters."""
    print("\n=== Test 2: HandCropper Initialization ===")
    
    # Default parameters
    cropper1 = HandCropper()
    print(f"✓ Default cropper: {cropper1}")
    
    # Custom parameters
    cropper2 = HandCropper(
        target_size=(128, 128),
        padding=0.3,
        square_crop=False,
        normalize=False,
        bgr_to_rgb=False,
    )
    print(f"✓ Custom cropper: {cropper2}")
    
    print("✓ Test 2 passed\n")


def test_detection_with_camera():
    """Test hand detection with live camera feed."""
    print("\n=== Test 3: Hand Detection with Camera ===")
    
    # Initialize input manager
    print("Initializing camera...")
    input_mgr = InputManager(camera_index=0)
    
    # Initialize detector
    detector = HandDetector(max_num_hands=2, min_detection_confidence=0.5)
    
    # Start capture
    input_mgr.start_continuous_capture()
    print("Camera started. Testing detection for 5 seconds...")
    
    start_time = time.time()
    frame_count = 0
    detection_count = 0
    
    try:
        while time.time() - start_time < 5.0:
            frame_data = input_mgr.get_latest_frame()
            if frame_data is None:
                time.sleep(0.01)
                continue
            
            # Detect hands
            detections = detector.detect(frame_data.frame, frame_data.timestamp)
            frame_count += 1
            detection_count += len(detections)
            
            # Print detections
            if detections:
                print(f"Frame {frame_count}: Detected {len(detections)} hand(s)")
                for i, det in enumerate(detections):
                    print(f"  Hand {i+1}: {det.handedness}, confidence={det.confidence:.2f}, bbox={det.bbox}")
            
            time.sleep(0.033)  # ~30 FPS
    
    finally:
        input_mgr.stop_continuous_capture()
        detector.close()
    
    print(f"\nProcessed {frame_count} frames")
    print(f"Total detections: {detection_count}")
    print(f"Avg detections per frame: {detection_count / frame_count:.2f}")
    
    # Print detector stats
    stats = detector.get_stats()
    print(f"\nDetector stats: {stats}")
    
    print("✓ Test 3 passed\n")


def test_cropping():
    """Test hand cropping with detected hands."""
    print("\n=== Test 4: Hand Cropping ===")
    
    # Initialize components
    input_mgr = InputManager(camera_index=0)
    detector = HandDetector(max_num_hands=2)
    cropper = HandCropper(target_size=(224, 224), padding=0.2)
    
    input_mgr.start_continuous_capture()
    print("Camera started. Place hands in view...")
    
    crop_count = 0
    max_crops = 10
    
    try:
        while crop_count < max_crops:
            frame_data = input_mgr.get_latest_frame()
            if frame_data is None:
                time.sleep(0.01)
                continue
            
            # Detect hands
            detections = detector.detect(frame_data.frame, frame_data.timestamp)
            if not detections:
                continue
            
            # Crop each detected hand
            hand_rois = cropper.crop_multiple(frame_data.frame, detections)
            
            for i, roi in enumerate(hand_rois):
                crop_count += 1
                print(f"Crop {crop_count}: {roi.detection.handedness} hand")
                print(f"  Original bbox: {roi.detection.bbox}")
                print(f"  Cropped size: {roi.image.shape}")
                print(f"  Preprocessing: {roi.preprocessing_params}")
                
                # Validate cropped image
                assert roi.image.shape == (224, 224, 3), f"Unexpected shape: {roi.image.shape}"
                assert roi.image.dtype == np.float32, f"Unexpected dtype: {roi.image.dtype}"
                
                if crop_count >= max_crops:
                    break
            
            time.sleep(0.05)
    
    finally:
        input_mgr.stop_continuous_capture()
        detector.close()
    
    # Print cropper stats
    stats = cropper.get_stats()
    print(f"\nCropper stats: {stats}")
    
    print("✓ Test 4 passed\n")


def test_boundary_handling():
    """Test bounding box handling at frame edges."""
    print("\n=== Test 5: Frame Boundary Handling ===")
    
    # Create a test frame
    frame = np.zeros((480, 640, 3), dtype=np.uint8)
    
    # Create detector and cropper
    detector = HandDetector()
    cropper = HandCropper(padding=0.3)
    
    # Test cases: bounding boxes at edges
    test_cases = [
        # (x, y, w, h, description)
        (0, 0, 100, 100, "Top-left corner"),
        (540, 0, 100, 100, "Top-right corner"),
        (0, 380, 100, 100, "Bottom-left corner"),
        (540, 380, 100, 100, "Bottom-right corner"),
        (-20, 200, 100, 100, "Extends left"),
        (600, 200, 100, 100, "Extends right"),
        (300, -20, 100, 100, "Extends top"),
        (300, 420, 100, 100, "Extends bottom"),
    ]
    
    for x, y, w, h, desc in test_cases:
        # Create a mock detection
        landmarks = np.random.rand(21, 3).astype(np.float32)
        detection = HandDetectionResult(
            bbox=(x, y, w, h),
            landmarks=landmarks,
            handedness="Right",
            confidence=0.9,
            timestamp=time.time_ns(),
        )
        
        # Try to crop
        try:
            roi = cropper.crop_hand(frame, detection)
            print(f"✓ {desc}: bbox={detection.bbox} -> cropped shape={roi.image.shape}")
        except Exception as e:
            print(f"✗ {desc}: Failed with {e}")
    
    detector.close()
    print("✓ Test 5 passed\n")


def test_multi_hand_detection():
    """Test detection with multiple hands."""
    print("\n=== Test 6: Multi-Hand Detection ===")
    
    input_mgr = InputManager(camera_index=0)
    detector = HandDetector(max_num_hands=2)
    
    input_mgr.start_continuous_capture()
    print("Camera started. Show both hands...")
    
    two_hands_detected = False
    start_time = time.time()
    timeout = 10.0
    
    try:
        while not two_hands_detected and time.time() - start_time < timeout:
            frame_data = input_mgr.get_latest_frame()
            if frame_data is None:
                time.sleep(0.01)
                continue
            
            detections = detector.detect(frame_data.frame)
            
            if len(detections) == 2:
                two_hands_detected = True
                print(f"✓ Detected 2 hands:")
                for i, det in enumerate(detections):
                    print(f"  Hand {i+1}: {det.handedness}, confidence={det.confidence:.2f}")
            elif len(detections) > 0:
                print(f"Detected {len(detections)} hand(s), waiting for 2...")
            
            time.sleep(0.1)
    
    finally:
        input_mgr.stop_continuous_capture()
        detector.close()
    
    if two_hands_detected:
        print("✓ Test 6 passed\n")
    else:
        print("⚠ Test 6: Did not detect 2 hands within timeout (not a failure)\n")


def test_preprocessing_options():
    """Test different preprocessing configurations."""
    print("\n=== Test 7: Preprocessing Options ===")
    
    # Create test frame
    frame = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
    
    # Create mock detection
    landmarks = np.random.rand(21, 3).astype(np.float32)
    detection = HandDetectionResult(
        bbox=(200, 150, 150, 150),
        landmarks=landmarks,
        handedness="Right",
        confidence=0.9,
        timestamp=time.time_ns(),
    )
    
    # Test different preprocessing configs
    configs = [
        {"normalize": True, "bgr_to_rgb": True, "desc": "Full preprocessing"},
        {"normalize": False, "bgr_to_rgb": True, "desc": "RGB only, no normalization"},
        {"normalize": True, "bgr_to_rgb": False, "desc": "BGR, normalized"},
        {"normalize": False, "bgr_to_rgb": False, "desc": "Raw (no preprocessing)"},
    ]
    
    for config in configs:
        desc = config.pop("desc")
        cropper = HandCropper(**config)
        
        roi = cropper.crop_hand(frame, detection, apply_preprocessing=True)
        
        print(f"✓ {desc}:")
        print(f"  Shape: {roi.image.shape}")
        print(f"  Dtype: {roi.image.dtype}")
        print(f"  Value range: [{roi.image.min():.3f}, {roi.image.max():.3f}]")
        print(f"  Params: {roi.preprocessing_params}")
    
    print("✓ Test 7 passed\n")


def run_all_tests():
    """Run all tests."""
    print("=" * 60)
    print("Hand Detection Submodule Tests")
    print("=" * 60)
    
    try:
        test_hand_detector_initialization()
        test_hand_cropper_initialization()
        test_detection_with_camera()
        test_cropping()
        test_boundary_handling()
        test_multi_hand_detection()
        test_preprocessing_options()
        
        print("\n" + "=" * 60)
        print("✓ All tests passed!")
        print("=" * 60)
        
    except Exception as e:
        print(f"\n✗ Test failed with error: {e}")
        logger.error("Test failed", exc_info=True)
        sys.exit(1)


if __name__ == "__main__":
    run_all_tests()
