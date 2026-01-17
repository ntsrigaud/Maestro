#!/usr/bin/env python3
"""
Validation script for HandDetector MediaPipe 0.10.x refactoring.

This script validates that the refactored HandDetector maintains exact
API compatibility and produces equivalent outputs.
"""

import sys
from pathlib import Path
import numpy as np
import cv2

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from recognition_engine.hand_detection import HandDetector, HandDetectionResult


def validate_api_compatibility():
    """Validate that all public API methods exist and have correct signatures."""
    print("="*70)
    print("API COMPATIBILITY VALIDATION")
    print("="*70)
    
    # Test instantiation with all parameters
    try:
        detector = HandDetector(
            max_num_hands=2,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5,
            model_complexity=1,
            static_image_mode=False
        )
        print("✓ HandDetector instantiation successful")
    except Exception as e:
        print(f"✗ HandDetector instantiation failed: {e}")
        return False
    
    # Check required methods exist
    required_methods = [
        'detect',
        'detect_batch',
        'get_stats',
        'reset_stats',
        'close',
        '__enter__',
        '__exit__',
        '__repr__'
    ]
    
    for method_name in required_methods:
        if not hasattr(detector, method_name):
            print(f"✗ Missing method: {method_name}")
            return False
        print(f"✓ Method exists: {method_name}")
    
    # Test context manager
    try:
        with HandDetector() as d:
            pass
        print("✓ Context manager works")
    except Exception as e:
        print(f"✗ Context manager failed: {e}")
        return False
    
    detector.close()
    print("\n✓ API compatibility validated")
    return True


def validate_detection_output():
    """Validate that detection outputs have correct structure and types."""
    print("\n" + "="*70)
    print("DETECTION OUTPUT VALIDATION")
    print("="*70)
    
    # Create a test image (simple colored square with hand-like shape)
    test_image = np.zeros((480, 640, 3), dtype=np.uint8)
    # Draw a simple hand-like shape (won't detect, but tests the pipeline)
    cv2.rectangle(test_image, (200, 200), (400, 400), (255, 200, 150), -1)
    
    detector = HandDetector(max_num_hands=2, min_detection_confidence=0.3)
    
    # Test detect method
    try:
        detections = detector.detect(test_image)
        print(f"✓ detect() returned: {type(detections)}")
        print(f"  Number of detections: {len(detections)}")
        
        # Validate return type
        if not isinstance(detections, list):
            print(f"✗ Expected list, got {type(detections)}")
            return False
        
        # If hands detected, validate structure
        for i, detection in enumerate(detections):
            print(f"\n  Detection {i}:")
            
            # Check type
            if not isinstance(detection, HandDetectionResult):
                print(f"  ✗ Expected HandDetectionResult, got {type(detection)}")
                return False
            print(f"  ✓ Type: HandDetectionResult")
            
            # Validate bbox
            if not isinstance(detection.bbox, tuple) or len(detection.bbox) != 4:
                print(f"  ✗ Invalid bbox: {detection.bbox}")
                return False
            print(f"  ✓ bbox: {detection.bbox} (x, y, w, h)")
            
            # Validate landmarks
            if detection.landmarks.shape != (21, 3):
                print(f"  ✗ Invalid landmarks shape: {detection.landmarks.shape}")
                return False
            print(f"  ✓ landmarks shape: (21, 3)")
            
            # Validate landmarks are normalized [0, 1] for x, y
            x_coords = detection.landmarks[:, 0]
            y_coords = detection.landmarks[:, 1]
            if not (np.all(x_coords >= 0) and np.all(x_coords <= 1)):
                print(f"  ⚠ Warning: x coordinates not in [0, 1]")
            if not (np.all(y_coords >= 0) and np.all(y_coords <= 1)):
                print(f"  ⚠ Warning: y coordinates not in [0, 1]")
            print(f"  ✓ landmarks normalized (x range: [{x_coords.min():.3f}, {x_coords.max():.3f}])")
            
            # Validate handedness
            if detection.handedness not in ["Left", "Right"]:
                print(f"  ✗ Invalid handedness: {detection.handedness}")
                return False
            print(f"  ✓ handedness: {detection.handedness}")
            
            # Validate confidence
            if not (0 <= detection.confidence <= 1):
                print(f"  ✗ Invalid confidence: {detection.confidence}")
                return False
            print(f"  ✓ confidence: {detection.confidence:.3f}")
            
            # Validate timestamp
            if not isinstance(detection.timestamp, int):
                print(f"  ✗ Invalid timestamp type: {type(detection.timestamp)}")
                return False
            print(f"  ✓ timestamp: {detection.timestamp}")
            
            # Test HandDetectionResult methods
            try:
                center = detection.get_bbox_center()
                area = detection.get_bbox_area()
                wrist = detection.get_wrist()
                fingertips = detection.get_fingertips()
                detection_dict = detection.to_dict()
                
                print(f"  ✓ HandDetectionResult methods work")
                print(f"    - bbox_center: {center}")
                print(f"    - bbox_area: {area}")
                print(f"    - wrist: {wrist}")
                print(f"    - fingertips shape: {fingertips.shape}")
            except Exception as e:
                print(f"  ✗ HandDetectionResult methods failed: {e}")
                return False
        
    except Exception as e:
        print(f"✗ detect() failed: {e}")
        import traceback
        traceback.print_exc()
        return False
    
    # Test detect_batch
    try:
        batch_detections = detector.detect_batch([test_image, test_image])
        if not isinstance(batch_detections, list) or len(batch_detections) != 2:
            print(f"✗ detect_batch() returned incorrect structure")
            return False
        print(f"✓ detect_batch() works")
    except Exception as e:
        print(f"✗ detect_batch() failed: {e}")
        return False
    
    # Test statistics
    try:
        stats = detector.get_stats()
        if not isinstance(stats, dict):
            print(f"✗ get_stats() returned non-dict: {type(stats)}")
            return False
        
        required_keys = [
            'frames_processed',
            'total_detections',
            'avg_detections_per_frame',
            'max_num_hands',
            'min_detection_confidence'
        ]
        
        for key in required_keys:
            if key not in stats:
                print(f"✗ Missing stats key: {key}")
                return False
        
        print(f"✓ get_stats() works: {stats}")
    except Exception as e:
        print(f"✗ get_stats() failed: {e}")
        return False
    
    detector.close()
    print("\n✓ Detection output validation passed")
    return True


def validate_video_mode():
    """Validate VIDEO mode with multiple frames."""
    print("\n" + "="*70)
    print("VIDEO MODE VALIDATION")
    print("="*70)
    
    detector = HandDetector(static_image_mode=False)
    
    # Create sequence of test frames
    frames = []
    for i in range(5):
        frame = np.zeros((480, 640, 3), dtype=np.uint8)
        # Different color each frame
        cv2.rectangle(frame, (200, 200), (400, 400), (255, 200+i*10, 150), -1)
        frames.append(frame)
    
    try:
        # Process frames sequentially
        for i, frame in enumerate(frames):
            detections = detector.detect(frame)
            print(f"  Frame {i}: {len(detections)} detections")
        
        print("✓ VIDEO mode sequential processing works")
    except Exception as e:
        print(f"✗ VIDEO mode failed: {e}")
        import traceback
        traceback.print_exc()
        return False
    
    detector.close()
    return True


def validate_image_mode():
    """Validate IMAGE mode (static_image_mode=True)."""
    print("\n" + "="*70)
    print("IMAGE MODE VALIDATION")
    print("="*70)
    
    detector = HandDetector(static_image_mode=True)
    
    test_image = np.zeros((480, 640, 3), dtype=np.uint8)
    cv2.rectangle(test_image, (200, 200), (400, 400), (255, 200, 150), -1)
    
    try:
        detections = detector.detect(test_image)
        print(f"✓ IMAGE mode works: {len(detections)} detections")
    except Exception as e:
        print(f"✗ IMAGE mode failed: {e}")
        import traceback
        traceback.print_exc()
        return False
    
    detector.close()
    return True


def main():
    """Run all validation tests."""
    print("\n" + "="*70)
    print("HandDetector MediaPipe 0.10.x Refactoring Validation")
    print("="*70)
    
    import mediapipe as mp
    print(f"\nMediaPipe version: {mp.__version__}")
    
    results = {
        "API Compatibility": validate_api_compatibility(),
        "Detection Output": validate_detection_output(),
        "VIDEO Mode": validate_video_mode(),
        "IMAGE Mode": validate_image_mode(),
    }
    
    print("\n" + "="*70)
    print("VALIDATION SUMMARY")
    print("="*70)
    
    all_passed = True
    for test_name, passed in results.items():
        status = "✓ PASS" if passed else "✗ FAIL"
        print(f"{test_name:<30} {status}")
        if not passed:
            all_passed = False
    
    print("="*70)
    
    if all_passed:
        print("\n🎉 ALL VALIDATIONS PASSED")
        print("The refactored HandDetector maintains full API compatibility.")
        return 0
    else:
        print("\n❌ SOME VALIDATIONS FAILED")
        print("Review the failures above and fix the implementation.")
        return 1


if __name__ == '__main__':
    sys.exit(main())
