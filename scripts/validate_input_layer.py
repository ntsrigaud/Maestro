#!/usr/bin/env python3
"""
Input Layer Validation Script

Purpose: Validate MediaPipe hand detection and cropping pipeline
         to ensure input quality before CLIP integration.

This script tests:
1. Camera initialization and frame capture
2. Hand detection reliability
3. Cropping quality and consistency
4. Failure mode handling (no hands, multiple hands, edge cases)

Usage:
    python scripts/validate_input_layer.py [--duration SECONDS] [--save-artifacts]
"""

import sys
import cv2
import numpy as np
import time
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from input_manager.input_manager import InputManager
from recognition_engine.hand_detection.hand_detector import HandDetector
from recognition_engine.hand_detection.hand_cropper import HandCropper


class InputLayerValidator:
    """Validates the input layer (camera + hand detection + cropping)."""

    def __init__(self, save_artifacts: bool = False):
        self.save_artifacts = save_artifacts
        self.artifacts_dir = Path("validation_artifacts")
        
        if self.save_artifacts:
            self.artifacts_dir.mkdir(exist_ok=True)
            timestamp = time.strftime("%Y%m%d_%H%M%S")
            self.session_dir = self.artifacts_dir / f"session_{timestamp}"
            self.session_dir.mkdir(exist_ok=True)
            print(f"Saving artifacts to: {self.session_dir}")

        # Statistics tracking
        self.stats = {
            "total_frames": 0,
            "frames_with_hands": 0,
            "frames_no_hands": 0,
            "frames_multiple_hands": 0,
            "total_detections": 0,
            "detection_confidences": [],
            "crop_sizes": [],
            "processing_times": [],
        }

    def validate_camera_capture(self) -> bool:
        """Validate camera initialization and frame capture."""
        print("\n=== Testing Camera Capture ===")
        
        try:
            input_mgr = InputManager(camera_index=0, buffer_size=10)
            input_mgr.start_continuous_capture()
            
            print("✓ Camera initialized")
            
            # Capture 10 frames
            frames_captured = 0
            for i in range(10):
                frame_data = input_mgr.get_latest_frame()
                if frame_data is not None:
                    frames_captured += 1
                    print(f"  Frame {i+1}: {frame_data.frame.shape}, timestamp: {frame_data.timestamp_ms():.2f}ms")
                time.sleep(0.1)
            
            input_mgr.stop_capture()
            
            if frames_captured == 10:
                print(f"✓ Successfully captured {frames_captured}/10 frames")
                return True
            else:
                print(f"✗ Only captured {frames_captured}/10 frames")
                return False
                
        except Exception as e:
            print(f"✗ Camera capture failed: {e}")
            return False

    def validate_hand_detection(self, duration_seconds: int = 10) -> bool:
        """Validate hand detection over a duration."""
        print(f"\n=== Testing Hand Detection ({duration_seconds}s) ===")
        print("Hold your hand in front of the camera...")
        
        try:
            input_mgr = InputManager(camera_index=0, buffer_size=10)
            detector = HandDetector(
                max_num_hands=2,
                min_detection_confidence=0.5,
                min_tracking_confidence=0.5
            )
            
            input_mgr.start_continuous_capture()
            
            start_time = time.time()
            frame_count = 0
            
            while time.time() - start_time < duration_seconds:
                frame_data = input_mgr.get_latest_frame()
                if frame_data is None:
                    time.sleep(0.033)  # ~30fps
                    continue
                
                frame_count += 1
                self.stats["total_frames"] += 1
                
                # Detect hands
                t0 = time.perf_counter()
                detections = detector.detect(frame_data.frame)
                detection_time = (time.perf_counter() - t0) * 1000  # ms
                
                self.stats["processing_times"].append(detection_time)
                
                num_hands = len(detections)
                self.stats["total_detections"] += num_hands
                
                if num_hands == 0:
                    self.stats["frames_no_hands"] += 1
                elif num_hands == 1:
                    self.stats["frames_with_hands"] += 1
                    self.stats["detection_confidences"].append(detections[0].confidence)
                else:
                    self.stats["frames_multiple_hands"] += 1
                    for det in detections:
                        self.stats["detection_confidences"].append(det.confidence)
                
                # Display progress
                if frame_count % 10 == 0:
                    elapsed = time.time() - start_time
                    hands_pct = (self.stats["frames_with_hands"] / self.stats["total_frames"]) * 100
                    print(f"  {elapsed:.1f}s: {self.stats['total_frames']} frames, "
                          f"{self.stats['frames_with_hands']} with hands ({hands_pct:.1f}%)")
            
            input_mgr.stop_capture()
            
            # Print summary
            print("\n--- Detection Summary ---")
            print(f"Total frames: {self.stats['total_frames']}")
            print(f"Frames with hands: {self.stats['frames_with_hands']} "
                  f"({self.stats['frames_with_hands']/self.stats['total_frames']*100:.1f}%)")
            print(f"Frames no hands: {self.stats['frames_no_hands']} "
                  f"({self.stats['frames_no_hands']/self.stats['total_frames']*100:.1f}%)")
            print(f"Frames multiple hands: {self.stats['frames_multiple_hands']}")
            
            if self.stats["detection_confidences"]:
                avg_conf = np.mean(self.stats["detection_confidences"])
                min_conf = np.min(self.stats["detection_confidences"])
                max_conf = np.max(self.stats["detection_confidences"])
                print(f"Detection confidence: avg={avg_conf:.3f}, min={min_conf:.3f}, max={max_conf:.3f}")
            
            avg_time = np.mean(self.stats["processing_times"])
            print(f"Processing time: avg={avg_time:.1f}ms, max={np.max(self.stats['processing_times']):.1f}ms")
            
            # Success criteria
            detection_rate = self.stats['frames_with_hands'] / self.stats['total_frames']
            if detection_rate > 0.3:  # At least 30% detection rate (user may not always show hand)
                print(f"✓ Hand detection working (detection rate: {detection_rate*100:.1f}%)")
                return True
            else:
                print(f"✗ Low detection rate: {detection_rate*100:.1f}%")
                return False
            
        except Exception as e:
            print(f"✗ Hand detection failed: {e}")
            import traceback
            traceback.print_exc()
            return False

    def validate_cropping(self, num_samples: int = 20) -> bool:
        """Validate hand cropping quality."""
        print(f"\n=== Testing Hand Cropping ({num_samples} samples) ===")
        print("Hold your hand steady in front of camera...")
        
        try:
            input_mgr = InputManager(camera_index=0, buffer_size=10)
            detector = HandDetector(max_num_hands=1, min_detection_confidence=0.6)
            cropper = HandCropper(target_size=(224, 224))
            
            input_mgr.start_continuous_capture()
            
            samples_collected = 0
            attempts = 0
            max_attempts = num_samples * 3
            
            while samples_collected < num_samples and attempts < max_attempts:
                attempts += 1
                frame_data = input_mgr.get_latest_frame()
                if frame_data is None:
                    time.sleep(0.033)
                    continue
                
                detections = detector.detect(frame_data.frame)
                if len(detections) != 1:
                    continue  # Only process single-hand frames
                
                detection = detections[0]
                cropped_hands = cropper.crop_hands(frame_data.frame, [detection])
                
                if len(cropped_hands) == 1:
                    cropped = cropped_hands[0]
                    samples_collected += 1
                    
                    # Record crop size
                    self.stats["crop_sizes"].append(cropped.cropped_image.shape[:2])
                    
                    # Save artifact if requested
                    if self.save_artifacts and samples_collected <= 10:
                        crop_path = self.session_dir / f"crop_sample_{samples_collected:03d}.jpg"
                        cv2.imwrite(str(crop_path), cropped.cropped_image)
                    
                    if samples_collected % 5 == 0:
                        print(f"  Collected {samples_collected}/{num_samples} samples")
            
            input_mgr.stop_capture()
            
            # Analyze crop quality
            print("\n--- Cropping Summary ---")
            print(f"Samples collected: {samples_collected}/{num_samples}")
            
            if samples_collected > 0:
                sizes = np.array(self.stats["crop_sizes"])
                print(f"Crop sizes: {sizes[0]} (target: (224, 224))")
                
                # All crops should be (224, 224)
                all_correct_size = np.all(sizes == [224, 224])
                if all_correct_size:
                    print("✓ All crops have correct target size (224x224)")
                else:
                    print("✗ Some crops have incorrect size")
                    return False
                
                if self.save_artifacts:
                    print(f"✓ Sample crops saved to: {self.session_dir}")
                
                return True
            else:
                print("✗ Failed to collect crop samples")
                return False
                
        except Exception as e:
            print(f"✗ Cropping validation failed: {e}")
            import traceback
            traceback.print_exc()
            return False

    def test_failure_modes(self) -> bool:
        """Test handling of edge cases and failure modes."""
        print("\n=== Testing Failure Modes ===")
        
        try:
            detector = HandDetector(max_num_hands=2, min_detection_confidence=0.5)
            
            # Test 1: Empty frame
            print("\nTest 1: Empty frame (all black)")
            empty_frame = np.zeros((480, 640, 3), dtype=np.uint8)
            detections = detector.detect(empty_frame)
            if len(detections) == 0:
                print("✓ Empty frame: 0 detections (expected)")
            else:
                print(f"✗ Empty frame: {len(detections)} detections (unexpected)")
                return False
            
            # Test 2: Noise frame
            print("\nTest 2: Random noise frame")
            noise_frame = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
            detections = detector.detect(noise_frame)
            print(f"  Noise frame: {len(detections)} detections")
            # Noise may or may not trigger false positives - just log it
            
            # Test 3: Invalid input
            print("\nTest 3: Invalid input handling")
            detections = detector.detect(None)
            if len(detections) == 0:
                print(f"✓ None input: Returns empty list (graceful handling)")
            else:
                print(f"✗ None input: Unexpected detections ({len(detections)})")
                return False
            
            # Test 4: Wrong shape
            print("\nTest 4: Wrong shape input")
            try:
                wrong_shape = np.zeros((100, 100), dtype=np.uint8)  # Grayscale
                detections = detector.detect(wrong_shape)
                print(f"  Grayscale input: {len(detections)} detections")
                # MediaPipe may handle grayscale - just log it
            except Exception as e:
                print(f"  Grayscale input: Error ({type(e).__name__})")
            
            print("\n✓ Failure mode tests complete")
            return True
            
        except Exception as e:
            print(f"✗ Failure mode testing failed: {e}")
            import traceback
            traceback.print_exc()
            return False

    def generate_report(self) -> str:
        """Generate validation report."""
        report = []
        report.append("\n" + "="*60)
        report.append("INPUT LAYER VALIDATION REPORT")
        report.append("="*60)
        
        if self.stats["total_frames"] > 0:
            detection_rate = self.stats["frames_with_hands"] / self.stats["total_frames"]
            report.append(f"\nDetection Statistics:")
            report.append(f"  Total frames processed: {self.stats['total_frames']}")
            report.append(f"  Frames with hands: {self.stats['frames_with_hands']} ({detection_rate*100:.1f}%)")
            if self.stats["detection_confidences"]:
                report.append(
                    f"  Average detection confidence: {np.mean(self.stats['detection_confidences']):.3f}"
                )
            report.append(f"  Average processing time: {np.mean(self.stats['processing_times']):.1f}ms")
        
        if self.stats["crop_sizes"]:
            report.append(f"\nCropping Statistics:")
            report.append(f"  Samples collected: {len(self.stats['crop_sizes'])}")
            report.append(f"  Target size: (224, 224)")
            report.append(f"  All crops correct size: {np.all(np.array(self.stats['crop_sizes']) == [224, 224])}")
        
        if self.save_artifacts:
            report.append(f"\nArtifacts saved to: {self.session_dir}")
        
        report.append("\n" + "="*60)
        
        return "\n".join(report)


def main():
    import argparse
    
    parser = argparse.ArgumentParser(description="Validate MediaPipe input layer")
    parser.add_argument("--duration", type=int, default=10,
                        help="Duration for hand detection test (seconds)")
    parser.add_argument("--save-artifacts", action="store_true",
                        help="Save sample crops and visualizations")
    parser.add_argument("--skip-camera", action="store_true",
                        help="Skip camera test (for headless environments)")
    
    args = parser.parse_args()
    
    validator = InputLayerValidator(save_artifacts=args.save_artifacts)
    
    print("="*60)
    print("INPUT LAYER VALIDATION")
    print("="*60)
    print("\nThis script validates:")
    print("1. Camera capture")
    print("2. Hand detection reliability")
    print("3. Cropping quality")
    print("4. Failure mode handling")
    print("\n" + "="*60)
    
    results = {}
    
    # Test 1: Camera capture
    if not args.skip_camera:
        results["camera"] = validator.validate_camera_capture()
    else:
        print("\n=== Skipping Camera Test ===")
        results["camera"] = None
    
    # Test 2: Hand detection
    if not args.skip_camera and results["camera"]:
        results["detection"] = validator.validate_hand_detection(args.duration)
    else:
        results["detection"] = None
    
    # Test 3: Cropping
    if not args.skip_camera and results["camera"] and results["detection"]:
        results["cropping"] = validator.validate_cropping(num_samples=20)
    else:
        results["cropping"] = None
    
    # Test 4: Failure modes
    results["failure_modes"] = validator.test_failure_modes()
    
    # Generate report
    print(validator.generate_report())
    
    # Final verdict
    print("\n" + "="*60)
    print("VALIDATION RESULTS")
    print("="*60)
    for test_name, result in results.items():
        if result is None:
            status = "SKIPPED"
        elif result:
            status = "✓ PASS"
        else:
            status = "✗ FAIL"
        print(f"{test_name:20s}: {status}")
    
    # Overall readiness
    all_pass = all(r in [True, None] for r in results.values())
    any_fail = any(r is False for r in results.values())
    
    print("\n" + "="*60)
    if any_fail:
        print("⚠️  INPUT LAYER VALIDATION: FAILED")
        print("Not ready for CLIP integration - fix issues above")
        return 1
    elif all_pass and results["camera"] is not None:
        print("✓ INPUT LAYER VALIDATION: PASSED")
        print("Ready for Phase 1 CLIP integration")
        return 0
    else:
        print("⚠️  INPUT LAYER VALIDATION: INCOMPLETE")
        print("Some tests were skipped - run full validation before CLIP integration")
        return 0


if __name__ == "__main__":
    sys.exit(main())
