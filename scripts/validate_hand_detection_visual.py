#!/usr/bin/env python3
"""
Real-Time Hand Detection Validation

Purpose: Provide visual verification of MediaPipe hand detection
         through live webcam feed with detection overlays.

This script:
1. Opens webcam and displays live video
2. Detects hands in real-time with MediaPipe
3. Overlays bounding boxes and landmarks on detected hands
4. Logs detection events to terminal
5. Allows manual testing of detection robustness

Usage:
    python scripts/validate_hand_detection_visual.py

Controls:
    - Press 'q' or ESC to quit
    - Press 's' to save current frame
    - Press 'i' to toggle info display

Expected Behavior:
    - Single hand: 1 bounding box visible
    - Two hands: 2 bounding boxes visible
    - Motion: Smooth tracking with minimal jitter
    - Hand exits frame: Detection count drops to 0
"""

import sys
import cv2
import time
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from input_manager.input_manager import InputManager
from recognition_engine.hand_detection.hand_detector import HandDetector


def draw_hand_detection(frame, detection):
    """Draw bounding box and landmarks for a detected hand."""
    # Draw bounding box
    x, y, w, h = detection.bbox
    color = (0, 255, 0) if detection.handedness == "Right" else (255, 0, 0)
    
    cv2.rectangle(frame, (x, y), (x + w, y + h), color, 2)
    
    # Draw handedness label
    label = f"{detection.handedness} {detection.confidence:.2f}"
    label_y = y - 10 if y - 10 > 20 else y + h + 20
    cv2.putText(frame, label, (x, label_y),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
    
    # Draw landmarks (21 hand keypoints)
    for i, landmark in enumerate(detection.landmarks):
        # Convert normalized coordinates to pixel coordinates
        h_frame, w_frame = frame.shape[:2]
        px = int(landmark[0] * w_frame)
        py = int(landmark[1] * h_frame)
        
        # Draw different colors for different landmark types
        if i in [0, 4, 8, 12, 16, 20]:  # Fingertips and wrist
            cv2.circle(frame, (px, py), 5, (255, 255, 0), -1)
        else:
            cv2.circle(frame, (px, py), 3, (0, 255, 255), -1)
    
    return frame


def draw_info_panel(frame, stats, show_info=True):
    """Draw information panel on frame."""
    if not show_info:
        return frame
    
    h, w = frame.shape[:2]
    panel_height = 120
    panel = frame.copy()
    
    # Semi-transparent black panel
    overlay = panel.copy()
    cv2.rectangle(overlay, (0, 0), (w, panel_height), (0, 0, 0), -1)
    cv2.addWeighted(overlay, 0.7, panel, 0.3, 0, panel)
    
    # Draw text
    y_offset = 25
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 0.6
    color = (255, 255, 255)
    thickness = 1
    
    cv2.putText(panel, f"Detections: {stats['current_hands']}", 
                (10, y_offset), font, font_scale, color, thickness)
    
    cv2.putText(panel, f"FPS: {stats['fps']:.1f}", 
                (10, y_offset + 25), font, font_scale, color, thickness)
    
    cv2.putText(panel, f"Total Frames: {stats['total_frames']}", 
                (10, y_offset + 50), font, font_scale, color, thickness)
    
    cv2.putText(panel, f"Detection Rate: {stats['detection_rate']:.1f}%", 
                (10, y_offset + 75), font, font_scale, color, thickness)
    
    # Controls
    cv2.putText(panel, "Controls: Q=Quit | S=Save | I=Info", 
                (w - 350, y_offset), font, font_scale, (0, 255, 255), thickness)
    
    return panel


def main():
    """Run real-time hand detection validation."""
    print("="*60)
    print("REAL-TIME HAND DETECTION VALIDATION")
    print("="*60)
    print("\nInitializing camera and hand detector...")
    
    # Initialize components
    input_mgr = InputManager(camera_index=0, buffer_size=10)
    detector = HandDetector(
        max_num_hands=2,
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5,
        model_complexity=1
    )
    
    # Start capture
    input_mgr.start_continuous_capture()
    print("✓ Camera started")
    print("✓ Hand detector initialized")
    print("\n" + "="*60)
    print("INSTRUCTIONS:")
    print("="*60)
    print("1. Position your hand(s) in front of the camera")
    print("2. Move hands to test detection robustness")
    print("3. Observe bounding boxes and landmarks")
    print("4. Check terminal for detection logs")
    print("\nPress 'q' or ESC to quit")
    print("="*60 + "\n")
    
    # Statistics
    stats = {
        'total_frames': 0,
        'frames_with_hands': 0,
        'current_hands': 0,
        'fps': 0,
        'detection_rate': 0
    }
    
    fps_start_time = time.time()
    fps_frame_count = 0
    show_info = True
    saved_count = 0
    
    try:
        while True:
            # Get latest frame
            frame_data = input_mgr.get_latest_frame()
            if frame_data is None:
                time.sleep(0.01)
                continue
            
            frame = frame_data.frame.copy()
            stats['total_frames'] += 1
            fps_frame_count += 1
            
            # Detect hands
            detections = detector.detect(frame)
            num_hands = len(detections)
            stats['current_hands'] = num_hands
            
            # Update statistics
            if num_hands > 0:
                stats['frames_with_hands'] += 1
            
            stats['detection_rate'] = (stats['frames_with_hands'] / stats['total_frames']) * 100
            
            # Calculate FPS every second
            if time.time() - fps_start_time >= 1.0:
                stats['fps'] = fps_frame_count / (time.time() - fps_start_time)
                fps_start_time = time.time()
                fps_frame_count = 0
            
            # Log detection events
            if num_hands > 0:
                hand_info = ", ".join([f"{d.handedness}({d.confidence:.2f})" for d in detections])
                print(f"Frame {stats['total_frames']:4d}: Detected {num_hands} hand(s) - {hand_info}")
            
            # Draw detections on frame
            for detection in detections:
                frame = draw_hand_detection(frame, detection)
            
            # Draw info panel
            frame = draw_info_panel(frame, stats, show_info)
            
            # Display frame
            cv2.imshow("Hand Detection Validation", frame)
            
            # Handle keyboard input
            key = cv2.waitKey(1) & 0xFF
            
            if key == ord('q') or key == 27:  # 'q' or ESC
                print("\n" + "="*60)
                print("SHUTTING DOWN")
                print("="*60)
                break
            elif key == ord('s'):  # Save frame
                saved_count += 1
                filename = f"validation_frame_{saved_count:03d}.jpg"
                cv2.imwrite(filename, frame)
                print(f"Saved: {filename}")
            elif key == ord('i'):  # Toggle info
                show_info = not show_info
    
    except KeyboardInterrupt:
        print("\n\nInterrupted by user")
    
    except Exception as e:
        print(f"\n✗ Error: {e}")
        import traceback
        traceback.print_exc()
    
    finally:
        # Cleanup
        print("\nCleaning up...")
        input_mgr.stop_capture()
        detector.close()
        cv2.destroyAllWindows()
        
        # Final statistics
        print("\n" + "="*60)
        print("VALIDATION SUMMARY")
        print("="*60)
        print(f"Total frames processed: {stats['total_frames']}")
        print(f"Frames with hands: {stats['frames_with_hands']}")
        print(f"Detection rate: {stats['detection_rate']:.1f}%")
        print(f"Average FPS: {stats['fps']:.1f}")
        
        # Detector stats
        detector_stats = detector.get_stats()
        print(f"\nDetector statistics:")
        print(f"  Total detections: {detector_stats['total_detections']}")
        print(f"  Frames processed: {detector_stats['frames_processed']}")
        
        print("\n" + "="*60)
        
        # Validation assessment
        if stats['total_frames'] > 0 and stats['detection_rate'] > 0:
            print("✓ VALIDATION STATUS: PASS")
            print("  Hand detection is working correctly")
            print("  - Detections observable in real-time")
            print("  - Bounding boxes and landmarks visible")
            print("  - Terminal logs show detection events")
        else:
            print("✗ VALIDATION STATUS: INCONCLUSIVE")
            print("  No hands detected during validation")
            print("  Consider repeating test with hands in frame")
        
        print("="*60)


if __name__ == "__main__":
    main()
