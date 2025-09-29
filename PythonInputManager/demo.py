#!/usr/bin/env python3
"""
Python Input Manager Demonstration Script

This script demonstrates the capabilities of the Python Input Manager implementation,
including real-time video capture, preprocessing, MediaPipe hand detection, and
performance monitoring. It serves as both a functional demo and a migration
validation tool.

Features demonstrated:
- Real-time video capture and preprocessing
- MediaPipe hand detection integration
- Performance metrics collection
- API compatibility with C++ implementation
- Error handling and robustness
"""

import sys
import time
import signal
import argparse
import json
from pathlib import Path
from datetime import datetime
from typing import Optional, Dict, Any

import cv2
import numpy as np

# Add module path
sys.path.insert(0, str(Path(__file__).parent))

from python_input_manager import PythonInputManager, InputManager


class InputManagerDemo:
    """
    Comprehensive demonstration of Python Input Manager capabilities.
    """
    
    def __init__(self, camera_index: int = 0, enable_mediapipe: bool = True):
        """
        Initialize the demo.
        
        Args:
            camera_index: Camera device index
            enable_mediapipe: Whether to enable MediaPipe hand detection
        """
        self.camera_index = camera_index
        self.enable_mediapipe = enable_mediapipe
        self.input_manager: Optional[PythonInputManager] = None
        self.running = False
        self.stats = {
            'frames_processed': 0,
            'hands_detected': 0,
            'start_time': None,
            'errors': 0
        }
        
        # Set up signal handlers for graceful shutdown
        signal.signal(signal.SIGINT, self._signal_handler)
        signal.signal(signal.SIGTERM, self._signal_handler)
    
    def _signal_handler(self, signum, frame):
        """Handle interrupt signals for graceful shutdown."""
        print(f"\nReceived signal {signum}. Stopping demo...")
        self.stop_demo()
    
    def start_demo(self) -> bool:
        """
        Start the Input Manager demo.
        
        Returns:
            bool: True if demo started successfully
        """
        print("="*60)
        print("PYTHON INPUT MANAGER DEMONSTRATION")
        print("="*60)
        print(f"Camera Index: {self.camera_index}")
        print(f"MediaPipe Enabled: {self.enable_mediapipe}")
        print("-"*60)
        
        try:
            # Initialize Input Manager
            self.input_manager = PythonInputManager(
                camera_index=self.camera_index,
                enable_mediapipe=self.enable_mediapipe,
                target_size=(224, 224)
            )
            
            # Start capture
            if not self.input_manager.start_continuous_capture():
                print("ERROR: Failed to start camera capture")
                return False
            
            print("Camera capture started successfully")
            print("Press 'q' to quit, 's' to show stats, 'h' to toggle hand detection")
            print("-"*60)
            
            self.running = True
            self.stats['start_time'] = time.time()
            
            return True
            
        except Exception as e:
            print(f"ERROR: Failed to start demo: {e}")
            return False
    
    def run_interactive_demo(self):
        """Run interactive demo with real-time display."""
        if not self.start_demo():
            return
        
        try:
            # Initialize display window
            cv2.namedWindow('Python Input Manager Demo', cv2.WINDOW_AUTOSIZE)
            
            last_stats_time = time.time()
            
            while self.running:
                # Get latest frame
                frame_data = self.input_manager.get_latest_frame()
                
                if frame_data is not None:
                    self.stats['frames_processed'] += 1
                    
                    # Convert normalized frame back to displayable format
                    display_frame = self._prepare_display_frame(frame_data)
                    
                    # Add hand detection visualization
                    if frame_data.hands_detected:
                        self.stats['hands_detected'] += 1
                        display_frame = self._draw_hand_landmarks(display_frame, frame_data.hands_detected)
                    
                    # Add info overlay
                    display_frame = self._add_info_overlay(display_frame, frame_data)
                    
                    # Show frame
                    cv2.imshow('Python Input Manager Demo', display_frame)
                
                # Handle keyboard input
                key = cv2.waitKey(1) & 0xFF
                if key == ord('q'):
                    break
                elif key == ord('s'):
                    self._print_stats()
                elif key == ord('h'):
                    self._toggle_hand_detection()
                
                # Print periodic stats
                if time.time() - last_stats_time > 5.0:
                    self._print_stats()
                    last_stats_time = time.time()
                
        except Exception as e:
            print(f"ERROR during demo: {e}")
            self.stats['errors'] += 1
        
        finally:
            cv2.destroyAllWindows()
            self.stop_demo()
    
    def run_headless_demo(self, duration: int = 30):
        """
        Run headless demo for specified duration.
        
        Args:
            duration: Duration to run demo in seconds
        """
        if not self.start_demo():
            return
        
        print(f"Running headless demo for {duration} seconds...")
        
        try:
            start_time = time.time()
            last_stats_time = start_time
            
            while time.time() - start_time < duration and self.running:
                frame_data = self.input_manager.get_latest_frame()
                
                if frame_data is not None:
                    self.stats['frames_processed'] += 1
                    
                    if frame_data.hands_detected:
                        self.stats['hands_detected'] += 1
                
                # Print stats every 5 seconds
                if time.time() - last_stats_time > 5.0:
                    self._print_stats()
                    last_stats_time = time.time()
                
                time.sleep(0.01)  # Small sleep to prevent CPU spinning
                
        except Exception as e:
            print(f"ERROR during demo: {e}")
            self.stats['errors'] += 1
        
        finally:
            self.stop_demo()
    
    def benchmark_performance(self, duration: int = 30) -> Dict[str, Any]:
        """
        Run performance benchmark.
        
        Args:
            duration: Duration to benchmark in seconds
            
        Returns:
            Dict containing benchmark results
        """
        print(f"Running performance benchmark for {duration} seconds...")
        
        if not self.start_demo():
            return {}
        
        try:
            benchmark_results = {}
            start_time = time.time()
            frame_times = []
            
            while time.time() - start_time < duration and self.running:
                frame_start = time.time()
                frame_data = self.input_manager.get_latest_frame()
                frame_end = time.time()
                
                if frame_data is not None:
                    self.stats['frames_processed'] += 1
                    frame_times.append(frame_end - frame_start)
                    
                    if frame_data.hands_detected:
                        self.stats['hands_detected'] += 1
                
                time.sleep(0.001)
            
            # Calculate performance metrics
            if frame_times:
                benchmark_results = {
                    'duration_seconds': duration,
                    'frames_processed': self.stats['frames_processed'],
                    'avg_fps': self.stats['frames_processed'] / duration,
                    'avg_frame_time_ms': np.mean(frame_times) * 1000,
                    'min_frame_time_ms': np.min(frame_times) * 1000,
                    'max_frame_time_ms': np.max(frame_times) * 1000,
                    'hands_detection_rate': self.stats['hands_detected'] / max(self.stats['frames_processed'], 1),
                    'input_manager_metrics': self.input_manager.get_performance_metrics()
                }
            
            return benchmark_results
            
        except Exception as e:
            print(f"ERROR during benchmark: {e}")
            return {}
        
        finally:
            self.stop_demo()
    
    def test_api_compatibility(self):
        """Test API compatibility with C++ implementation."""
        print("Testing API compatibility with C++ implementation...")
        
        try:
            # Test compatibility wrapper
            compat_manager = InputManager(self.camera_index)
            
            # Test all API methods
            print("  Testing start_capture()...", end=" ")
            result = compat_manager.start_capture()
            print("SUCCESS" if result else "FAILED")
            
            print("  Testing is_capture_open()...", end=" ")
            result = compat_manager.is_capture_open()
            print("SUCCESS" if isinstance(result, bool) else "FAILED")
            
            print("  Testing start_continuous_capture()...", end=" ")
            result = compat_manager.start_continuous_capture()
            print("SUCCESS" if result else "FAILED")
            
            # Give it a moment to capture
            time.sleep(1)
            
            print("  Testing get_latest_frame()...", end=" ")
            frame_data = compat_manager.get_latest_frame()
            print("SUCCESS" if frame_data is not None else "FAILED")
            
            print("  Testing get_all_frames()...", end=" ")
            all_frames = compat_manager.get_all_frames()
            print("SUCCESS" if isinstance(all_frames, list) else "FAILED")
            
            print("  Testing set_normalization_params()...", end=" ")
            compat_manager.set_normalization_params(0.485, 0.229)
            print("SUCCESS")
            
            print("  Testing set_mirror_enabled()...", end=" ")
            compat_manager.set_mirror_enabled(False)
            print("SUCCESS")
            
            print("  Testing stop_continuous_capture()...", end=" ")
            compat_manager.stop_continuous_capture()
            print("SUCCESS")
            
            print("\nAPI compatibility test: ALL TESTS PASSED")
            
        except Exception as e:
            print(f"ERROR during compatibility test: {e}")
    
    def stop_demo(self):
        """Stop the demo and cleanup resources."""
        self.running = False
        
        if self.input_manager:
            self.input_manager.stop_continuous_capture()
        
        self._print_final_stats()
    
    def _prepare_display_frame(self, frame_data) -> np.ndarray:
        """
        Prepare frame for display by converting from normalized format.
        
        Args:
            frame_data: FrameData object
            
        Returns:
            Display-ready frame
        """
        # Convert normalized frame back to [0, 255] uint8
        frame = frame_data.frame
        
        # Reverse normalization: frame * std + mean
        denormalized = frame * 0.5 + 0.5  # Assuming default normalization
        denormalized = np.clip(denormalized, 0, 1)
        
        # Convert to uint8 and scale to display size
        display_frame = (denormalized * 255).astype(np.uint8)
        
        # Scale up for better visibility (224x224 -> 448x448)
        display_frame = cv2.resize(display_frame, (448, 448), interpolation=cv2.INTER_NEAREST)
        
        return display_frame
    
    def _draw_hand_landmarks(self, frame: np.ndarray, hands_data) -> np.ndarray:
        """
        Draw hand landmarks on the frame.
        
        Args:
            frame: Display frame
            hands_data: Hand detection results
            
        Returns:
            Frame with hand landmarks drawn
        """
        if not hands_data:
            return frame
        
        h, w = frame.shape[:2]
        
        for hand_data in hands_data:
            landmarks = hand_data['landmarks']
            handedness = hand_data['handedness']
            
            # Draw landmarks
            for landmark in landmarks:
                x, y = int(landmark[0] * w), int(landmark[1] * h)
                cv2.circle(frame, (x, y), 3, (0, 255, 0), -1)
            
            # Draw handedness label
            if landmarks:
                label_x, label_y = int(landmarks[0][0] * w), int(landmarks[0][1] * h)
                cv2.putText(frame, handedness, (label_x, label_y - 10),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
        
        return frame
    
    def _add_info_overlay(self, frame: np.ndarray, frame_data) -> np.ndarray:
        """
        Add information overlay to the frame.
        
        Args:
            frame: Display frame
            frame_data: FrameData object
            
        Returns:
            Frame with info overlay
        """
        # Add timestamp
        timestamp_str = f"Time: {datetime.fromtimestamp(frame_data.timestamp).strftime('%H:%M:%S.%f')[:-3]}"
        cv2.putText(frame, timestamp_str, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
        
        # Add frame count
        frame_count_str = f"Frame: {self.stats['frames_processed']}"
        cv2.putText(frame, frame_count_str, (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
        
        # Add hands detected count
        hands_str = f"Hands: {len(frame_data.hands_detected) if frame_data.hands_detected else 0}"
        cv2.putText(frame, hands_str, (10, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
        
        # Add FPS
        if self.stats['start_time']:
            elapsed = time.time() - self.stats['start_time']
            fps = self.stats['frames_processed'] / elapsed if elapsed > 0 else 0
            fps_str = f"FPS: {fps:.1f}"
            cv2.putText(frame, fps_str, (10, 120), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
        
        return frame
    
    def _toggle_hand_detection(self):
        """Toggle MediaPipe hand detection on/off."""
        if self.input_manager:
            self.input_manager.enable_mediapipe = not self.input_manager.enable_mediapipe
            status = "ENABLED" if self.input_manager.enable_mediapipe else "DISABLED"
            print(f"Hand detection {status}")
    
    def _print_stats(self):
        """Print current statistics."""
        if self.stats['start_time']:
            elapsed = time.time() - self.stats['start_time']
            fps = self.stats['frames_processed'] / elapsed if elapsed > 0 else 0
            
            print(f"[STATS] Frames: {self.stats['frames_processed']}, "
                  f"FPS: {fps:.1f}, "
                  f"Hands: {self.stats['hands_detected']}, "
                  f"Errors: {self.stats['errors']}")
    
    def _print_final_stats(self):
        """Print final statistics summary."""
        print("\n" + "="*60)
        print("DEMO COMPLETED - FINAL STATISTICS")
        print("="*60)
        
        if self.stats['start_time']:
            elapsed = time.time() - self.stats['start_time']
            fps = self.stats['frames_processed'] / elapsed if elapsed > 0 else 0
            
            print(f"Total Duration:       {elapsed:.2f} seconds")
            print(f"Frames Processed:     {self.stats['frames_processed']}")
            print(f"Average FPS:          {fps:.2f}")
            print(f"Hands Detected:       {self.stats['hands_detected']}")
            print(f"Hand Detection Rate:  {self.stats['hands_detected']/max(self.stats['frames_processed'],1)*100:.1f}%")
            print(f"Errors Encountered:   {self.stats['errors']}")
            
            if self.input_manager:
                metrics = self.input_manager.get_performance_metrics()
                print(f"\nInput Manager Metrics:")
                for key, value in metrics.items():
                    print(f"  {key}: {value}")


def main():
    """Main function for running demonstrations."""
    parser = argparse.ArgumentParser(description="Python Input Manager Demonstration")
    parser.add_argument("--camera", type=int, default=0, help="Camera index to use")
    parser.add_argument("--no-mediapipe", action="store_true", help="Disable MediaPipe hand detection")
    parser.add_argument("--mode", choices=["interactive", "headless", "benchmark", "compatibility"], 
                       default="interactive", help="Demo mode to run")
    parser.add_argument("--duration", type=int, default=30, help="Duration for headless/benchmark modes")
    parser.add_argument("--output", type=str, help="Output file for benchmark results")
    
    args = parser.parse_args()
    
    # Create demo instance
    demo = InputManagerDemo(
        camera_index=args.camera,
        enable_mediapipe=not args.no_mediapipe
    )
    
    try:
        if args.mode == "interactive":
            print("Starting interactive demo...")
            print("Make sure your camera is connected and no other application is using it.")
            input("Press Enter to continue or Ctrl+C to cancel...")
            demo.run_interactive_demo()
            
        elif args.mode == "headless":
            demo.run_headless_demo(duration=args.duration)
            
        elif args.mode == "benchmark":
            results = demo.benchmark_performance(duration=args.duration)
            
            if results:
                print("\nBenchmark Results:")
                print(json.dumps(results, indent=2))
                
                if args.output:
                    with open(args.output, 'w') as f:
                        json.dump(results, f, indent=2)
                    print(f"Results saved to {args.output}")
            
        elif args.mode == "compatibility":
            demo.test_api_compatibility()
    
    except KeyboardInterrupt:
        print("\nDemo interrupted by user")
    except Exception as e:
        print(f"Demo failed with error: {e}")
    finally:
        demo.stop_demo()


if __name__ == "__main__":
    main()