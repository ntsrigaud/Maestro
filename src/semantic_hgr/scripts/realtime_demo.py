#!/usr/bin/env python3
"""
Real-Time Semantic Gesture Recognition Demo

Purpose: Production-representative demonstration of end-to-end semantic
         gesture understanding pipeline with performance monitoring.

System Components:
- Camera input capture
- Hand detection (MediaPipe)
- CLIP gesture encoding
- Semantic intent extraction
- Real-time visualization
- Performance metrics logging

Features:
- Top-3 intent display with confidence bars
- Per-frame latency measurement
- Confidence threshold rejection
- Gesture command mapping
- Robustness testing modes

Usage:
    python scripts/realtime_demo.py \
        --gesture-set configs/demo_gestures.json \
        --confidence-threshold 0.7 \
        --output logs/demo_session.json

Controls:
    Q / ESC   - Quit demo
    S         - Save current frame
    P         - Pause/unpause
    T         - Toggle performance overlay
    R         - Start robustness test
    H         - Show help
"""

import json
import time
import argparse
import cv2
import numpy as np
from pathlib import Path
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass, field, asdict
from collections import deque
import warnings

from semantic_hgr.input_manager.input_manager import InputManager
from semantic_hgr.recognition_engine.hand_detection.hand_detector import HandDetector
from semantic_hgr.representation import GestureEncoder
from semantic_hgr.semantics.clip_intent import SemanticIntentExtractor


@dataclass
class DemoConfig:
    """Configuration for demo session."""
    gesture_definitions: Dict[str, str]  # intent -> command mapping
    confidence_threshold: float = 0.7
    top_k: int = 3
    camera_index: int = 0
    frame_width: int = 1280
    frame_height: int = 720
    
    def to_dict(self):
        return asdict(self)


@dataclass
class PerformanceMetrics:
    """Performance metrics for single frame."""
    frame_id: int
    timestamp: float
    hand_detection_ms: float
    encoding_ms: float
    intent_extraction_ms: float
    total_latency_ms: float
    num_hands_detected: int
    top_intent: Optional[str] = None
    top_confidence: Optional[float] = None
    
    def to_dict(self):
        return asdict(self)


@dataclass
class DemoSession:
    """Complete demo session data."""
    config: DemoConfig
    start_time: float
    end_time: Optional[float] = None
    frames_processed: int = 0
    successful_predictions: int = 0
    rejected_predictions: int = 0
    performance_metrics: List[PerformanceMetrics] = field(default_factory=list)
    
    def to_dict(self):
        return {
            "config": self.config.to_dict(),
            "start_time": self.start_time,
            "end_time": self.end_time,
            "frames_processed": self.frames_processed,
            "successful_predictions": self.successful_predictions,
            "rejected_predictions": self.rejected_predictions,
            "performance_summary": self.compute_summary(),
            "performance_metrics": [m.to_dict() for m in self.performance_metrics]
        }
    
    def compute_summary(self) -> Dict:
        """Compute summary statistics."""
        if not self.performance_metrics:
            return {}
        
        latencies = [m.total_latency_ms for m in self.performance_metrics]
        
        return {
            "mean_latency_ms": float(np.mean(latencies)),
            "std_latency_ms": float(np.std(latencies)),
            "min_latency_ms": float(np.min(latencies)),
            "max_latency_ms": float(np.max(latencies)),
            "p50_latency_ms": float(np.percentile(latencies, 50)),
            "p95_latency_ms": float(np.percentile(latencies, 95)),
            "p99_latency_ms": float(np.percentile(latencies, 99)),
            "success_rate": self.successful_predictions / max(self.frames_processed, 1)
        }


class RealtimeGestureDemo:
    """Real-time semantic gesture recognition demonstration."""
    
    def __init__(self, config: DemoConfig):
        """
        Initialize demo components.
        
        Args:
            config: Demo configuration
        """
        self.config = config
        self.session = DemoSession(config=config, start_time=time.time())
        
        print("Initializing demo components...")
        
        # Initialize pipeline components
        self.input_mgr = InputManager(
            camera_index=config.camera_index,
            buffer_size=10
        )
        
        self.hand_detector = HandDetector(
            max_num_hands=1,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5
        )
        
        self.encoder = GestureEncoder(
            model_name="ViT-B-32",
            pretrained="openai",
            device="cpu"  # Change to "cuda" for GPU
        )
        
        self.intent_extractor = SemanticIntentExtractor(
            model_name="ViT-B-32",
            pretrained="openai",
            device="cpu"
        )
        
        # State
        self.paused = False
        self.show_performance = True
        self.robustness_mode = None
        
        # Performance tracking
        self.fps_window = deque(maxlen=30)
        self.last_fps_time = time.time()
        
        print("✓ Components initialized")
        print(f"  Available intents: {len(self.intent_extractor.intent_names)}")
        print(f"  Gesture mappings: {len(config.gesture_definitions)}")
    
    def process_frame(self, frame: np.ndarray, frame_id: int) -> Tuple[np.ndarray, PerformanceMetrics]:
        """
        Process single frame through full pipeline.
        
        Args:
            frame: Input frame (BGR)
            frame_id: Frame identifier
            
        Returns:
            (annotated_frame, performance_metrics)
        """
        timestamp = time.time()
        annotated_frame = frame.copy()
        
        # 1. Hand detection
        t0 = time.perf_counter()
        detections = self.hand_detector.detect(frame)
        hand_detection_ms = (time.perf_counter() - t0) * 1000
        
        num_hands = len(detections)
        
        if num_hands == 0:
            # No hand detected
            metrics = PerformanceMetrics(
                frame_id=frame_id,
                timestamp=timestamp,
                hand_detection_ms=hand_detection_ms,
                encoding_ms=0.0,
                intent_extraction_ms=0.0,
                total_latency_ms=hand_detection_ms,
                num_hands_detected=0
            )
            
            self._draw_status(annotated_frame, "No hand detected", (0, 0, 255))
            return annotated_frame, metrics
        
        # Extract hand region
        detection = detections[0]
        x, y, w, h = detection.bbox
        hand_region = frame[y:y+h, x:x+w]
        
        if hand_region.size == 0:
            metrics = PerformanceMetrics(
                frame_id=frame_id,
                timestamp=timestamp,
                hand_detection_ms=hand_detection_ms,
                encoding_ms=0.0,
                intent_extraction_ms=0.0,
                total_latency_ms=hand_detection_ms,
                num_hands_detected=num_hands
            )
            return annotated_frame, metrics
        
        # 2. CLIP encoding
        t1 = time.perf_counter()
        embedding = self.encoder.encode(hand_region)
        encoding_ms = (time.perf_counter() - t1) * 1000
        
        # 3. Intent extraction
        t2 = time.perf_counter()
        results = self.intent_extractor.extract_intent(
            embedding,
            top_k=self.config.top_k,
            min_confidence=self.config.confidence_threshold
        )
        intent_extraction_ms = (time.perf_counter() - t2) * 1000
        
        total_latency_ms = hand_detection_ms + encoding_ms + intent_extraction_ms
        
        # Determine prediction status
        if results:
            top_intent = results[0]['intent']
            top_confidence = results[0]['confidence']
            status = "ACCEPTED"
            self.session.successful_predictions += 1
        else:
            top_intent = None
            top_confidence = None
            status = "REJECTED (low confidence)"
            self.session.rejected_predictions += 1
        
        metrics = PerformanceMetrics(
            frame_id=frame_id,
            timestamp=timestamp,
            hand_detection_ms=hand_detection_ms,
            encoding_ms=encoding_ms,
            intent_extraction_ms=intent_extraction_ms,
            total_latency_ms=total_latency_ms,
            num_hands_detected=num_hands,
            top_intent=top_intent,
            top_confidence=top_confidence
        )
        
        # Visualize results
        self._draw_hand_detection(annotated_frame, detection)
        self._draw_intent_results(annotated_frame, results, status)
        
        if self.show_performance:
            self._draw_performance_overlay(annotated_frame, metrics)
        
        return annotated_frame, metrics
    
    def _draw_hand_detection(self, frame: np.ndarray, detection):
        """Draw hand detection bounding box."""
        x, y, w, h = detection.bbox
        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
        
        label = f"{detection.handedness} {detection.confidence:.2f}"
        cv2.putText(frame, label, (x, y - 10),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
    
    def _draw_intent_results(self, frame: np.ndarray, results: List[Dict], status: str):
        """Draw intent extraction results with confidence bars."""
        h, w = frame.shape[:2]
        panel_width = 400
        panel_x = w - panel_width - 20
        panel_y = 20
        
        # Semi-transparent panel
        overlay = frame.copy()
        panel_height = 60 + len(results) * 70
        cv2.rectangle(overlay, (panel_x, panel_y), 
                     (panel_x + panel_width, panel_y + panel_height),
                     (0, 0, 0), -1)
        cv2.addWeighted(overlay, 0.7, frame, 0.3, 0, frame)
        
        # Title
        cv2.putText(frame, "SEMANTIC INTENT", (panel_x + 10, panel_y + 25),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        
        # Status
        status_color = (0, 255, 0) if results else (0, 0, 255)
        cv2.putText(frame, status, (panel_x + 10, panel_y + 50),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, status_color, 1)
        
        # Results
        for i, result in enumerate(results):
            y_pos = panel_y + 80 + i * 70
            
            intent = result['intent']
            confidence = result['confidence']
            
            # Intent label
            cv2.putText(frame, f"{i+1}. {intent}", (panel_x + 10, y_pos),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
            
            # Confidence bar
            bar_width = int((panel_width - 40) * confidence)
            bar_y = y_pos + 10
            cv2.rectangle(frame, (panel_x + 10, bar_y),
                         (panel_x + 10 + bar_width, bar_y + 15),
                         (0, 255, 0), -1)
            
            # Confidence value
            cv2.putText(frame, f"{confidence:.3f}", (panel_x + 10, bar_y + 35),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
            
            # Command mapping
            command = self.config.gesture_definitions.get(intent, "N/A")
            cv2.putText(frame, f"→ {command}", (panel_x + 150, bar_y + 35),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)
    
    def _draw_performance_overlay(self, frame: np.ndarray, metrics: PerformanceMetrics):
        """Draw performance metrics overlay."""
        h, w = frame.shape[:2]
        
        # Calculate FPS
        current_time = time.time()
        dt = current_time - self.last_fps_time
        if dt > 0:
            fps = 1.0 / dt
            self.fps_window.append(fps)
        self.last_fps_time = current_time
        
        avg_fps = np.mean(self.fps_window) if self.fps_window else 0
        
        # Performance panel (top-left)
        overlay = frame.copy()
        panel_height = 150
        cv2.rectangle(overlay, (10, 10), (300, 10 + panel_height), (0, 0, 0), -1)
        cv2.addWeighted(overlay, 0.7, frame, 0.3, 0, frame)
        
        y_pos = 35
        line_height = 25
        
        cv2.putText(frame, "PERFORMANCE", (20, y_pos),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
        
        y_pos += line_height
        cv2.putText(frame, f"FPS: {avg_fps:.1f}", (20, y_pos),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        
        y_pos += line_height
        cv2.putText(frame, f"Latency: {metrics.total_latency_ms:.1f} ms", (20, y_pos),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        
        y_pos += line_height
        cv2.putText(frame, f"  Detect: {metrics.hand_detection_ms:.1f} ms", (20, y_pos),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.4, (200, 200, 200), 1)
        
        y_pos += line_height
        cv2.putText(frame, f"  Encode: {metrics.encoding_ms:.1f} ms", (20, y_pos),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.4, (200, 200, 200), 1)
        
        y_pos += line_height
        cv2.putText(frame, f"  Intent: {metrics.intent_extraction_ms:.1f} ms", (20, y_pos),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.4, (200, 200, 200), 1)
        
        # Latency target indicator
        target_met = metrics.total_latency_ms <= 120
        color = (0, 255, 0) if target_met else (0, 0, 255)
        status = "✓ Target met" if target_met else "✗ Above target"
        cv2.putText(frame, status, (160, 35 + line_height),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.4, color, 1)
    
    def _draw_status(self, frame: np.ndarray, message: str, color: Tuple[int, int, int]):
        """Draw status message."""
        h, w = frame.shape[:2]
        cv2.putText(frame, message, (w // 2 - 150, h // 2),
                   cv2.FONT_HERSHEY_SIMPLEX, 1.0, color, 2)
    
    def _draw_controls(self, frame: np.ndarray):
        """Draw control instructions."""
        h, w = frame.shape[:2]
        
        controls = [
            "Q/ESC: Quit",
            "S: Save frame",
            "P: Pause",
            "T: Toggle perf",
            "H: Help"
        ]
        
        y_pos = h - 120
        for control in controls:
            cv2.putText(frame, control, (10, y_pos),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)
            y_pos += 20
    
    def run(self, output_log: Optional[Path] = None):
        """
        Run demonstration loop.
        
        Args:
            output_log: Optional path to save session log
        """
        print("\n" + "="*70)
        print("STARTING REAL-TIME DEMO")
        print("="*70)
        print(f"\nConfiguration:")
        print(f"  Confidence threshold: {self.config.confidence_threshold}")
        print(f"  Top-K: {self.config.top_k}")
        print(f"  Target latency: ≤120ms")
        print("\nControls:")
        print("  Q / ESC - Quit demo")
        print("  S       - Save current frame")
        print("  P       - Pause/unpause")
        print("  T       - Toggle performance overlay")
        print("  H       - Show help")
        print("="*70 + "\n")
        
        # Start camera
        self.input_mgr.start_continuous_capture()
        
        frame_id = 0
        saved_count = 0
        
        try:
            while True:
                if self.paused:
                    key = cv2.waitKey(100) & 0xFF
                    if key == ord('p'):
                        self.paused = False
                        print("▶ Resumed")
                    elif key == ord('q') or key == 27:
                        break
                    continue
                
                # Get frame
                frame_data = self.input_mgr.get_latest_frame()
                if frame_data is None:
                    continue
                
                frame = frame_data.frame
                
                # Process frame
                annotated_frame, metrics = self.process_frame(frame, frame_id)
                
                # Store metrics
                self.session.performance_metrics.append(metrics)
                self.session.frames_processed += 1
                
                # Draw controls
                self._draw_controls(annotated_frame)
                
                # Display
                cv2.imshow("Semantic Gesture Recognition Demo", annotated_frame)
                
                # Handle keyboard
                key = cv2.waitKey(1) & 0xFF
                
                if key == ord('q') or key == 27:  # Quit
                    break
                elif key == ord('s'):  # Save
                    saved_count += 1
                    filename = f"demo_frame_{saved_count:03d}.jpg"
                    cv2.imwrite(filename, annotated_frame)
                    print(f"✓ Saved: {filename}")
                elif key == ord('p'):  # Pause
                    self.paused = True
                    print("⏸ Paused")
                elif key == ord('t'):  # Toggle performance
                    self.show_performance = not self.show_performance
                elif key == ord('h'):  # Help
                    self._print_help()
                
                frame_id += 1
        
        except KeyboardInterrupt:
            print("\n⚠ Interrupted by user")
        
        finally:
            # Cleanup
            self.session.end_time = time.time()
            self.input_mgr.stop_capture()
            self.hand_detector.close()
            cv2.destroyAllWindows()
            
            # Print summary
            self._print_summary()
            
            # Save log
            if output_log:
                self._save_session_log(output_log)
    
    def _print_help(self):
        """Print help to console."""
        print("\n" + "="*70)
        print("CONTROLS")
        print("="*70)
        print("Q / ESC - Quit demonstration")
        print("S       - Save current frame to file")
        print("P       - Pause/unpause video stream")
        print("T       - Toggle performance metrics overlay")
        print("H       - Show this help message")
        print("="*70 + "\n")
    
    def _print_summary(self):
        """Print session summary."""
        summary = self.session.compute_summary()
        
        print("\n" + "="*70)
        print("DEMO SESSION SUMMARY")
        print("="*70)
        print(f"\nDuration: {self.session.end_time - self.session.start_time:.1f} seconds")
        print(f"Frames processed: {self.session.frames_processed}")
        print(f"Successful predictions: {self.session.successful_predictions}")
        print(f"Rejected predictions: {self.session.rejected_predictions}")
        print(f"Success rate: {summary['success_rate']:.2%}")
        
        print(f"\nLatency Statistics:")
        print(f"  Mean: {summary['mean_latency_ms']:.2f} ms")
        print(f"  Std:  {summary['std_latency_ms']:.2f} ms")
        print(f"  Min:  {summary['min_latency_ms']:.2f} ms")
        print(f"  Max:  {summary['max_latency_ms']:.2f} ms")
        print(f"  P50:  {summary['p50_latency_ms']:.2f} ms")
        print(f"  P95:  {summary['p95_latency_ms']:.2f} ms")
        print(f"  P99:  {summary['p99_latency_ms']:.2f} ms")
        
        target_met_count = sum(1 for m in self.session.performance_metrics 
                               if m.total_latency_ms <= 120)
        target_met_rate = target_met_count / max(self.session.frames_processed, 1)
        
        print(f"\nTarget Latency (≤120ms):")
        print(f"  Frames meeting target: {target_met_count}/{self.session.frames_processed} "
              f"({target_met_rate:.1%})")
        
        if summary['mean_latency_ms'] <= 120:
            print("  ✓ Mean latency meets target")
        else:
            print(f"  ✗ Mean latency exceeds target by {summary['mean_latency_ms'] - 120:.2f} ms")
        
        print("="*70)
    
    def _save_session_log(self, output_path: Path):
        """Save session log to JSON."""
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        with open(output_path, 'w') as f:
            json.dump(self.session.to_dict(), f, indent=2)
        
        print(f"\n✓ Session log saved: {output_path}")


def load_gesture_config(config_path: Path) -> DemoConfig:
    """Load gesture configuration from JSON."""
    with open(config_path, 'r') as f:
        data = json.load(f)
    
    return DemoConfig(
        gesture_definitions=data['gesture_definitions'],
        confidence_threshold=data.get('confidence_threshold', 0.7),
        top_k=data.get('top_k', 3),
        camera_index=data.get('camera_index', 0),
        frame_width=data.get('frame_width', 1280),
        frame_height=data.get('frame_height', 720)
    )


def create_default_config(output_path: Path):
    """Create default gesture configuration file."""
    default_config = {
        "gesture_definitions": {
            "scroll_up": "Scroll page up",
            "scroll_down": "Scroll page down",
            "scroll_left": "Scroll horizontally left",
            "scroll_right": "Scroll horizontally right",
            "zoom_in": "Zoom in / Increase magnification",
            "zoom_out": "Zoom out / Decrease magnification",
            "swipe_left": "Navigate to previous item",
            "swipe_right": "Navigate to next item",
            "click": "Primary mouse click",
            "double_click": "Double-click action",
            "drag": "Drag object",
            "select": "Select item",
            "volume_up": "Increase audio volume",
            "volume_down": "Decrease audio volume",
            "rotate_cw": "Rotate clockwise"
        },
        "confidence_threshold": 0.7,
        "top_k": 3,
        "camera_index": 0,
        "frame_width": 1280,
        "frame_height": 720
    }
    
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, 'w') as f:
        json.dump(default_config, f, indent=2)
    
    print(f"✓ Created default configuration: {output_path}")


def main():
    parser = argparse.ArgumentParser(
        description="Real-Time Semantic Gesture Recognition Demo"
    )
    parser.add_argument('--gesture-set', type=Path, default=None,
                        help='Path to gesture configuration JSON')
    parser.add_argument('--create-config', type=Path, default=None,
                        help='Create default configuration file and exit')
    parser.add_argument('--output', type=Path, default=None,
                        help='Path to save session log JSON')
    parser.add_argument('--confidence-threshold', type=float, default=0.7,
                        help='Minimum confidence threshold for acceptance')
    parser.add_argument('--camera', type=int, default=0,
                        help='Camera index')
    
    args = parser.parse_args()
    
    # Handle config creation
    if args.create_config:
        create_default_config(args.create_config)
        return 0
    
    # Load or create config
    if args.gesture_set and args.gesture_set.exists():
        print(f"Loading gesture configuration: {args.gesture_set}")
        config = load_gesture_config(args.gesture_set)
    else:
        print("Using default gesture configuration")
        # Create temporary default
        default_gestures = {
            "scroll_up": "Scroll up",
            "scroll_down": "Scroll down",
            "zoom_in": "Zoom in",
            "zoom_out": "Zoom out",
            "swipe_left": "Previous",
            "swipe_right": "Next",
            "click": "Click"
        }
        config = DemoConfig(
            gesture_definitions=default_gestures,
            confidence_threshold=args.confidence_threshold,
            camera_index=args.camera
        )
    
    # Run demo
    demo = RealtimeGestureDemo(config)
    demo.run(output_log=args.output)
    
    return 0


if __name__ == "__main__":
    sys.exit(main())
