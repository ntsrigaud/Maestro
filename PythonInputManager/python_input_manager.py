"""
Python Input Manager Module

A high-performance Python implementation of the Input Manager for real-time video capture,
preprocessing, and hand detection. This module replaces the C++/pybind11 hybrid approach
with a pure Python solution that provides better maintainability, development velocity,
and ecosystem integration while maintaining comparable performance.

Key Features:
- Real-time video capture with OpenCV
- Optimized preprocessing pipeline (BGR→RGB, resize, normalize)
- MediaPipe integration for hand detection
- Thread-safe frame buffering
- Compatible API with legacy C++ implementation
- Comprehensive error handling and logging
"""

import cv2
import numpy as np
import mediapipe as mp
import time
import logging
from threading import Thread, Lock, Event
from queue import Queue, Empty
from dataclasses import dataclass, field
from typing import Optional, List, Tuple, Dict, Any, Callable
from contextlib import contextmanager


# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@dataclass
class FrameData:
    """
    Container for frame data with metadata.
    
    Attributes:
        frame: Preprocessed frame as numpy array (224x224x3, float32)
        timestamp: Unix timestamp when frame was captured
        hands_detected: MediaPipe hand detection results
        metadata: Additional frame metadata
    """
    frame: np.ndarray
    timestamp: float
    hands_detected: Optional[List] = None
    metadata: Optional[Dict[str, Any]] = field(default_factory=dict)
    
    def get_frame(self) -> np.ndarray:
        """
        Returns the frame data as numpy array for API compatibility.
        
        Returns:
            numpy.ndarray: Frame data with shape (224, 224, 3)
        """
        return self.frame


class PythonInputManager:
    """
    High-performance Python Input Manager for video capture and preprocessing.
    
    This class provides real-time video capture with preprocessing capabilities
    including frame normalization, resizing, color conversion, and optional
    MediaPipe hand detection integration.
    """
    
    def __init__(
        self, 
        camera_index: int = 0, 
        target_size: Tuple[int, int] = (224, 224),
        enable_mediapipe: bool = True,
        max_queue_size: int = 30
    ):
        """
        Initialize the Python Input Manager.
        
        Args:
            camera_index: Index of the camera device to use
            target_size: Target frame size for preprocessing (width, height)
            enable_mediapipe: Whether to enable MediaPipe hand detection
            max_queue_size: Maximum number of frames to buffer in memory
        """
        self.camera_index = camera_index
        self.target_size = target_size
        self.max_queue_size = max_queue_size
        
        # Frame processing parameters
        self.mirror_enabled = True
        self.normalization_params = {
            'mean': 0.5,
            'std': 0.5
        }
        
        # Threading components
        self.capture_thread: Optional[Thread] = None
        self.is_capturing = Event()
        self.frame_lock = Lock()
        self.frame_queue: Queue[FrameData] = Queue(maxsize=max_queue_size)
        
        # Camera and processing components
        self.cap: Optional[cv2.VideoCapture] = None
        self.latest_frame: Optional[FrameData] = None
        self.captured_frames: List[FrameData] = []
        
        # MediaPipe initialization
        self.enable_mediapipe = enable_mediapipe
        self.hands = None
        if enable_mediapipe:
            try:
                self.hands = mp.solutions.hands.Hands(
                    static_image_mode=False,
                    max_num_hands=2,
                    min_detection_confidence=0.7,
                    min_tracking_confidence=0.5
                )
                logger.info("MediaPipe Hands initialized successfully")
            except Exception as e:
                logger.warning(f"Failed to initialize MediaPipe: {e}")
                self.enable_mediapipe = False
        
        # Performance metrics
        self.frame_count = 0
        self.processing_times = []
        
    def __enter__(self):
        """Context manager entry."""
        return self
        
    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit with cleanup."""
        self.stop_continuous_capture()
        
    def is_capture_open(self) -> bool:
        """
        Check if camera capture is currently open.
        
        Returns:
            bool: True if camera is open, False otherwise
        """
        return self.cap is not None and self.cap.isOpened()
    
    def start_capture(self) -> bool:
        """
        Initialize camera capture device.
        
        Returns:
            bool: True if camera opened successfully, False otherwise
        """
        if self.is_capture_open():
            return True
        
        try:
            self.cap = cv2.VideoCapture(self.camera_index)
            if not self.cap.isOpened():
                logger.error(f"Failed to open camera {self.camera_index}")
                return False
            
            # Configure camera properties
            self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
            self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
            self.cap.set(cv2.CAP_PROP_FPS, 30)
            self.cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)  # Reduce latency
            
            logger.info(f"Camera {self.camera_index} opened successfully")
            return True
            
        except Exception as e:
            logger.error(f"Error starting camera capture: {e}")
            return False
    
    def stop_capture(self) -> None:
        """Stop camera capture and release resources."""
        if self.cap is not None:
            self.cap.release()
            self.cap = None
            logger.info("Camera capture stopped")
    
    def start_continuous_capture(self) -> bool:
        """
        Start continuous frame capture in background thread.
        
        Returns:
            bool: True if capture started successfully, False otherwise
        """
        if not self.start_capture():
            return False
        
        if self.is_capturing.is_set():
            logger.warning("Capture already running")
            return True
        
        # Clear previous data
        with self.frame_lock:
            while not self.frame_queue.empty():
                try:
                    self.frame_queue.get_nowait()
                except Empty:
                    break
            self.captured_frames.clear()
            self.latest_frame = None
        
        # Start capture thread
        self.is_capturing.set()
        self.capture_thread = Thread(target=self._capture_loop, daemon=True)
        self.capture_thread.start()
        
        logger.info("Continuous capture started")
        return True
    
    def stop_continuous_capture(self) -> None:
        """Stop continuous capture and cleanup resources."""
        if not self.is_capturing.is_set():
            return
        
        self.is_capturing.clear()
        
        if self.capture_thread and self.capture_thread.is_alive():
            self.capture_thread.join(timeout=2.0)
            if self.capture_thread.is_alive():
                logger.warning("Capture thread did not stop gracefully")
        
        self.stop_capture()
        logger.info("Continuous capture stopped")
    
    def _capture_loop(self) -> None:
        """Main capture loop running in background thread."""
        logger.info("Capture loop started")
        
        while self.is_capturing.is_set():
            if not self.is_capture_open():
                logger.error("Camera not available in capture loop")
                break
            
            start_time = time.time()
            
            # Capture frame
            ret, frame = self.cap.read()
            if not ret or frame is None:
                logger.warning("Failed to read frame from camera")
                continue
            
            # Process frame
            try:
                # Apply mirroring first if enabled (for consistency with hand detection)
                display_frame = frame
                if self.mirror_enabled:
                    display_frame = cv2.flip(frame, 1)
                
                # Process the mirrored frame for preprocessing
                processed_frame = self._process_frame(display_frame, apply_mirror=False)  # Don't mirror again
                
                if processed_frame is not None:
                    # Detect hands if enabled (use the same mirrored frame)
                    hands_result = None
                    if self.enable_mediapipe and self.hands:
                        hands_result = self._detect_hands(display_frame)
                    # Create frame data
                    frame_data = FrameData(
                        frame=processed_frame,
                        timestamp=time.time(),
                        hands_detected=hands_result,
                        metadata={
                            'frame_count': self.frame_count,
                            'processing_time': time.time() - start_time
                        }
                    )
                    
                    # Update buffers
                    with self.frame_lock:
                        self.latest_frame = frame_data
                        self.captured_frames.append(frame_data)
                        
                        # Add to queue (non-blocking)
                        if not self.frame_queue.full():
                            self.frame_queue.put(frame_data)
                    
                    self.frame_count += 1
                    self.processing_times.append(time.time() - start_time)
                    
                    # Keep only recent processing times for metrics
                    if len(self.processing_times) > 100:
                        self.processing_times = self.processing_times[-100:]
                
            except Exception as e:
                logger.error(f"Error processing frame: {e}")
            
            # Small sleep to prevent CPU spinning
            time.sleep(0.001)
        
        logger.info("Capture loop ended")
    
    def _process_frame(self, frame: np.ndarray, apply_mirror: bool = True) -> Optional[np.ndarray]:
        """
        Apply preprocessing pipeline to frame.
        
        Args:
            frame: Raw frame from camera
            apply_mirror: Whether to apply mirroring (default True for backward compatibility)
            
        Returns:
            Preprocessed frame or None if processing failed
        """
        try:
            # Mirror if enabled and apply_mirror is True
            if self.mirror_enabled and apply_mirror:
                frame = cv2.flip(frame, 1)
            
            # Convert BGR to RGB
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            
            # Resize to target size
            resized_frame = cv2.resize(rgb_frame, self.target_size, interpolation=cv2.INTER_AREA)
            
            # Convert to float32 and normalize to [0, 1]
            float_frame = resized_frame.astype(np.float32) / 255.0
            
            # Apply normalization parameters
            mean = self.normalization_params['mean']
            std = self.normalization_params['std']
            normalized_frame = (float_frame - mean) / std
            
            return normalized_frame
            
        except Exception as e:
            logger.error(f"Frame processing error: {e}")
            return None
    
    def _detect_hands(self, frame: np.ndarray) -> Optional[List[Dict]]:
        """
        Detect hands in frame using MediaPipe.
        
        Args:
            frame: RGB frame for hand detection
            
        Returns:
            List of hand detection results or None if no hands detected
        """
        if not self.hands:
            return None
        
        try:
            # Convert BGR to RGB for MediaPipe
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = self.hands.process(rgb_frame)
            
            if results.multi_hand_landmarks:
                hands_data = []
                for i, hand_landmarks in enumerate(results.multi_hand_landmarks):
                    hand_data = {
                        'landmarks': [(lm.x, lm.y, lm.z) for lm in hand_landmarks.landmark],
                        'handedness': results.multi_handedness[i].classification[0].label if results.multi_handedness else 'Unknown'
                    }
                    hands_data.append(hand_data)
                return hands_data
            
        except Exception as e:
            logger.error(f"Hand detection error: {e}")
        
        return None
    
    def get_latest_frame(self) -> Optional[FrameData]:
        """
        Get the most recently captured frame.
        
        Returns:
            Most recent FrameData or None if no frames available
        """
        with self.frame_lock:
            return self.latest_frame
    
    def get_all_frames(self) -> List[FrameData]:
        """
        Get all captured frames since continuous capture started.
        
        Returns:
            List of all FrameData objects
        """
        with self.frame_lock:
            return self.captured_frames.copy()
    
    def set_normalization_params(self, mean: float, std: float) -> None:
        """
        Set normalization parameters for frame preprocessing.
        
        Args:
            mean: Mean value for normalization
            std: Standard deviation for normalization
        """
        self.normalization_params = {'mean': mean, 'std': std}
        logger.info(f"Normalization parameters updated: mean={mean}, std={std}")
    
    def set_mirror_enabled(self, enabled: bool) -> None:
        """
        Enable or disable frame mirroring.
        
        Args:
            enabled: Whether to mirror frames horizontally
        """
        self.mirror_enabled = enabled
        logger.info(f"Frame mirroring {'enabled' if enabled else 'disabled'}")
    
    def get_performance_metrics(self) -> Dict[str, Any]:
        """
        Get performance metrics for the Input Manager.
        
        Returns:
            Dictionary containing performance statistics
        """
        if not self.processing_times:
            return {}
        
        processing_times = np.array(self.processing_times)
        
        return {
            'frame_count': self.frame_count,
            'avg_processing_time_ms': np.mean(processing_times) * 1000,
            'min_processing_time_ms': np.min(processing_times) * 1000,
            'max_processing_time_ms': np.max(processing_times) * 1000,
            'fps': 1.0 / np.mean(processing_times) if np.mean(processing_times) > 0 else 0,
            'queue_size': self.frame_queue.qsize(),
            'mediapipe_enabled': self.enable_mediapipe
        }


# Compatibility wrapper for legacy C++ API
class InputManager:
    """
    Compatibility wrapper that maintains the original C++ API interface.
    This allows for seamless migration without changing existing code.
    """
    
    def __init__(self, camera_index: int = 0):
        """Initialize with camera index for C++ API compatibility."""
        self._python_manager = PythonInputManager(camera_index=camera_index, enable_mediapipe=False)
    
    def start_capture(self) -> bool:
        """Start camera capture."""
        return self._python_manager.start_capture()
    
    def stop_capture(self) -> None:
        """Stop camera capture."""
        self._python_manager.stop_capture()
    
    def is_capture_open(self) -> bool:
        """Check if capture is open."""
        return self._python_manager.is_capture_open()
    
    def start_continuous_capture(self) -> bool:
        """Start continuous capture."""
        return self._python_manager.start_continuous_capture()
    
    def stop_continuous_capture(self) -> None:
        """Stop continuous capture."""
        self._python_manager.stop_continuous_capture()
    
    def get_latest_frame(self) -> Optional[FrameData]:
        """Get latest frame."""
        return self._python_manager.get_latest_frame()
    
    def get_all_frames(self) -> List[FrameData]:
        """Get all frames."""
        return self._python_manager.get_all_frames()
    
    def set_normalization_params(self, mean: float, std: float) -> None:
        """Set normalization parameters."""
        self._python_manager.set_normalization_params(mean, std)
    
    def set_mirror_enabled(self, enabled: bool) -> None:
        """Set mirror enabled."""
        self._python_manager.set_mirror_enabled(enabled)