"""
Hand detection using MediaPipe Hands.

This module provides the HandDetector class for detecting and localizing hands
in video frames using Google's MediaPipe Hands solution.
"""

import cv2
import numpy as np
import mediapipe as mp
import time
from typing import List, Optional, Tuple
from recognition_engine.hand_detection.hand_data import HandDetectionResult
from shared.logger import get_logger

# Initialize module logger
logger = get_logger(__name__)


class HandDetector:
    """
    Hand detector using MediaPipe Hands for real-time hand localization.
    
    This class wraps MediaPipe's Hands solution to detect hands in images and video
    frames, extract hand landmarks, compute bounding boxes, and provide confidence scores.
    
    Features:
        - Multi-hand detection (up to max_num_hands)
        - Real-time video stream processing
        - Configurable confidence thresholds
        - Bounding box extraction from landmarks
        - Handedness classification (Left/Right)
    
    Example:
        >>> detector = HandDetector(max_num_hands=2, min_detection_confidence=0.5)
        >>> frame = cv2.imread("hand_image.jpg")
        >>> detections = detector.detect(frame)
        >>> for detection in detections:
        ...     print(f"Detected {detection.handedness} hand with confidence {detection.confidence:.2f}")
    """
    
    def __init__(
        self,
        max_num_hands: int = 2,
        min_detection_confidence: float = 0.5,
        min_tracking_confidence: float = 0.5,
        model_complexity: int = 1,
        static_image_mode: bool = False,
    ):
        """
        Initialize the hand detector.
        
        Args:
            max_num_hands: Maximum number of hands to detect (1-2 recommended)
            min_detection_confidence: Minimum confidence for hand detection [0, 1]
            min_tracking_confidence: Minimum confidence for hand tracking [0, 1]
            model_complexity: Complexity of the hand landmark model (0=lite, 1=full)
                             0: Lite model (faster, less accurate)
                             1: Full model (slower, more accurate)
            static_image_mode: Whether to treat each image as independent
                              False: Optimized for video (uses tracking)
                              True: Optimized for static images (no tracking)
        """
        self._max_num_hands = max_num_hands
        self._min_detection_confidence = min_detection_confidence
        self._min_tracking_confidence = min_tracking_confidence
        self._model_complexity = model_complexity
        self._static_image_mode = static_image_mode
        
        # Initialize MediaPipe Hands
        self._mp_hands = mp.solutions.hands
        self._hands = self._mp_hands.Hands(
            static_image_mode=static_image_mode,
            max_num_hands=max_num_hands,
            min_detection_confidence=min_detection_confidence,
            min_tracking_confidence=min_tracking_confidence,
            model_complexity=model_complexity,
        )
        
        # Performance tracking
        self._frame_count = 0
        self._detection_count = 0
        
        logger.info(
            f"HandDetector initialized: "
            f"max_hands={max_num_hands}, "
            f"detection_conf={min_detection_confidence}, "
            f"tracking_conf={min_tracking_confidence}, "
            f"complexity={model_complexity}"
        )
    
    def detect(self, frame: np.ndarray, timestamp: Optional[int] = None) -> List[HandDetectionResult]:
        """
        Detect hands in a single frame.
        
        Args:
            frame: Input image as numpy array (BGR format from OpenCV)
            timestamp: Optional timestamp in nanoseconds. If None, current time is used
            
        Returns:
            List of HandDetectionResult objects, one per detected hand
            Empty list if no hands detected
        """
        if frame is None or frame.size == 0:
            logger.warning("Empty frame provided to detect()")
            return []
        
        # Use current time if no timestamp provided
        if timestamp is None:
            timestamp = time.time_ns()
        
        self._frame_count += 1
        
        # Convert BGR to RGB (MediaPipe expects RGB)
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        
        # Process the frame
        results = self._hands.process(frame_rgb)
        
        # No hands detected
        if not results.multi_hand_landmarks:
            return []
        
        # Extract detection results
        detections = []
        frame_height, frame_width = frame.shape[:2]
        
        for idx, (hand_landmarks, hand_handedness) in enumerate(
            zip(results.multi_hand_landmarks, results.multi_handedness)
        ):
            # Extract landmarks as numpy array (normalized coordinates)
            landmarks = self._extract_landmarks(hand_landmarks)
            
            # Calculate bounding box from landmarks
            bbox = self._calculate_bbox(landmarks, frame_width, frame_height)
            
            # Extract handedness and confidence
            handedness = hand_handedness.classification[0].label  # "Left" or "Right"
            confidence = hand_handedness.classification[0].score
            
            # Create detection result
            detection = HandDetectionResult(
                bbox=bbox,
                landmarks=landmarks,
                handedness=handedness,
                confidence=confidence,
                timestamp=timestamp,
                frame_index=self._frame_count,
            )
            
            detections.append(detection)
            self._detection_count += 1
        
        if detections:
            logger.debug(
                f"Frame {self._frame_count}: Detected {len(detections)} hand(s) "
                f"with confidences {[d.confidence for d in detections]}"
            )
        
        return detections
    
    def detect_batch(
        self, 
        frames: List[np.ndarray],
        timestamps: Optional[List[int]] = None
    ) -> List[List[HandDetectionResult]]:
        """
        Detect hands in multiple frames (batch processing).
        
        Args:
            frames: List of input frames (BGR format)
            timestamps: Optional list of timestamps (one per frame)
            
        Returns:
            List of detection lists, one per frame
        """
        if not frames:
            return []
        
        if timestamps is None:
            timestamps = [None] * len(frames)
        
        if len(timestamps) != len(frames):
            raise ValueError(
                f"Number of timestamps ({len(timestamps)}) must match "
                f"number of frames ({len(frames)})"
            )
        
        results = []
        for frame, ts in zip(frames, timestamps):
            detections = self.detect(frame, timestamp=ts)
            results.append(detections)
        
        return results
    
    def _extract_landmarks(self, hand_landmarks) -> np.ndarray:
        """
        Extract hand landmarks as numpy array.
        
        Args:
            hand_landmarks: MediaPipe hand landmarks object
            
        Returns:
            Array of shape (21, 3) with normalized (x, y, z) coordinates
        """
        landmarks = np.array([
            [lm.x, lm.y, lm.z]
            for lm in hand_landmarks.landmark
        ], dtype=np.float32)
        
        return landmarks
    
    def _calculate_bbox(
        self, 
        landmarks: np.ndarray, 
        frame_width: int, 
        frame_height: int
    ) -> Tuple[int, int, int, int]:
        """
        Calculate bounding box from hand landmarks.
        
        Args:
            landmarks: Hand landmarks array of shape (21, 3)
            frame_width: Width of the frame in pixels
            frame_height: Height of the frame in pixels
            
        Returns:
            Tuple of (x, y, width, height) in pixel coordinates
        """
        # Get x, y coordinates (landmarks are normalized to [0, 1])
        x_coords = landmarks[:, 0] * frame_width
        y_coords = landmarks[:, 1] * frame_height
        
        # Find extrema
        x_min = int(np.min(x_coords))
        y_min = int(np.min(y_coords))
        x_max = int(np.max(x_coords))
        y_max = int(np.max(y_coords))
        
        # Calculate width and height
        width = x_max - x_min
        height = y_max - y_min
        
        return (x_min, y_min, width, height)
    
    def get_stats(self) -> dict:
        """
        Get detection statistics.
        
        Returns:
            Dictionary with performance statistics
        """
        return {
            "frames_processed": self._frame_count,
            "total_detections": self._detection_count,
            "avg_detections_per_frame": (
                self._detection_count / self._frame_count 
                if self._frame_count > 0 else 0
            ),
            "max_num_hands": self._max_num_hands,
            "min_detection_confidence": self._min_detection_confidence,
            "model_complexity": self._model_complexity,
        }
    
    def reset_stats(self):
        """Reset performance statistics."""
        self._frame_count = 0
        self._detection_count = 0
        logger.debug("HandDetector statistics reset")
    
    def close(self):
        """Release MediaPipe resources."""
        if self._hands:
            self._hands.close()
            self._hands = None
            logger.info("HandDetector closed")
    
    def __enter__(self):
        """Context manager entry."""
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit."""
        self.close()
    
    def __del__(self):
        """Cleanup on deletion."""
        self.close()
    
    def __repr__(self) -> str:
        """String representation for debugging."""
        return (
            f"HandDetector("
            f"max_hands={self._max_num_hands}, "
            f"detection_conf={self._min_detection_confidence}, "
            f"complexity={self._model_complexity})"
        )
