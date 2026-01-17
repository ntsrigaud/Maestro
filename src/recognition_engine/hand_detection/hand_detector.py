"""
Hand detection using MediaPipe Hands.

This module provides the HandDetector class for detecting and localizing hands
in video frames using Google's MediaPipe Hands solution.

COMPATIBILITY NOTE: Refactored for MediaPipe 0.10.x+ API while maintaining
exact backward compatibility with the previous 0.9.x interface.
"""

import cv2
import numpy as np
import time
import os
import urllib.request
from pathlib import Path
from typing import List, Optional, Tuple
from recognition_engine.hand_detection.hand_data import HandDetectionResult
from shared.logger import get_logger

# Initialize module logger
logger = get_logger(__name__)

# Import MediaPipe - handle API versions
try:
    import mediapipe as mp
    from mediapipe.tasks import python
    from mediapipe.tasks.python import vision
    
    MEDIAPIPE_VERSION = mp.__version__
    logger.info(f"MediaPipe version {MEDIAPIPE_VERSION} loaded successfully")
    
except ImportError as e:
    raise ImportError(
        f"MediaPipe not found or incomplete: {e}\n"
        "Install with: pip install mediapipe>=0.10.0"
    )

# Model configuration
MODEL_URL = "https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/1/hand_landmarker.task"
MODEL_NAME = "hand_landmarker.task"


def _get_model_path() -> str:
    """
    Get or download the MediaPipe hand landmarker model.
    
    Returns:
        Absolute path to the model file
    """
    # Try multiple locations in order of preference
    model_locations = [
        # 1. User's home directory cache
        Path.home() / ".mediapipe" / "models" / MODEL_NAME,
        # 2. Project directory
        Path(__file__).parent / "models" / MODEL_NAME,
        # 3. System temp directory (fallback)
        Path("/tmp") / "mediapipe_models" / MODEL_NAME,
    ]
    
    # Check if model exists in any location
    for model_path in model_locations:
        if model_path.exists():
            logger.debug(f"Using cached model: {model_path}")
            return str(model_path)
    
    # Model not found, download to first location
    download_path = model_locations[0]
    download_path.parent.mkdir(parents=True, exist_ok=True)
    
    logger.info(f"Downloading hand landmarker model to {download_path}...")
    try:
        urllib.request.urlretrieve(MODEL_URL, download_path)
        logger.info(f"✓ Model downloaded successfully")
        return str(download_path)
    except Exception as e:
        # Try fallback location if home directory fails
        download_path = model_locations[2]
        download_path.parent.mkdir(parents=True, exist_ok=True)
        try:
            urllib.request.urlretrieve(MODEL_URL, download_path)
            logger.info(f"✓ Model downloaded to fallback location: {download_path}")
            return str(download_path)
        except Exception as e2:
            raise RuntimeError(
                f"Failed to download MediaPipe hand landmarker model:\n"
                f"  Primary error: {e}\n"
                f"  Fallback error: {e2}\n"
                f"Manual download: {MODEL_URL}\n"
                f"Save to: {model_locations[0]}"
            )


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

        # Initialize MediaPipe Hands using new API (0.10.x+)
        # Configure running mode based on static_image_mode
        running_mode = (
            vision.RunningMode.IMAGE if static_image_mode
            else vision.RunningMode.VIDEO
        )
        
        # Get or download the model
        model_path = _get_model_path()
        
        # Create base options with explicit model path
        base_options = python.BaseOptions(
            model_asset_path=model_path
        )
        
        # Create hand landmarker options
        # Note: MediaPipe 0.10+ doesn't have separate model_complexity parameter
        # Model quality is fixed; we ignore this parameter but keep API compatible
        options = vision.HandLandmarkerOptions(
            base_options=base_options,
            running_mode=running_mode,
            num_hands=max_num_hands,
            min_hand_detection_confidence=min_detection_confidence,
            min_hand_presence_confidence=min_tracking_confidence,
            min_tracking_confidence=min_tracking_confidence,
        )
        
        # Create the hand landmarker
        self._detector = vision.HandLandmarker.create_from_options(options)

        # Performance tracking
        self._frame_count = 0
        self._detection_count = 0
        
        # Track last frame time for VIDEO mode timestamp calculation
        self._last_timestamp_ms = 0

        logger.info(
            f"HandDetector initialized (MediaPipe {MEDIAPIPE_VERSION}): "
            f"max_hands={max_num_hands}, "
            f"detection_conf={min_detection_confidence}, "
            f"tracking_conf={min_tracking_confidence}, "
            f"mode={'IMAGE' if static_image_mode else 'VIDEO'}"
        )

    def detect(
        self, frame: np.ndarray, timestamp: Optional[int] = None
    ) -> List[HandDetectionResult]:
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
        
        # Create MediaPipe Image
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=frame_rgb)
        
        # Process based on running mode
        if self._static_image_mode:
            # IMAGE mode: no timestamp needed
            detection_result = self._detector.detect(mp_image)
        else:
            # VIDEO mode: requires timestamp in milliseconds
            # Convert nanoseconds to milliseconds
            timestamp_ms = timestamp // 1_000_000
            
            # Ensure timestamps are monotonically increasing for VIDEO mode
            if timestamp_ms <= self._last_timestamp_ms:
                timestamp_ms = self._last_timestamp_ms + 1
            
            self._last_timestamp_ms = timestamp_ms
            
            detection_result = self._detector.detect_for_video(mp_image, timestamp_ms)

        # No hands detected
        if not detection_result.hand_landmarks:
            return []

        # Extract detection results
        detections = []
        frame_height, frame_width = frame.shape[:2]

        # Iterate through detected hands
        for idx in range(len(detection_result.hand_landmarks)):
            hand_landmarks_list = detection_result.hand_landmarks[idx]
            handedness_list = detection_result.handedness[idx]
            
            # Extract landmarks as numpy array (normalized coordinates)
            landmarks = self._extract_landmarks(hand_landmarks_list)

            # Calculate bounding box from landmarks
            bbox = self._calculate_bbox(landmarks, frame_width, frame_height)

            # Extract handedness and confidence
            # handedness_list is a list of Category objects
            handedness_category = handedness_list[0]
            handedness = handedness_category.category_name  # "Left" or "Right"
            confidence = handedness_category.score

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
        self, frames: List[np.ndarray], timestamps: Optional[List[int]] = None
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

    def _extract_landmarks(self, hand_landmarks_list) -> np.ndarray:
        """
        Extract hand landmarks as numpy array.

        Args:
            hand_landmarks_list: MediaPipe hand landmarks list (from new API)

        Returns:
            Array of shape (21, 3) with normalized (x, y, z) coordinates
        """
        # New API: hand_landmarks_list is a list of NormalizedLandmark objects
        landmarks = np.array(
            [[lm.x, lm.y, lm.z] for lm in hand_landmarks_list], 
            dtype=np.float32
        )

        return landmarks

    def _calculate_bbox(
        self, landmarks: np.ndarray, frame_width: int, frame_height: int
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
                if self._frame_count > 0
                else 0
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
        if hasattr(self, '_detector') and self._detector:
            self._detector.close()
            self._detector = None
            logger.info("HandDetector closed")

    def __enter__(self):
        """Context manager entry."""
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit."""
        self.close()

    def __del__(self):
        """Cleanup on deletion."""
        try:
            self.close()
        except:
            pass  # Silently handle cleanup errors during destruction

    def __repr__(self) -> str:
        """String representation for debugging."""
        return (
            f"HandDetector("
            f"max_hands={self._max_num_hands}, "
            f"detection_conf={self._min_detection_confidence}, "
            f"complexity={self._model_complexity})"
        )
