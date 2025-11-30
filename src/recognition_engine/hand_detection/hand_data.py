"""
Data structures for hand detection results.

This module defines dataclasses to store hand detection information including
bounding boxes, landmarks, handedness, and confidence scores.
"""

import numpy as np
from dataclasses import dataclass
from typing import Tuple, Optional


@dataclass
class HandDetectionResult:
    """
    Container for hand detection results from MediaPipe Hands.
    
    Attributes:
        bbox: Bounding box as (x, y, width, height) in pixels
        landmarks: Array of 21 hand landmarks with (x, y, z) coordinates
                  Shape: (21, 3) - normalized to [0, 1] for x, y; z is depth
        handedness: Which hand was detected ("Left" or "Right")
        confidence: Detection confidence score [0, 1]
        timestamp: Timestamp in nanoseconds when detection was made
        frame_index: Optional frame number for sequence tracking
    
    MediaPipe Hand Landmarks (21 points):
        0:  WRIST
        1-4:  THUMB (CMC, MCP, IP, TIP)
        5-8:  INDEX_FINGER (MCP, PIP, DIP, TIP)
        9-12:  MIDDLE_FINGER (MCP, PIP, DIP, TIP)
        13-16: RING_FINGER (MCP, PIP, DIP, TIP)
        17-20: PINKY (MCP, PIP, DIP, TIP)
    """
    
    bbox: Tuple[int, int, int, int]  # (x, y, width, height)
    landmarks: np.ndarray              # Shape: (21, 3) with (x, y, z)
    handedness: str                    # "Left" or "Right"
    confidence: float                  # [0, 1]
    timestamp: int                     # nanoseconds since epoch
    frame_index: Optional[int] = None  # For sequence tracking
    
    def __post_init__(self):
        """Validate the hand detection result data."""
        # Validate bbox
        if len(self.bbox) != 4:
            raise ValueError(f"bbox must have 4 elements, got {len(self.bbox)}")
        
        # Validate landmarks shape
        if self.landmarks.shape != (21, 3):
            raise ValueError(
                f"landmarks must have shape (21, 3), got {self.landmarks.shape}"
            )
        
        # Validate handedness
        if self.handedness not in ["Left", "Right"]:
            raise ValueError(
                f"handedness must be 'Left' or 'Right', got '{self.handedness}'"
            )
        
        # Validate confidence
        if not 0 <= self.confidence <= 1:
            raise ValueError(
                f"confidence must be in [0, 1], got {self.confidence}"
            )
    
    def get_bbox_center(self) -> Tuple[int, int]:
        """
        Get the center point of the bounding box.
        
        Returns:
            (center_x, center_y) coordinates in pixels
        """
        x, y, w, h = self.bbox
        return (x + w // 2, y + h // 2)
    
    def get_bbox_area(self) -> int:
        """
        Calculate the area of the bounding box in pixels.
        
        Returns:
            Area in pixels squared
        """
        _, _, w, h = self.bbox
        return w * h
    
    def get_landmark(self, index: int) -> np.ndarray:
        """
        Get a specific landmark by index.
        
        Args:
            index: Landmark index (0-20)
            
        Returns:
            Array of (x, y, z) coordinates
        """
        if not 0 <= index < 21:
            raise ValueError(f"Landmark index must be in [0, 20], got {index}")
        return self.landmarks[index]
    
    def get_wrist(self) -> np.ndarray:
        """Get wrist landmark (index 0)."""
        return self.landmarks[0]
    
    def get_fingertips(self) -> np.ndarray:
        """
        Get all fingertip landmarks.
        
        Returns:
            Array of shape (5, 3) with fingertip (x, y, z) coordinates
            Order: [thumb, index, middle, ring, pinky]
        """
        fingertip_indices = [4, 8, 12, 16, 20]
        return self.landmarks[fingertip_indices]
    
    def to_dict(self) -> dict:
        """
        Convert detection result to dictionary for serialization.
        
        Returns:
            Dictionary representation of the detection
        """
        return {
            "bbox": {
                "x": int(self.bbox[0]),
                "y": int(self.bbox[1]),
                "width": int(self.bbox[2]),
                "height": int(self.bbox[3]),
            },
            "landmarks": self.landmarks.tolist(),
            "handedness": self.handedness,
            "confidence": float(self.confidence),
            "timestamp": int(self.timestamp),
            "frame_index": self.frame_index,
        }
    
    def __repr__(self) -> str:
        """String representation for debugging."""
        x, y, w, h = self.bbox
        return (
            f"HandDetectionResult("
            f"hand={self.handedness}, "
            f"confidence={self.confidence:.2f}, "
            f"bbox=({x}, {y}, {w}×{h}), "
            f"timestamp={self.timestamp})"
        )


@dataclass
class HandROI:
    """
    Container for cropped and preprocessed hand region of interest.
    
    This is the output of HandCropper, ready for gesture classification.
    
    Attributes:
        image: Preprocessed hand image, typically 224×224×3
        detection: Original detection result this ROI was extracted from
        preprocessing_params: Dictionary of preprocessing parameters used
    """
    
    image: np.ndarray
    detection: HandDetectionResult
    preprocessing_params: dict
    
    def __post_init__(self):
        """Validate the hand ROI data."""
        if self.image.ndim not in [2, 3]:
            raise ValueError(
                f"image must be 2D or 3D array, got shape {self.image.shape}"
            )
    
    def get_shape(self) -> Tuple[int, ...]:
        """Get the shape of the preprocessed image."""
        return self.image.shape
    
    def to_dict(self) -> dict:
        """
        Convert ROI to dictionary for serialization.
        
        Note: Image data is not included to reduce size
        
        Returns:
            Dictionary representation
        """
        return {
            "image_shape": self.image.shape,
            "image_dtype": str(self.image.dtype),
            "detection": self.detection.to_dict(),
            "preprocessing_params": self.preprocessing_params,
        }
    
    def __repr__(self) -> str:
        """String representation for debugging."""
        return (
            f"HandROI("
            f"shape={self.image.shape}, "
            f"dtype={self.image.dtype}, "
            f"hand={self.detection.handedness})"
        )
