"""
Simple preprocessing utilities for gesture recognition.

This module provides deterministic preprocessing that matches training exactly.
No hand detection - uses center-crop approach aligned with HaGRID full-frame training.
"""

import cv2
import numpy as np
from typing import Tuple


def center_crop_frame(
    frame: np.ndarray,
    target_size: int = 224,
    bgr_to_rgb: bool = True
) -> np.ndarray:
    """
    Center-crop frame to square, resize to target size.
    
    This preprocessing matches the training pipeline exactly:
    1. Extract center square crop from frame
    2. Resize to target_size x target_size
    3. Optionally convert BGR to RGB
    
    Args:
        frame: Input frame (H, W, 3) in BGR or RGB format
        target_size: Output size (default: 224 for ViT-B/16)
        bgr_to_rgb: Convert from BGR to RGB (default: True)
        
    Returns:
        Preprocessed frame (target_size, target_size, 3) in RGB format
        
    Example:
        >>> frame = cv2.VideoCapture(0).read()[1]  # BGR from webcam
        >>> preprocessed = center_crop_frame(frame, target_size=224)
        >>> # Now ready for ViT inference
    """
    # Get frame dimensions
    h, w = frame.shape[:2]
    
    # Calculate center crop (square)
    size = min(h, w)
    x = (w - size) // 2
    y = (h - size) // 2
    
    # Extract center square
    cropped = frame[y:y+size, x:x+size]
    
    # Resize to target size
    resized = cv2.resize(cropped, (target_size, target_size), interpolation=cv2.INTER_LINEAR)
    
    # Convert BGR to RGB if needed
    if bgr_to_rgb:
        resized = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB)
    
    return resized


def get_crop_coordinates(frame_shape: Tuple[int, int]) -> Tuple[int, int, int, int]:
    """
    Get coordinates for center square crop without actually cropping.
    
    Useful for visualization (drawing bounding box on original frame).
    
    Args:
        frame_shape: (height, width) of input frame
        
    Returns:
        (x1, y1, x2, y2) coordinates of center square crop
        
    Example:
        >>> h, w = frame.shape[:2]
        >>> x1, y1, x2, y2 = get_crop_coordinates((h, w))
        >>> cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
    """
    h, w = frame_shape
    size = min(h, w)
    x1 = (w - size) // 2
    y1 = (h - size) // 2
    x2 = x1 + size
    y2 = y1 + size
    return x1, y1, x2, y2
