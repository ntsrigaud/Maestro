"""
Recognition Engine Module

The Recognition Engine is the core AI module responsible for semantic gesture recognition.
It consists of two main submodules:
1. Hand Detection - Localize and extract hand regions from video frames
2. Gesture Classification - Classify static and dynamic gestures (to be implemented)
"""

from recognition_engine.hand_detection import (
    HandDetector,
    HandCropper,
    HandDetectionResult,
)

__all__ = [
    "HandDetector",
    "HandCropper",
    "HandDetectionResult",
]

__version__ = "0.1.0"
