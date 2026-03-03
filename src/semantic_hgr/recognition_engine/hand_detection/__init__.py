"""
Hand Detection Submodule

This submodule provides hand detection and localization capabilities using MediaPipe Hands.
It detects hands in video frames, extracts bounding boxes, and crops hand regions for
downstream gesture classification.
"""

from semantic_hgr.recognition_engine.hand_detection.hand_data import HandDetectionResult, HandROI
from semantic_hgr.recognition_engine.hand_detection.hand_detector import HandDetector
from semantic_hgr.recognition_engine.hand_detection.hand_cropper import HandCropper

__all__ = [
    "HandDetectionResult",
    "HandROI",
    "HandDetector",
    "HandCropper",
]
