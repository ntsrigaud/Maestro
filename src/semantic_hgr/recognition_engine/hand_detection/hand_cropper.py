"""
Hand region cropping and ROI extraction.

This module provides the HandCropper class for extracting hand regions from frames
using bounding boxes from hand detections, with configurable padding and preprocessing.
"""

import cv2
import numpy as np
from typing import Tuple
from semantic_hgr.recognition_engine.hand_detection.hand_data import HandDetectionResult, HandROI
from semantic_hgr.shared.logger import get_logger

# Initialize module logger
logger = get_logger(__name__)


class HandCropper:
    """
    Extracts hand regions from frames using detection bounding boxes.

    This class crops hand regions with configurable padding and ensures the
    resulting ROI has the correct dimensions for downstream processing.

    Features:
        - Configurable padding around bounding boxes
        - Square aspect ratio cropping
        - Frame boundary handling
        - Automatic resizing to target dimensions
        - Preprocessing integration (normalization, color conversion)

    Example:
        >>> cropper = HandCropper(target_size=(224, 224), padding=0.2)
        >>> frame = cv2.imread("hand_image.jpg")
        >>> detection = detector.detect(frame)[0]
        >>> hand_roi = cropper.crop_hand(frame, detection)
        >>> print(hand_roi.image.shape)  # (224, 224, 3)
    """

    def __init__(
        self,
        target_size: Tuple[int, int] = (224, 224),
        padding: float = 0.2,
        square_crop: bool = True,
        normalize: bool = True,
        normalization_mean: Tuple[float, float, float] = (0.485, 0.456, 0.406),
        normalization_std: Tuple[float, float, float] = (0.229, 0.224, 0.225),
        bgr_to_rgb: bool = True,
    ):
        """
        Initialize the hand cropper.

        Args:
            target_size: Target (width, height) for resized ROI
            padding: Padding factor around bounding box (0.2 = 20% padding)
            square_crop: Whether to crop square regions (for ViT and similar models)
            normalize: Whether to normalize pixel values
            normalization_mean: Mean values for normalization (ImageNet defaults)
            normalization_std: Std values for normalization (ImageNet defaults)
            bgr_to_rgb: Whether to convert BGR (OpenCV) to RGB format
        """
        self._target_size = target_size
        self._padding = padding
        self._square_crop = square_crop
        self._normalize = normalize
        self._normalization_mean = np.array(normalization_mean, dtype=np.float32)
        self._normalization_std = np.array(normalization_std, dtype=np.float32)
        self._bgr_to_rgb = bgr_to_rgb

        # Statistics
        self._crop_count = 0
        self._boundary_clips = 0

        logger.info(
            f"HandCropper initialized: "
            f"target_size={target_size}, "
            f"padding={padding}, "
            f"square_crop={square_crop}, "
            f"normalize={normalize}"
        )

    def crop_hand(
        self,
        frame: np.ndarray,
        detection: HandDetectionResult,
        apply_preprocessing: bool = True,
    ) -> HandROI:
        """
        Crop a hand region from a frame.

        Args:
            frame: Input frame (BGR format from OpenCV)
            detection: Hand detection result with bounding box
            apply_preprocessing: Whether to apply preprocessing (resize, normalize, etc.)

        Returns:
            HandROI object containing the cropped and processed hand region
        """
        if frame is None or frame.size == 0:
            raise ValueError("Empty frame provided to crop_hand()")

        self._crop_count += 1
        frame_height, frame_width = frame.shape[:2]

        # Extract and expand bounding box
        x, y, w, h = detection.bbox
        x_padded, y_padded, w_padded, h_padded = self._add_padding(
            x, y, w, h, frame_width, frame_height
        )

        # Make square if requested
        if self._square_crop:
            x_padded, y_padded, w_padded, h_padded = self._make_square(
                x_padded, y_padded, w_padded, h_padded, frame_width, frame_height
            )

        # Clip to frame boundaries
        x_clipped, y_clipped, w_clipped, h_clipped = self._clip_to_frame(
            x_padded, y_padded, w_padded, h_padded, frame_width, frame_height
        )

        if (
            x_clipped != x_padded
            or y_clipped != y_padded
            or w_clipped != w_padded
            or h_clipped != h_padded
        ):
            self._boundary_clips += 1
            logger.debug(
                f"Bounding box clipped to frame boundaries: "
                f"({x_padded}, {y_padded}, {w_padded}, {h_padded}) -> "
                f"({x_clipped}, {y_clipped}, {w_clipped}, {h_clipped})"
            )

        # Crop the region
        cropped = frame[
            y_clipped : y_clipped + h_clipped, x_clipped : x_clipped + w_clipped
        ]

        if cropped.size == 0:
            raise ValueError(
                f"Empty crop result for bbox ({x_clipped}, {y_clipped}, "
                f"{w_clipped}, {h_clipped}) in frame {frame_width}×{frame_height}"
            )

        # Apply preprocessing if requested
        if apply_preprocessing:
            processed = self._preprocess(cropped)
            preprocessing_params = {
                "target_size": self._target_size,
                "normalized": self._normalize,
                "bgr_to_rgb": self._bgr_to_rgb,
                "normalization_mean": self._normalization_mean.tolist(),
                "normalization_std": self._normalization_std.tolist(),
            }
        else:
            processed = cropped
            preprocessing_params = {"raw": True}

        # Create HandROI
        hand_roi = HandROI(
            image=processed,
            detection=detection,
            preprocessing_params=preprocessing_params,
        )

        return hand_roi

    def crop_multiple(
        self,
        frame: np.ndarray,
        detections: list[HandDetectionResult],
        apply_preprocessing: bool = True,
    ) -> list[HandROI]:
        """
        Crop multiple hand regions from a single frame.

        Args:
            frame: Input frame (BGR format)
            detections: List of hand detections
            apply_preprocessing: Whether to apply preprocessing

        Returns:
            List of HandROI objects, one per detection
        """
        if not detections:
            return []

        hand_rois = []
        for detection in detections:
            try:
                hand_roi = self.crop_hand(frame, detection, apply_preprocessing)
                hand_rois.append(hand_roi)
            except Exception as e:
                logger.error(
                    f"Failed to crop hand for detection {detection}: {e}", exc_info=True
                )

        return hand_rois

    def _add_padding(
        self,
        x: int,
        y: int,
        w: int,
        h: int,
        frame_width: int,
        frame_height: int,
    ) -> Tuple[int, int, int, int]:
        """
        Add padding around a bounding box.

        Args:
            x, y, w, h: Original bounding box
            frame_width, frame_height: Frame dimensions

        Returns:
            Padded bounding box (x, y, w, h)
        """
        # Calculate padding in pixels
        pad_w = int(w * self._padding)
        pad_h = int(h * self._padding)

        # Apply padding (note: may extend outside frame bounds)
        x_padded = x - pad_w
        y_padded = y - pad_h
        w_padded = w + 2 * pad_w
        h_padded = h + 2 * pad_h

        return (x_padded, y_padded, w_padded, h_padded)

    def _make_square(
        self,
        x: int,
        y: int,
        w: int,
        h: int,
        frame_width: int,
        frame_height: int,
    ) -> Tuple[int, int, int, int]:
        """
        Convert a bounding box to square aspect ratio.

        Args:
            x, y, w, h: Input bounding box
            frame_width, frame_height: Frame dimensions

        Returns:
            Square bounding box (x, y, size, size)
        """
        # Use the larger dimension
        size = max(w, h)

        # Center the square on the original box
        center_x = x + w // 2
        center_y = y + h // 2

        x_square = center_x - size // 2
        y_square = center_y - size // 2

        return (x_square, y_square, size, size)

    def _clip_to_frame(
        self,
        x: int,
        y: int,
        w: int,
        h: int,
        frame_width: int,
        frame_height: int,
    ) -> Tuple[int, int, int, int]:
        """
        Clip a bounding box to frame boundaries.

        Args:
            x, y, w, h: Input bounding box
            frame_width, frame_height: Frame dimensions

        Returns:
            Clipped bounding box (x, y, w, h)
        """
        # Clip coordinates
        x_clipped = max(0, x)
        y_clipped = max(0, y)

        # Adjust width and height
        x_end = min(x + w, frame_width)
        y_end = min(y + h, frame_height)

        w_clipped = x_end - x_clipped
        h_clipped = y_end - y_clipped

        return (x_clipped, y_clipped, w_clipped, h_clipped)

    def _preprocess(self, image: np.ndarray) -> np.ndarray:
        """
        Apply preprocessing to a cropped hand region.

        Args:
            image: Cropped hand region (BGR format)

        Returns:
            Preprocessed image
        """
        # Resize to target size
        processed = cv2.resize(image, self._target_size, interpolation=cv2.INTER_LINEAR)

        # Convert BGR to RGB if requested
        if self._bgr_to_rgb:
            processed = cv2.cvtColor(processed, cv2.COLOR_BGR2RGB)

        # Normalize if requested
        if self._normalize:
            # Convert to float32 and scale to [0, 1]
            processed = processed.astype(np.float32) / 255.0

            # Apply Z-score normalization
            processed = (processed - self._normalization_mean) / self._normalization_std

        return processed

    def get_stats(self) -> dict:
        """
        Get cropping statistics.

        Returns:
            Dictionary with statistics
        """
        return {
            "crops_performed": self._crop_count,
            "boundary_clips": self._boundary_clips,
            "clip_rate": (
                self._boundary_clips / self._crop_count if self._crop_count > 0 else 0
            ),
            "target_size": self._target_size,
            "padding": self._padding,
            "square_crop": self._square_crop,
        }

    def reset_stats(self):
        """Reset statistics."""
        self._crop_count = 0
        self._boundary_clips = 0
        logger.debug("HandCropper statistics reset")

    def __repr__(self) -> str:
        """String representation for debugging."""
        return (
            f"HandCropper("
            f"target_size={self._target_size}, "
            f"padding={self._padding}, "
            f"square={self._square_crop})"
        )
