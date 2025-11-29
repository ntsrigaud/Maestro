"""
Pure Python implementation of InputManager for cross-platform compatibility.

This module provides a high-performance interface to capture and preprocess
video frames from a camera without requiring C++ compilation.
"""

import cv2
import numpy as np
import threading
import time
from dataclasses import dataclass
from typing import Optional, List, Tuple
from collections import deque


@dataclass
class FrameData:
    """Container for frame data with metadata."""

    frame: np.ndarray
    timestamp: int  # milliseconds since epoch
    frame_number: int = 0

    def to_numpy(self) -> np.ndarray:
        """Convert frame to numpy array (for API compatibility)."""
        return self.frame


class InputManager:
    """
    Manages video capture from camera with preprocessing capabilities.

    Features:
    - Thread-safe continuous capture
    - Frame buffering with configurable size
    - Automatic mirror detection and correction
    - Frame preprocessing (resize, normalize, color conversion)
    - Cross-platform support (Linux, macOS, Windows)

    Example:
        >>> manager = InputManager(camera_index=0)
        >>> manager.start_capture()
        >>> frame_data = manager.get_latest_frame()
        >>> frame = frame_data.to_numpy()
        >>> manager.stop_capture()
    """

    def __init__(
        self,
        camera_index: int = 0,
        buffer_size: int = 30,
        target_fps: int = 30,
        width: Optional[int] = None,
        height: Optional[int] = None,
    ):
        """
        Initialize InputManager.

        Args:
            camera_index: Camera device index (default: 0 for built-in camera)
            buffer_size: Number of frames to keep in buffer (default: 30)
            target_fps: Target frame rate for capture (default: 30)
            width: Target frame width (None = use camera default)
            height: Target frame height (None = use camera default)
        """
        self._camera_index = camera_index
        self._buffer_size = buffer_size
        self._target_fps = target_fps
        self._width = width
        self._height = height

        # Capture state
        self._capture: Optional[cv2.VideoCapture] = None
        self._is_capturing = False
        self._capture_thread: Optional[threading.Thread] = None

        # Frame storage
        self._frame_buffer: deque = deque(maxlen=buffer_size)
        self._latest_frame: Optional[FrameData] = None
        self._frame_lock = threading.Lock()
        self._frame_counter = 0

        # Preprocessing settings
        self._mirror_enabled = True
        self._auto_detect_mirror = True
        self._convert_to_rgb = True
        self._normalization_mean = 0.0
        self._normalization_std = 1.0
        self._target_size: Optional[Tuple[int, int]] = None

    def is_capture_open(self) -> bool:
        """Check if camera is open and ready."""
        return self._capture is not None and self._capture.isOpened()

    def start_capture(self) -> bool:
        """
        Open camera device.

        Returns:
            True if camera opened successfully, False otherwise
        """
        if self._capture is not None:
            return True

        self._capture = cv2.VideoCapture(self._camera_index)

        if not self._capture.isOpened():
            print(f"Failed to open camera {self._camera_index}")
            return False

        # Configure camera properties
        if self._width is not None:
            self._capture.set(cv2.CAP_PROP_FRAME_WIDTH, self._width)
        if self._height is not None:
            self._capture.set(cv2.CAP_PROP_FRAME_HEIGHT, self._height)
        if self._target_fps is not None:
            self._capture.set(cv2.CAP_PROP_FPS, self._target_fps)

        # Auto-detect if camera frames are already mirrored
        if self._auto_detect_mirror:
            self._detect_mirror_state()

        print(f"Camera {self._camera_index} opened successfully")
        return True

    def stop_capture(self):
        """Release camera resources."""
        self.stop_continuous_capture()

        if self._capture is not None:
            self._capture.release()
            self._capture = None
            print("Camera released")

    def start_continuous_capture(self) -> bool:
        """
        Start continuous frame capture in background thread.

        Returns:
            True if started successfully, False otherwise
        """
        if not self.is_capture_open():
            if not self.start_capture():
                return False

        if self._is_capturing:
            print("Warning: Continuous capture already running")
            return True

        self._is_capturing = True
        self._capture_thread = threading.Thread(target=self._capture_loop, daemon=True)
        self._capture_thread.start()

        # Wait for first frame to be captured (max 1 second)
        for _ in range(100):  # Try for 1 second (100 * 10ms)
            if self._latest_frame is not None:
                break
            time.sleep(0.01)

        if self._latest_frame is None:
            print("Warning: No frame captured yet after 1 second")

        print("Continuous capture started")
        return True

    def stop_continuous_capture(self):
        """Stop continuous frame capture."""
        if not self._is_capturing:
            return

        self._is_capturing = False

        if self._capture_thread is not None:
            self._capture_thread.join(timeout=2.0)
            self._capture_thread = None

        print("Continuous capture stopped")

    def get_latest_frame(self) -> Optional[FrameData]:
        """
        Get the most recent captured frame.

        Returns:
            FrameData object with latest frame, or None if no frame available
        """
        with self._frame_lock:
            return self._latest_frame

    def get_all_frames(self) -> List[FrameData]:
        """
        Get all frames currently in buffer.

        Returns:
            List of FrameData objects
        """
        with self._frame_lock:
            return list(self._frame_buffer)

    def get_frame(self) -> Optional[np.ndarray]:
        """
        Capture and return a single frame (synchronous).

        Returns:
            Preprocessed frame as numpy array, or None on failure
        """
        if not self.is_capture_open():
            return None

        ret, frame = self._capture.read()
        if not ret:
            return None

        return self._preprocess_frame(frame)

    # Configuration methods

    def set_mirror_enabled(self, enabled: bool):
        """Enable or disable frame mirroring."""
        self._mirror_enabled = enabled

    def set_auto_detect_mirror(self, enabled: bool):
        """Enable or disable automatic mirror detection."""
        self._auto_detect_mirror = enabled

    def set_normalization_params(self, mean: float, std: float):
        """
        Set normalization parameters for preprocessing.

        Args:
            mean: Mean value for normalization
            std: Standard deviation for normalization (must be > 0)

        Raises:
            ValueError: If std is zero or negative
        """
        if std <= 0:
            raise ValueError(f"Standard deviation must be greater than 0, got {std}")
        self._normalization_mean = mean
        self._normalization_std = std

    def set_target_size(self, width: int, height: int):
        """Set target size for frame resizing."""
        self._target_size = (width, height)

    def set_rgb_conversion(self, enabled: bool):
        """Enable or disable BGR to RGB conversion."""
        self._convert_to_rgb = enabled

    # Private methods

    def _capture_loop(self):
        """Background thread for continuous frame capture."""
        frame_time = 1.0 / self._target_fps if self._target_fps > 0 else 0

        while self._is_capturing:
            start_time = time.time()

            ret, frame = self._capture.read()
            if not ret:
                print("Warning: Failed to read frame")
                time.sleep(0.1)
                continue

            # Preprocess frame
            processed_frame = self._preprocess_frame(frame)
            if processed_frame is None:
                continue

            # Create frame data
            frame_data = FrameData(
                frame=processed_frame,
                timestamp=int(time.time_ns()),
                frame_number=self._frame_counter,
            )
            self._frame_counter += 1

            # Update buffer and latest frame (thread-safe)
            with self._frame_lock:
                self._frame_buffer.append(frame_data)
                self._latest_frame = frame_data

            # Maintain target frame rate
            elapsed = time.time() - start_time
            sleep_time = max(0, frame_time - elapsed)
            if sleep_time > 0:
                time.sleep(sleep_time)

    def _preprocess_frame(self, frame: np.ndarray) -> Optional[np.ndarray]:
        """
        Apply preprocessing pipeline to frame.

        Args:
            frame: Raw frame from camera

        Returns:
            Preprocessed frame
        """
        if frame is None:
            return None

        # 1. Mirror (flip horizontally)
        if self._mirror_enabled:
            frame = cv2.flip(frame, 1)

        # 2. Convert BGR to RGB (OpenCV uses BGR by default)
        if self._convert_to_rgb:
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        # 3. Resize if target size specified
        if self._target_size is not None:
            frame = cv2.resize(frame, self._target_size)

        # 4. Normalize
        if self._normalization_std != 1.0 or self._normalization_mean != 0.0:
            frame = frame.astype(np.float32)
            frame = (frame - self._normalization_mean) / self._normalization_std

        return frame

    def _detect_mirror_state(self):
        """
        Auto-detect if camera frames are already mirrored.

        This is a heuristic - some cameras (especially MacBook built-in)
        already provide mirrored frames.
        """
        # This is a placeholder - you could implement actual detection
        # by analyzing text or known asymmetric features
        # For now, assume most webcams need mirroring
        self._mirror_enabled = True

    def __enter__(self):
        """Context manager entry."""
        self.start_capture()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit."""
        self.stop_capture()

    def __del__(self):
        """Destructor - ensure camera is released."""
        self.stop_capture()

    def get_camera_info(self) -> dict:
        """
        Get information about the camera.

        Returns:
            Dictionary with camera properties
        """
        if not self.is_capture_open():
            return {}

        return {
            "width": int(self._capture.get(cv2.CAP_PROP_FRAME_WIDTH)),
            "height": int(self._capture.get(cv2.CAP_PROP_FRAME_HEIGHT)),
            "fps": int(self._capture.get(cv2.CAP_PROP_FPS)),
            "backend": self._capture.getBackendName(),
            "is_capturing": self._is_capturing,
            "buffer_size": len(self._frame_buffer),
            "frames_captured": self._frame_counter,
        }
