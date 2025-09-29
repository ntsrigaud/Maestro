"""
Comprehensive unit test suite for Python Input Manager implementation.

This test suite covers all aspects of the Python Input Manager including:
- Frame processing pipeline
- Threading and concurrency
- MediaPipe integration
- Error handling and edge cases
- API compatibility with C++ implementation
- Performance characteristics
"""

import pytest
import numpy as np
import cv2
import time
import threading
from unittest.mock import Mock, patch, MagicMock
from pathlib import Path
import sys

# Add module path
sys.path.insert(0, str(Path(__file__).parent))

from python_input_manager import PythonInputManager, InputManager, FrameData


class TestFrameData:
    """Test FrameData container class."""
    
    def test_frame_data_creation(self):
        """Test FrameData creation and basic properties."""
        frame = np.random.rand(224, 224, 3).astype(np.float32)
        timestamp = time.time()
        
        frame_data = FrameData(frame=frame, timestamp=timestamp)
        
        assert frame_data.frame.shape == (224, 224, 3)
        assert frame_data.timestamp == timestamp
        assert frame_data.hands_detected is None
        assert isinstance(frame_data.metadata, dict)
    
    def test_get_frame_method(self):
        """Test get_frame method for API compatibility."""
        frame = np.random.rand(224, 224, 3).astype(np.float32)
        frame_data = FrameData(frame=frame, timestamp=time.time())
        
        returned_frame = frame_data.get_frame()
        np.testing.assert_array_equal(returned_frame, frame)
    
    def test_frame_data_with_hands(self):
        """Test FrameData with hand detection results."""
        frame = np.random.rand(224, 224, 3).astype(np.float32)
        hands_data = [
            {'landmarks': [(0.5, 0.5, 0.0)], 'handedness': 'Right'}
        ]
        
        frame_data = FrameData(
            frame=frame,
            timestamp=time.time(),
            hands_detected=hands_data
        )
        
        assert len(frame_data.hands_detected) == 1
        assert frame_data.hands_detected[0]['handedness'] == 'Right'


class TestPythonInputManager:
    """Test PythonInputManager core functionality."""
    
    def setup_method(self):
        """Set up test fixtures before each test method."""
        self.input_manager = None
    
    def teardown_method(self):
        """Clean up after each test method."""
        if self.input_manager:
            self.input_manager.stop_continuous_capture()
    
    def test_initialization_default_params(self):
        """Test InputManager initialization with default parameters."""
        input_manager = PythonInputManager()
        
        assert input_manager.camera_index == 0
        assert input_manager.target_size == (224, 224)
        assert input_manager.mirror_enabled is True
        assert input_manager.normalization_params['mean'] == 0.5
        assert input_manager.normalization_params['std'] == 0.5
        assert input_manager.enable_mediapipe is True
        assert input_manager.max_queue_size == 30
    
    def test_initialization_custom_params(self):
        """Test InputManager initialization with custom parameters."""
        input_manager = PythonInputManager(
            camera_index=1,
            target_size=(256, 256),
            enable_mediapipe=False,
            max_queue_size=50
        )
        
        assert input_manager.camera_index == 1
        assert input_manager.target_size == (256, 256)
        assert input_manager.enable_mediapipe is False
        assert input_manager.max_queue_size == 50
    
    @patch('cv2.VideoCapture')
    def test_start_capture_success(self, mock_video_capture):
        """Test successful camera capture start."""
        mock_cap = Mock()
        mock_cap.isOpened.return_value = True
        mock_video_capture.return_value = mock_cap
        
        input_manager = PythonInputManager()
        result = input_manager.start_capture()
        
        assert result is True
        assert input_manager.is_capture_open() is True
        mock_cap.set.assert_called()  # Camera properties should be set
    
    @patch('cv2.VideoCapture')
    def test_start_capture_failure(self, mock_video_capture):
        """Test camera capture start failure."""
        mock_cap = Mock()
        mock_cap.isOpened.return_value = False
        mock_video_capture.return_value = mock_cap
        
        input_manager = PythonInputManager()
        result = input_manager.start_capture()
        
        assert result is False
        assert input_manager.is_capture_open() is False
    
    @patch('cv2.VideoCapture')
    def test_stop_capture(self, mock_video_capture):
        """Test camera capture stop."""
        mock_cap = Mock()
        mock_cap.isOpened.return_value = True
        mock_video_capture.return_value = mock_cap
        
        input_manager = PythonInputManager()
        input_manager.start_capture()
        input_manager.stop_capture()
        
        mock_cap.release.assert_called_once()
        assert input_manager.cap is None
    
    def test_frame_processing_pipeline(self):
        """Test the frame preprocessing pipeline."""
        input_manager = PythonInputManager(target_size=(224, 224))
        
        # Create test frame (simulate camera BGR format)
        test_frame = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
        
        processed_frame = input_manager._process_frame(test_frame)
        
        assert processed_frame is not None
        assert processed_frame.shape == (224, 224, 3)
        assert processed_frame.dtype == np.float32
        # Check normalization (values should be around (0-1 - 0.5) / 0.5 = [-1, 1])
        assert -2.0 <= processed_frame.min() <= 2.0
        assert -2.0 <= processed_frame.max() <= 2.0
    
    def test_frame_processing_with_mirroring(self):
        """Test frame processing with mirroring enabled."""
        input_manager = PythonInputManager()
        input_manager.set_mirror_enabled(True)
        
        # Create asymmetric test frame to check mirroring
        test_frame = np.zeros((100, 200, 3), dtype=np.uint8)
        test_frame[:, :100, :] = 255  # Left half white
        
        processed_frame = input_manager._process_frame(test_frame)
        
        assert processed_frame is not None
        # After mirroring, the right side should be brighter (originally left side)
        left_mean = np.mean(processed_frame[:, :112, :])  # Left half of 224
        right_mean = np.mean(processed_frame[:, 112:, :])  # Right half of 224
        assert right_mean > left_mean
    
    def test_frame_processing_without_mirroring(self):
        """Test frame processing with mirroring disabled."""
        input_manager = PythonInputManager()
        input_manager.set_mirror_enabled(False)
        
        # Create asymmetric test frame
        test_frame = np.zeros((100, 200, 3), dtype=np.uint8)
        test_frame[:, :100, :] = 255  # Left half white
        
        processed_frame = input_manager._process_frame(test_frame)
        
        assert processed_frame is not None
        # Without mirroring, left side should be brighter
        left_mean = np.mean(processed_frame[:, :112, :])
        right_mean = np.mean(processed_frame[:, 112:, :])
        assert left_mean > right_mean
    
    def test_normalization_parameters(self):
        """Test custom normalization parameter setting."""
        input_manager = PythonInputManager()
        input_manager.set_normalization_params(mean=0.485, std=0.229)
        
        assert input_manager.normalization_params['mean'] == 0.485
        assert input_manager.normalization_params['std'] == 0.229
        
        # Test with actual frame
        test_frame = np.ones((100, 100, 3), dtype=np.uint8) * 128  # Mid-gray
        processed_frame = input_manager._process_frame(test_frame)
        
        # Expected: (0.5 - 0.485) / 0.229 ≈ 0.065
        expected_value = (0.5 - 0.485) / 0.229
        np.testing.assert_allclose(processed_frame, expected_value, rtol=0.1)
    
    def test_performance_metrics(self):
        """Test performance metrics collection."""
        input_manager = PythonInputManager()
        
        # Simulate some processing times
        input_manager.processing_times = [0.01, 0.015, 0.012, 0.008]
        input_manager.frame_count = 4
        
        metrics = input_manager.get_performance_metrics()
        
        assert 'frame_count' in metrics
        assert 'avg_processing_time_ms' in metrics
        assert 'fps' in metrics
        assert 'mediapipe_enabled' in metrics
        
        assert metrics['frame_count'] == 4
        assert metrics['avg_processing_time_ms'] > 0
        assert metrics['fps'] > 0
    
    @patch('python_input_manager.mp.solutions.hands.Hands')
    def test_mediapipe_integration_disabled(self, mock_hands):
        """Test MediaPipe integration when disabled."""
        input_manager = PythonInputManager(enable_mediapipe=False)
        
        assert input_manager.hands is None
        assert input_manager.enable_mediapipe is False
        mock_hands.assert_not_called()
    
    @patch('python_input_manager.mp.solutions.hands.Hands')
    def test_mediapipe_integration_enabled(self, mock_hands):
        """Test MediaPipe integration when enabled."""
        mock_hands_instance = Mock()
        mock_hands.return_value = mock_hands_instance
        
        input_manager = PythonInputManager(enable_mediapipe=True)
        
        assert input_manager.hands == mock_hands_instance
        mock_hands.assert_called_once()
    
    def test_hand_detection_no_hands(self):
        """Test hand detection with no hands in frame."""
        input_manager = PythonInputManager(enable_mediapipe=False)
        input_manager.hands = Mock()
        input_manager.hands.process.return_value.multi_hand_landmarks = None
        
        test_frame = np.zeros((100, 100, 3), dtype=np.uint8)
        result = input_manager._detect_hands(test_frame)
        
        assert result is None
    
    def test_hand_detection_with_hands(self):
        """Test hand detection with hands detected."""
        input_manager = PythonInputManager(enable_mediapipe=False)
        
        # Mock MediaPipe results
        mock_landmark = Mock()
        mock_landmark.x, mock_landmark.y, mock_landmark.z = 0.5, 0.5, 0.0
        
        mock_hand_landmarks = Mock()
        mock_hand_landmarks.landmark = [mock_landmark]
        
        mock_handedness = Mock()
        mock_handedness.classification = [Mock()]
        mock_handedness.classification[0].label = "Right"
        
        mock_results = Mock()
        mock_results.multi_hand_landmarks = [mock_hand_landmarks]
        mock_results.multi_handedness = [mock_handedness]
        
        input_manager.hands = Mock()
        input_manager.hands.process.return_value = mock_results
        
        test_frame = np.zeros((100, 100, 3), dtype=np.uint8)
        result = input_manager._detect_hands(test_frame)
        
        assert result is not None
        assert len(result) == 1
        assert result[0]['handedness'] == "Right"
        assert len(result[0]['landmarks']) == 1
    
    def test_context_manager(self):
        """Test context manager functionality."""
        with patch('cv2.VideoCapture') as mock_video_capture:
            mock_cap = Mock()
            mock_cap.isOpened.return_value = True
            mock_video_capture.return_value = mock_cap
            
            with PythonInputManager() as input_manager:
                assert input_manager is not None
            
            # Should have stopped capture on exit
            # Note: We can't easily test this without actual camera integration


class TestInputManagerCompatibility:
    """Test the InputManager compatibility wrapper."""
    
    def test_initialization(self):
        """Test compatibility wrapper initialization."""
        wrapper = InputManager(camera_index=1)
        
        assert wrapper._python_manager.camera_index == 1
        assert wrapper._python_manager.enable_mediapipe is False  # Disabled in compatibility mode
    
    @patch('cv2.VideoCapture')
    def test_api_compatibility(self, mock_video_capture):
        """Test that all C++ API methods are available."""
        mock_cap = Mock()
        mock_cap.isOpened.return_value = True
        mock_video_capture.return_value = mock_cap
        
        wrapper = InputManager()
        
        # Test all API methods exist and work
        assert hasattr(wrapper, 'start_capture')
        assert hasattr(wrapper, 'stop_capture')
        assert hasattr(wrapper, 'is_capture_open')
        assert hasattr(wrapper, 'start_continuous_capture')
        assert hasattr(wrapper, 'stop_continuous_capture')
        assert hasattr(wrapper, 'get_latest_frame')
        assert hasattr(wrapper, 'get_all_frames')
        assert hasattr(wrapper, 'set_normalization_params')
        assert hasattr(wrapper, 'set_mirror_enabled')
        
        # Test method calls work
        assert wrapper.start_capture() is True
        assert wrapper.is_capture_open() is True
        wrapper.set_mirror_enabled(False)
        wrapper.set_normalization_params(0.3, 0.7)
        wrapper.stop_capture()


class TestThreadingAndConcurrency:
    """Test threading and concurrency aspects."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.input_manager = None
    
    def teardown_method(self):
        """Clean up after tests."""
        if self.input_manager:
            self.input_manager.stop_continuous_capture()
    
    @patch('cv2.VideoCapture')
    def test_continuous_capture_threading(self, mock_video_capture):
        """Test that continuous capture runs in separate thread."""
        mock_cap = Mock()
        mock_cap.isOpened.return_value = True
        mock_cap.read.return_value = (True, np.zeros((480, 640, 3), dtype=np.uint8))
        mock_video_capture.return_value = mock_cap
        
        self.input_manager = PythonInputManager()
        
        assert self.input_manager.capture_thread is None
        
        result = self.input_manager.start_continuous_capture()
        assert result is True
        
        # Give thread time to start
        time.sleep(0.1)
        
        assert self.input_manager.capture_thread is not None
        assert self.input_manager.capture_thread.is_alive()
        assert self.input_manager.is_capturing.is_set()
        
        self.input_manager.stop_continuous_capture()
        
        # Give thread time to stop
        time.sleep(0.1)
        
        assert not self.input_manager.is_capturing.is_set()
    
    @patch('cv2.VideoCapture')
    def test_thread_safety_frame_access(self, mock_video_capture):
        """Test thread-safe access to latest frames."""
        mock_cap = Mock()
        mock_cap.isOpened.return_value = True
        mock_cap.read.return_value = (True, np.zeros((480, 640, 3), dtype=np.uint8))
        mock_video_capture.return_value = mock_cap
        
        self.input_manager = PythonInputManager()
        self.input_manager.start_continuous_capture()
        
        # Test concurrent access from multiple threads
        results = []
        errors = []
        
        def access_frames():
            try:
                for _ in range(10):
                    frame = self.input_manager.get_latest_frame()
                    results.append(frame)
                    time.sleep(0.001)
            except Exception as e:
                errors.append(e)
        
        threads = [threading.Thread(target=access_frames) for _ in range(3)]
        
        for thread in threads:
            thread.start()
        
        for thread in threads:
            thread.join()
        
        assert len(errors) == 0  # No threading errors
        
        self.input_manager.stop_continuous_capture()


class TestErrorHandling:
    """Test error handling and edge cases."""
    
    def test_invalid_camera_index(self):
        """Test handling of invalid camera index."""
        with patch('cv2.VideoCapture') as mock_video_capture:
            mock_cap = Mock()
            mock_cap.isOpened.return_value = False
            mock_video_capture.return_value = mock_cap
            
            input_manager = PythonInputManager(camera_index=999)
            result = input_manager.start_capture()
            
            assert result is False
            assert not input_manager.is_capture_open()
    
    def test_frame_processing_error_handling(self):
        """Test error handling in frame processing."""
        input_manager = PythonInputManager()
        
        # Test with None frame
        result = input_manager._process_frame(None)
        assert result is None
        
        # Test with invalid frame shape
        invalid_frame = np.array([1, 2, 3])
        result = input_manager._process_frame(invalid_frame)
        assert result is None
    
    def test_mediapipe_error_handling(self):
        """Test MediaPipe error handling."""
        input_manager = PythonInputManager(enable_mediapipe=False)
        
        # Test with no hands initialized
        result = input_manager._detect_hands(np.zeros((100, 100, 3)))
        assert result is None
        
        # Test with hands but processing error
        input_manager.hands = Mock()
        input_manager.hands.process.side_effect = Exception("MediaPipe error")
        
        result = input_manager._detect_hands(np.zeros((100, 100, 3)))
        assert result is None
    
    def test_queue_overflow_handling(self):
        """Test queue overflow handling."""
        input_manager = PythonInputManager(max_queue_size=2)
        
        # Fill queue beyond capacity
        for i in range(5):
            frame_data = FrameData(
                frame=np.zeros((224, 224, 3)),
                timestamp=time.time()
            )
            if not input_manager.frame_queue.full():
                input_manager.frame_queue.put(frame_data)
        
        # Should not exceed max size
        assert input_manager.frame_queue.qsize() <= 2


if __name__ == "__main__":
    pytest.main([__file__, "-v"])