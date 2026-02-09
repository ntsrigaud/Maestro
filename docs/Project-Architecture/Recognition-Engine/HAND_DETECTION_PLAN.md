# Hand Detection Submodule - Implementation Plan

## Overview

The Hand Detection Submodule is the first component of the Recognition Engine, responsible for localizing hands in video frames and extracting hand regions of interest (ROI) for downstream gesture classification.

## Architecture

```
recognition_engine/
├── __init__.py
├── hand_detection/
│   ├── __init__.py
│   ├── hand_detector.py        # MediaPipe Hands integration
│   ├── hand_data.py             # Data structures for hand detection results
│   └── README.md
├── gesture_classification/      # Future: ViT + LSTM (not implemented yet)
└── README.md
```

## Phase 1: Core Hand Detection with MediaPipe

### Objectives

1. [x] Set up MediaPipe Hands integration
2. [x] Implement hand localization in frames
3. [x] Extract bounding boxes for detected hands
4. [x] Multi-hand support (detect multiple hands)
5. [x] Confidence scoring for detection quality

### Key Components

#### 1.1 HandDetectionResult Data Structure

```python
@dataclass
class HandDetectionResult:
    """Container for hand detection results."""
    bbox: Tuple[int, int, int, int]  # (x, y, width, height)
    landmarks: np.ndarray             # 21 hand landmarks (x, y, z)
    handedness: str                   # "Left" or "Right"
    confidence: float                 # Detection confidence [0, 1]
    timestamp: int                    # Nanoseconds since epoch
```

#### 1.2 HandDetector Class

```python
class HandDetector:
    def __init__(
        self,
        max_num_hands: int = 2,
        min_detection_confidence: float = 0.5,
        min_tracking_confidence: float = 0.5,
        model_complexity: int = 1
    ):
        # Initialize MediaPipe Hands

    def detect(self, frame: np.ndarray) -> List[HandDetectionResult]:
        # Detect hands in a single frame

    def detect_batch(self, frames: List[np.ndarray]) -> List[List[HandDetectionResult]]:
        # Process multiple frames efficiently
```

### MediaPipe Hands Configuration

- **Model Complexity**: 0 (lite), 1 (full) - balance accuracy vs speed
- **Max Hands**: 2 (support two-hand gestures)
- **Detection Confidence**: 0.5 (adjustable threshold)
- **Tracking Confidence**: 0.5 (for video stream continuity)
- **Static Image Mode**: False (optimized for video)

### Implementation Steps

**Step 1.1**: Install MediaPipe dependency

```bash
pip install mediapipe>=0.10.0
```

**Step 1.2**: Create HandDetectionResult dataclass

- Store bounding box coordinates
- Store 21 hand landmarks (MediaPipe format)
- Store handedness (Left/Right)
- Store confidence scores
- Store timestamp for temporal tracking

**Step 1.3**: Implement HandDetector class

- Initialize MediaPipe Hands solution
- Implement `detect()` for single frame processing
- Handle BGR to RGB conversion (OpenCV → MediaPipe)
- Calculate bounding boxes from landmarks
- Extract confidence scores

**Step 1.4**: Add multi-hand support

- Process all detected hands in frame
- Return list of HandDetectionResult objects
- Handle hand tracking across frames

## Phase 2: Hand Region Cropping

### Objectives

1. [x] Extract hand ROI using bounding boxes
2. [x] Add padding for context preservation
3. [x] Normalize dimensions for consistent output
4. [x] Handle edge cases (hands near frame boundaries)

### Key Components

#### 2.1 HandCropper Class

```python
class HandCropper:
    def __init__(
        self,
        target_size: Tuple[int, int] = (224, 224),
        padding_ratio: float = 0.2,
        maintain_aspect_ratio: bool = True
    ):
        # Configuration for cropping

    def crop_hand(
        self,
        frame: np.ndarray,
        detection: HandDetectionResult
    ) -> np.ndarray:
        # Crop and preprocess single hand

    def crop_hands(
        self,
        frame: np.ndarray,
        detections: List[HandDetectionResult]
    ) -> List[np.ndarray]:
        # Crop multiple hands from frame
```

### Cropping Strategy

- **Bounding Box**: Calculate from landmark extrema
- **Padding**: Add 20% padding to include wrist and context
- **Aspect Ratio**: Maintain square aspect ratio (1:1)
- **Target Size**: Resize to 224×224 (ViT standard input)
- **Edge Handling**: Clip coordinates to frame boundaries

### Implementation Steps

**Step 2.1**: Implement bounding box calculation

- Find min/max x, y from 21 landmarks
- Add configurable padding
- Ensure square bounding box

**Step 2.2**: Implement ROI extraction

- Crop region from original frame
- Handle boundary conditions
- Preserve aspect ratio

**Step 2.3**: Implement preprocessing pipeline

- Resize to target dimensions (224×224)
- Convert color space if needed
- Normalize pixel values (optional for now)

## Phase 3: Image Preprocessing

### Objectives

1. [x] Resize cropped hands to standard size
2. [x] Normalize pixel values for model input
3. [x] Optional augmentations for robustness
4. [x] Consistent preprocessing pipeline

### Preprocessing Pipeline

```python
def preprocess_hand_roi(roi: np.ndarray) -> np.ndarray:
    # Resize to 224×224 (ViT input size)
    resized = cv2.resize(roi, (224, 224))

    # Convert BGR to RGB (if needed by model)
    rgb = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB)

    # Normalize to [0, 1] or [-1, 1] (model-dependent)
    normalized = rgb.astype(np.float32) / 255.0

    return normalized
```

### Normalization Strategies

- **[0, 1]**: Simple division by 255
- **[-1, 1]**: (pixel / 127.5) - 1
- **ImageNet Stats**: mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]

## Phase 4: Real-Time Performance Optimization

### Objectives

1. [x] Measure baseline performance (FPS, latency)
2. [x] Optimize MediaPipe configuration
3. [x] Implement frame skipping if needed
4. [x] Profile bottlenecks

### Performance Targets

- **Detection Rate**: 30+ FPS on CPU
- **Latency**: <30ms per frame
- **CPU Usage**: <50% on modern laptop CPU
- **Memory**: <500MB RAM overhead

### Optimization Strategies

**4.1 MediaPipe Optimization**

- Use `model_complexity=0` (lite model) for speed
- Reduce `max_num_hands` if single-hand only
- Enable static image mode for discrete gestures

**4.2 Processing Optimization**

- Process at lower resolution (640×480 vs 1080p)
- Skip frames if detection is stable (tracking mode)
- Use numpy operations efficiently
- Avoid unnecessary copies

**4.3 Multi-threading**

- Separate detection thread from main capture
- Use queue for frame buffering
- Parallel processing of multiple hands

### Benchmarking Tools

```python
class PerformanceMonitor:
    def measure_fps(self) -> float:
        # Calculate actual FPS

    def measure_latency(self) -> float:
        # Measure end-to-end latency

    def profile_bottlenecks(self) -> Dict[str, float]:
        # Identify slow operations
```

## Phase 5: Integration & Testing

### 5.1 Integration with Input Manager

```python
from input_manager import InputManager, FrameData
from recognition_engine.hand_detection import HandDetector, HandCropper

# Setup
input_mgr = InputManager(camera_index=0)
hand_detector = HandDetector(max_num_hands=2)
hand_cropper = HandCropper(target_size=(224, 224))

# Continuous processing
input_mgr.start_continuous_capture()
frame_data = input_mgr.get_latest_frame()

# Detect hands
detections = hand_detector.detect(frame_data.frame)

# Crop hands
hand_rois = hand_cropper.crop_hands(frame_data.frame, detections)
```

### 5.2 Testing Strategy

**Unit Tests**:

- Test HandDetector with known images
- Test HandCropper with various bounding boxes
- Test edge cases (no hands, partial hands, multiple hands)

**Integration Tests**:

- Test with live camera feed
- Test with recorded video sequences
- Test with various lighting conditions

**Performance Tests**:

- Measure FPS under different loads
- Profile memory usage
- Test on different hardware

### 5.3 Example Scripts

- `test_hand_detection.py` - Basic detection test
- `demo_hand_cropping.py` - Visual cropping demo
- `benchmark_performance.py` - Performance measurement

## Dependencies

### Required Packages

```txt
mediapipe>=0.10.0      # Hand detection and tracking
opencv-python>=4.8.0   # Already installed (Input Manager)
numpy>=1.24.0          # Already installed
```

### Installation

```bash
pip install mediapipe>=0.10.0
```

## Success Criteria

### Phase 1 - Hand Detection

- [x] Detect hands in video frames with 95%+ accuracy
- [x] Extract bounding boxes accurately
- [x] Support multi-hand detection (up to 2 hands)
- [x] Provide confidence scores for each detection

### Phase 2 - Hand Cropping

- [x] Extract clean hand ROIs with appropriate padding
- [x] Handle edge cases (boundaries, occlusions)
- [x] Maintain consistent output dimensions

### Phase 3 - Preprocessing

- [x] Standardized 224×224 output size
- [x] Proper normalization for model input
- [x] Consistent color space handling

### Phase 4 - Performance

- [x] Achieve 30+ FPS on laptop CPU
- [x] <30ms per-frame latency
- [x] Stable real-time operation

### Phase 5 - Integration

- [x] Seamless integration with Input Manager
- [x] Clean API for downstream modules
- [x] Comprehensive test coverage

## Next Steps After Hand Detection

Once hand detection is complete:

1. **Gesture Classification Submodule** (ViT + LSTM)
   - Static gesture path (single frame → ViT)
   - Dynamic gesture path (frame sequence → ViT → LSTM)
   - Static vs dynamic routing logic

2. **Semantic Understanding Submodule** (ALBEF-inspired)
   - Visual embedding projection
   - Cosine similarity matching
   - Command mapping

## Timeline Estimate

- **Phase 1** (Core Detection): 2-3 hours
- **Phase 2** (Cropping): 1-2 hours
- **Phase 3** (Preprocessing): 1 hour
- **Phase 4** (Optimization): 2-3 hours
- **Phase 5** (Integration & Testing): 2-3 hours

**Total**: ~10-14 hours of focused development

## References

- [MediaPipe Hands Documentation](https://ai.google.dev/edge/mediapipe/solutions/vision/hand_landmarker)
- [MediaPipe Python API](https://developers.google.com/mediapipe/api/solutions/python/mp/solutions/hands)
- Input Manager Module (already implemented)
- Project Architecture Document (DETAILED_SYSTEM_ARCHITECTURE.md)
