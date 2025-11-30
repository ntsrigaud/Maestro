# Hand Detection Submodule

Comprehensive documentation for the hand detection submodule of the Recognition Engine.

## Table of Contents

- [Overview](#overview)
- [Architecture](#architecture)
- [Quick Start](#quick-start)
- [API Reference](#api-reference)
- [Usage Examples](#usage-examples)
- [Parameter Tuning](#parameter-tuning)
- [Integration Guide](#integration-guide)
- [Performance Considerations](#performance-considerations)
- [Troubleshooting](#troubleshooting)

## Overview

The hand detection submodule provides real-time hand localization and region extraction using Google's MediaPipe Hands solution. It detects hands in video frames, computes bounding boxes, extracts 21 hand landmarks, and prepares cropped regions for downstream gesture classification.

### Key Features

- **Multi-hand detection**: Detect up to 2 hands simultaneously
- **High accuracy**: MediaPipe Hands with 21-landmark tracking
- **Real-time performance**: Optimized for 30+ FPS on laptop CPUs
- **Flexible preprocessing**: Configurable normalization and resizing
- **Robust boundary handling**: Safe cropping at frame edges

### Components

1. **HandDetector**: Detects hands and extracts landmarks using MediaPipe
2. **HandCropper**: Extracts and preprocesses hand regions for classification
3. **HandDetectionResult**: Data structure for detection results
4. **HandROI**: Data structure for cropped hand regions

## Architecture

```
┌─────────────────────────────────────────────────┐
│              Input Frame (BGR)                  │
└──────────────────┬──────────────────────────────┘
                   │
                   ▼
┌─────────────────────────────────────────────────┐
│           HandDetector (MediaPipe)              │
│  • Detect hands (up to max_num_hands)           │
│  • Extract 21 landmarks per hand                │
│  • Calculate bounding boxes                     │
│  • Determine handedness (Left/Right)            │
└──────────────────┬──────────────────────────────┘
                   │
                   ▼
        List[HandDetectionResult]
                   │
                   ▼
┌─────────────────────────────────────────────────┐
│           HandCropper                           │
│  • Crop hand regions with padding               │
│  • Ensure square aspect ratio                   │
│  • Resize to target size (224×224)              │
│  • Normalize pixels (ImageNet stats)            │
└──────────────────┬──────────────────────────────┘
                   │
                   ▼
           List[HandROI]
```

## Quick Start

### Installation

```bash
# Install the package with hand detection dependencies
pip install -e .
```

### Basic Usage

```python
from input_manager import InputManager
from recognition_engine.hand_detection import HandDetector, HandCropper

# Initialize components
input_mgr = InputManager(camera_index=0)
detector = HandDetector(max_num_hands=2)
cropper = HandCropper(target_size=(224, 224))

# Start capture
input_mgr.start_continuous_capture()

# Process frames
frame_data = input_mgr.get_latest_frame()
detections = detector.detect(frame_data.frame, frame_data.timestamp)
hand_rois = cropper.crop_multiple(frame_data.frame, detections)

# Use the hand regions for classification
for roi in hand_rois:
    print(f"Hand: {roi.detection.handedness}")
    print(f"Confidence: {roi.detection.confidence:.2f}")
    print(f"Image shape: {roi.image.shape}")

# Cleanup
input_mgr.stop_continuous_capture()
detector.close()
```

### Running the Demo

```bash
# Basic demo
python scripts/demo_hand_detection.py

# With options
python scripts/demo_hand_detection.py --camera 0 --max-hands 2

# Without landmarks or cropped view
python scripts/demo_hand_detection.py --no-landmarks --no-cropped
```

## API Reference

### HandDetector

Detects hands in frames using MediaPipe Hands.

**Constructor**:

```python
HandDetector(
    max_num_hands: int = 2,
    min_detection_confidence: float = 0.5,
    min_tracking_confidence: float = 0.5,
    model_complexity: int = 1,
    static_image_mode: bool = False,
)
```

**Parameters**:

- `max_num_hands`: Maximum number of hands to detect (1-2)
- `min_detection_confidence`: Minimum confidence for initial detection [0, 1]
- `min_tracking_confidence`: Minimum confidence for tracking [0, 1]
- `model_complexity`: Model complexity (0=lite, 1=full)
- `static_image_mode`: Treat each frame independently (no tracking)

**Methods**:

```python
def detect(frame: np.ndarray, timestamp: Optional[int] = None) -> List[HandDetectionResult]:
    """
    Detect hands in a single frame.

    Args:
        frame: Input image (BGR format)
        timestamp: Optional timestamp in nanoseconds

    Returns:
        List of HandDetectionResult objects
    """
```

```python
def detect_batch(frames: List[np.ndarray], timestamps: Optional[List[int]] = None) -> List[List[HandDetectionResult]]:
    """
    Detect hands in multiple frames.

    Args:
        frames: List of input frames
        timestamps: Optional timestamps

    Returns:
        List of detection lists, one per frame
    """
```

```python
def get_stats() -> dict:
    """Get detection statistics."""
```

```python
def reset_stats():
    """Reset performance statistics."""
```

```python
def close():
    """Release MediaPipe resources."""
```

### HandCropper

Extracts and preprocesses hand regions from frames.

**Constructor**:

```python
HandCropper(
    target_size: Tuple[int, int] = (224, 224),
    padding: float = 0.2,
    square_crop: bool = True,
    normalize: bool = True,
    normalization_mean: Tuple[float, float, float] = (0.485, 0.456, 0.406),
    normalization_std: Tuple[float, float, float] = (0.229, 0.224, 0.225),
    bgr_to_rgb: bool = True,
)
```

**Parameters**:

- `target_size`: Target (width, height) for resized ROI
- `padding`: Padding factor around bbox (0.2 = 20%)
- `square_crop`: Whether to crop square regions
- `normalize`: Whether to normalize pixel values
- `normalization_mean`: Mean values for normalization (ImageNet defaults)
- `normalization_std`: Std values for normalization (ImageNet defaults)
- `bgr_to_rgb`: Convert BGR to RGB

**Methods**:

```python
def crop_hand(frame: np.ndarray, detection: HandDetectionResult, apply_preprocessing: bool = True) -> HandROI:
    """
    Crop a hand region from a frame.

    Args:
        frame: Input frame (BGR format)
        detection: Hand detection result
        apply_preprocessing: Whether to apply preprocessing

    Returns:
        HandROI object
    """
```

```python
def crop_multiple(frame: np.ndarray, detections: list[HandDetectionResult], apply_preprocessing: bool = True) -> list[HandROI]:
    """
    Crop multiple hand regions from a single frame.

    Args:
        frame: Input frame
        detections: List of detections
        apply_preprocessing: Whether to apply preprocessing

    Returns:
        List of HandROI objects
    """
```

### HandDetectionResult

Data structure for hand detection results.

**Attributes**:

- `bbox`: Tuple[int, int, int, int] - Bounding box (x, y, width, height)
- `landmarks`: np.ndarray - Shape (21, 3) with normalized (x, y, z) coordinates
- `handedness`: str - "Left" or "Right"
- `confidence`: float - Detection confidence [0, 1]
- `timestamp`: int - Timestamp in nanoseconds
- `frame_index`: Optional[int] - Frame number

**Methods**:

- `get_bbox_center()`: Returns center point of bounding box
- `get_fingertips()`: Returns landmarks for 5 fingertips
- `get_wrist()`: Returns wrist landmark (index 0)
- `get_landmark(index)`: Returns specific landmark by index
- `to_dict()`: Serializes to dictionary

### HandROI

Data structure for cropped hand regions.

**Attributes**:

- `image`: np.ndarray - Preprocessed hand region
- `detection`: HandDetectionResult - Original detection
- `preprocessing_params`: dict - Preprocessing configuration

**Methods**:

- `get_bbox()`: Returns original bounding box
- `get_handedness()`: Returns handedness label
- `get_confidence()`: Returns detection confidence

## Usage Examples

### Example 1: Basic Detection

```python
from input_manager import InputManager
from recognition_engine.hand_detection import HandDetector

input_mgr = InputManager()
detector = HandDetector()

input_mgr.start_continuous_capture()

while True:
    frame_data = input_mgr.get_latest_frame()
    if frame_data is None:
        continue

    detections = detector.detect(frame_data.frame)

    for det in detections:
        print(f"{det.handedness} hand: confidence={det.confidence:.2f}")

    if len(detections) > 0:
        break

input_mgr.stop_continuous_capture()
detector.close()
```

### Example 2: Context Manager Pattern

```python
from input_manager import InputManager
from recognition_engine.hand_detection import HandDetector

with InputManager() as input_mgr, HandDetector() as detector:
    input_mgr.start_continuous_capture()

    frame_data = input_mgr.get_latest_frame()
    detections = detector.detect(frame_data.frame)

    print(f"Detected {len(detections)} hands")
```

### Example 3: Custom Preprocessing

```python
from recognition_engine.hand_detection import HandCropper

# Cropper without normalization (for visualization)
viz_cropper = HandCropper(
    target_size=(224, 224),
    padding=0.3,
    normalize=False,
    bgr_to_rgb=False,
)

# Cropper for model inference (normalized)
model_cropper = HandCropper(
    target_size=(224, 224),
    padding=0.2,
    normalize=True,
    normalization_mean=(0.485, 0.456, 0.406),
    normalization_std=(0.229, 0.224, 0.225),
    bgr_to_rgb=True,
)
```

### Example 4: Accessing Landmarks

```python
detection = detector.detect(frame)[0]

# Get specific landmarks
wrist = detection.get_wrist()
fingertips = detection.get_fingertips()
index_tip = detection.get_landmark(8)

# Get all landmarks
landmarks = detection.landmarks  # Shape: (21, 3)

# Landmark indices (MediaPipe Hand):
# 0: Wrist
# 1-4: Thumb
# 5-8: Index
# 9-12: Middle
# 13-16: Ring
# 17-20: Pinky
```

### Example 5: Multi-Hand Processing

```python
detector = HandDetector(max_num_hands=2)
cropper = HandCropper()

frame_data = input_mgr.get_latest_frame()
detections = detector.detect(frame_data.frame)
hand_rois = cropper.crop_multiple(frame_data.frame, detections)

for i, roi in enumerate(hand_rois):
    print(f"Hand {i+1}:")
    print(f"  Handedness: {roi.get_handedness()}")
    print(f"  Confidence: {roi.get_confidence():.2f}")
    print(f"  Image shape: {roi.image.shape}")
```

## Parameter Tuning

### Detection Confidence

**min_detection_confidence**: Controls initial hand detection sensitivity

- **Lower (0.3-0.4)**: More detections, more false positives
- **Medium (0.5-0.6)**: Balanced (default: 0.5)
- **Higher (0.7-0.9)**: Fewer detections, higher precision

```python
# High precision mode
detector = HandDetector(min_detection_confidence=0.7)

# High recall mode
detector = HandDetector(min_detection_confidence=0.3)
```

### Model Complexity

**model_complexity**: Trade-off between speed and accuracy

- **0 (Lite)**: Faster, less accurate (~50% faster)
- **1 (Full)**: Slower, more accurate (default)

```python
# Fast mode for real-time applications
detector = HandDetector(model_complexity=0)

# Accurate mode for offline processing
detector = HandDetector(model_complexity=1)
```

### Cropping Parameters

**padding**: Amount of context around hand

- **0.1**: Tight crop (10% padding)
- **0.2**: Standard crop (20% padding, default)
- **0.3**: Loose crop (30% padding)

```python
# Tight crop for focused hand region
cropper = HandCropper(padding=0.1)

# Loose crop for more context
cropper = HandCropper(padding=0.3)
```

**target_size**: Output resolution

- **(128, 128)**: Fast, lower quality
- **(224, 224)**: Standard for ViT (default)
- **(256, 256)**: Higher quality, slower

## Integration Guide

### With InputManager

```python
from input_manager import InputManager, FrameData
from recognition_engine.hand_detection import HandDetector, HandCropper

class HandGestureRecognizer:
    def __init__(self):
        self.input_mgr = InputManager()
        self.detector = HandDetector()
        self.cropper = HandCropper()

    def start(self):
        self.input_mgr.start_continuous_capture()

    def process_frame(self) -> list:
        frame_data = self.input_mgr.get_latest_frame()
        if frame_data is None:
            return []

        detections = self.detector.detect(
            frame_data.frame,
            frame_data.timestamp
        )

        hand_rois = self.cropper.crop_multiple(
            frame_data.frame,
            detections
        )

        return hand_rois

    def stop(self):
        self.input_mgr.stop_continuous_capture()
        self.detector.close()
```

### With Custom Pipeline

```python
class CustomPipeline:
    def __init__(self):
        self.detector = HandDetector(max_num_hands=1)
        self.cropper = HandCropper(normalize=True)

    def process(self, frame):
        # Detect
        detections = self.detector.detect(frame)
        if not detections:
            return None

        # Crop
        roi = self.cropper.crop_hand(frame, detections[0])

        # Your custom processing
        features = self.extract_features(roi.image)
        gesture = self.classify_gesture(features)

        return gesture

    def extract_features(self, image):
        # Your feature extraction logic
        pass

    def classify_gesture(self, features):
        # Your classification logic
        pass
```

## Performance Considerations

### Target Performance

- **FPS**: 30+ on laptop CPU (Apple M3, Intel i5/i7)
- **Latency**: <35ms per frame (detection + cropping)
- **Memory**: ~200MB (MediaPipe models + buffers)

### Optimization Tips

1. **Use model_complexity=0** for real-time applications
2. **Reduce max_num_hands** if detecting single hand
3. **Set static_image_mode=False** for video streams (enables tracking)
4. **Disable normalization** in HandCropper if not needed
5. **Use smaller target_size** if acceptable for your model

### Benchmarking

```python
import time

detector = HandDetector()
cropper = HandCropper()

# Warm-up
for _ in range(10):
    _ = detector.detect(frame)

# Benchmark
start = time.perf_counter()
for _ in range(100):
    detections = detector.detect(frame)
    hand_rois = cropper.crop_multiple(frame, detections)
end = time.perf_counter()

fps = 100 / (end - start)
print(f"Average FPS: {fps:.1f}")
```

## Troubleshooting

### Issue: No hands detected

**Possible causes**:

- Hands not in frame
- Poor lighting
- min_detection_confidence too high

**Solutions**:

```python
# Lower detection threshold
detector = HandDetector(min_detection_confidence=0.3)

# Check frame is valid
if frame is None or frame.size == 0:
    print("Invalid frame")
```

### Issue: Slow performance

**Possible causes**:

- model_complexity=1 (full model)
- max_num_hands=2 when only 1 hand needed
- Large target_size in HandCropper

**Solutions**:

```python
# Use lite model
detector = HandDetector(model_complexity=0, max_num_hands=1)

# Reduce target size
cropper = HandCropper(target_size=(128, 128))
```

### Issue: Cropping fails at frame edges

**Possible causes**:

- Hand partially out of frame
- Excessive padding

**Solutions**:

```python
# Reduce padding
cropper = HandCropper(padding=0.1)

# The cropper automatically clips to frame boundaries
# Check logs for boundary clip warnings
```

### Issue: Import errors

**Possible causes**:

- Package not installed
- MediaPipe not installed

**Solutions**:

```bash
# Reinstall package
pip install -e .

# Verify installation
python scripts/validate_hand_detection.py
```

## Next Steps

- **Gesture Classification**: Integrate with ViT-based gesture classifier
- **Action Mapping**: Map gestures to system actions
- **Performance Profiling**: Benchmark on target hardware
- **Model Training**: Train custom gesture recognition models

## References

- [MediaPipe Hands](https://google.github.io/mediapipe/solutions/hands.html)
- [Hand Landmark Model](https://google.github.io/mediapipe/solutions/hands#hand-landmark-model)
- [Project Architecture](../Project-Architecture/DETAILED_SYSTEM_ARCHITECTURE.md)
