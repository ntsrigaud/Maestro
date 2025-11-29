# InputManager Module

This module provides a high-performance interface to capture and preprocess video frames from a camera using pure Python.

## Quick Start

**Zero build required!** Just install the package and run:

```bash
# 1. Install the package (from project root)
pip install -e .

# 2. Try the examples
python src/input_manager/example_usage.py
```

Works on **Linux, macOS, and Windows**.

### Basic Usage

```python
from input_manager import InputManager

# Create manager and start capture
manager = InputManager(camera_index=0)
manager.start_continuous_capture()

# Get latest frame
frame_data = manager.get_latest_frame()
frame = frame_data.to_numpy()  # Convert to numpy array

# Cleanup
manager.stop_continuous_capture()
```

---

## Table of Contents

1. [Architecture](#architecture)
2. [Installation](#installation)
3. [API Reference](#api-reference)
4. [Examples](#examples)

---

## Architecture

### Data Flow

```
Camera → OpenCV → Capture Thread → Preprocessing → FrameData → NumPy Array
```

**Key Components:**

1. **InputManager Class**: Manages camera lifecycle and frame capture
2. **Capture Thread**: Background thread for continuous frame acquisition
3. **Frame Buffer**: Thread-safe deque storing recent frames
4. **Preprocessing Pipeline**: Mirror, color conversion, resize, normalize
5. **FrameData**: Container for frame with metadata (timestamp, frame number)

---

## Installation

**Requirements:**

- Python 3.8 or higher
- pip

**Steps:**

```bash
# From project root
pip install -e .
```

**Dependencies:**

- `opencv-python>=4.8.0` - Computer vision library
- `numpy>=1.24.0` - Array processing

Dependencies are automatically installed with the package.

---

## API Reference

### `InputManager` Class

Main interface for camera capture and frame processing.

#### Constructor

```python
InputManager(
    camera_index: int = 0,
    buffer_size: int = 30,
    target_fps: int = 30,
    width: Optional[int] = None,
    height: Optional[int] = None
)
```

**Parameters:**

- `camera_index`: Camera device index (0 = built-in camera)
- `buffer_size`: Number of frames to keep in circular buffer
- `target_fps`: Target frame rate for capture
- `width`: Target frame width (None = camera default)
- `height`: Target frame height (None = camera default)

#### Methods

##### `start_capture() -> bool`

Open camera device.

**Returns:** `True` if successful, `False` otherwise

```python
manager = InputManager()
if manager.start_capture():
    print("Camera opened")
```

##### `stop_capture()`

Release camera resources and stop all capture.

```python
manager.stop_capture()
```

##### `is_capture_open() -> bool`

Check if camera is currently open.

**Returns:** `True` if camera is open, `False` otherwise

##### `start_continuous_capture() -> bool`

Start background thread for continuous frame capture.

**Returns:** `True` if successful, `False` otherwise

```python
manager.start_continuous_capture()
# Frames are now being captured in background
```

##### `stop_continuous_capture()`

Stop background capture thread.

##### `get_latest_frame() -> Optional[FrameData]`

Get the most recent captured frame.

**Returns:** `FrameData` object or `None` if no frame available

```python
frame_data = manager.get_latest_frame()
if frame_data:
    frame = frame_data.to_numpy()
```

##### `get_all_frames() -> List[FrameData]`

Get all frames currently in buffer.

**Returns:** List of `FrameData` objects

```python
frames = manager.get_all_frames()
print(f"Buffer has {len(frames)} frames")
```

##### `get_frame() -> Optional[np.ndarray]`

Capture and return a single frame (synchronous).

**Returns:** Preprocessed frame as numpy array, or `None` on failure

```python
frame = manager.get_frame()  # Blocks until frame captured
```

##### Configuration Methods

```python
# Enable/disable horizontal mirror
manager.set_mirror_enabled(enabled: bool)

# Set normalization parameters
manager.set_normalization_params(mean: float, std: float)

# Set target output size
manager.set_target_size(width: int, height: int)

# Enable/disable BGR to RGB conversion
manager.set_rgb_conversion(enabled: bool)

# Get camera information
info = manager.get_camera_info()  # Returns dict
```

### `FrameData` Class

Container for frame with metadata.

#### Attributes

- `frame: np.ndarray` - The frame data
- `timestamp: int` - Capture time in milliseconds since epoch
- `frame_number: int` - Sequential frame counter

#### Methods

##### `to_numpy() -> np.ndarray`

Convert frame to numpy array (for API compatibility).

```python
frame_data = manager.get_latest_frame()
frame = frame_data.to_numpy()
```

---

## Examples

### Example 1: Basic Capture

```python
from input_manager import InputManager

# Create and start
manager = InputManager(camera_index=0)
manager.start_continuous_capture()

# Get frames
for _ in range(100):
    frame_data = manager.get_latest_frame()
    if frame_data:
        frame = frame_data.to_numpy()
        print(f"Frame shape: {frame.shape}")

# Cleanup
manager.stop_continuous_capture()
```

### Example 2: With Preprocessing

```python
from input_manager import InputManager

manager = InputManager(camera_index=0)

# Configure preprocessing
manager.set_mirror_enabled(True)
manager.set_rgb_conversion(True)
manager.set_target_size(640, 480)
manager.set_normalization_params(mean=127.5, std=127.5)

manager.start_continuous_capture()
frame_data = manager.get_latest_frame()
# Frame is now: mirrored, RGB, 640x480, normalized
```

### Example 3: Context Manager

```python
from input_manager import InputManager

with InputManager() as manager:
    manager.start_continuous_capture()
    frame = manager.get_latest_frame()
    # Camera automatically released on exit
```

### Example 4: Frame Buffer Analysis

```python
from input_manager import InputManager
import time

manager = InputManager(buffer_size=10)
manager.start_continuous_capture()
time.sleep(2)  # Capture for 2 seconds

# Analyze captured frames
frames = manager.get_all_frames()
for i, frame_data in enumerate(frames):
    print(f"Frame {i}: timestamp={frame_data.timestamp}, "
          f"shape={frame_data.frame.shape}")

manager.stop_continuous_capture()
```

**More examples:** Run `python src/input_manager/example_usage.py` for comprehensive examples.

---

## Performance

The Python implementation achieves:

- **30-60 FPS** on modern laptops
- **< 33ms latency** per frame
- **Sufficient for real-time gesture recognition**

---

## Troubleshooting

### Camera fails to open

- Check if another application is using the camera
- Try different camera indices (0, 1, 2, ...)
- On macOS: Grant camera permissions in System Preferences > Security & Privacy

### No module named 'cv2'

```bash
pip install opencv-python
```

### Poor performance

- Reduce `target_fps`
- Decrease frame resolution with `width` and `height` parameters
- Disable unnecessary preprocessing steps
