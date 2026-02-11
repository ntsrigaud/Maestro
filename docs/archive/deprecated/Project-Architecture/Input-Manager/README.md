# Input Manager

The Input Manager is responsible for capturing video frames from the camera and providing them to the Recognition Engine with precise temporal metadata.

## Documentation

- **[Timestamp Format Specification](./TIMESTAMP_FORMAT.md)** - Detailed explanation of nanosecond timestamp format and usage patterns

## Technical Design

The Input Manager is implemented as a standalone Python module with a thread-safe interface.

### Key Responsibilities:

- **Camera Handling:**
  - Utilize OpenCV (`cv2`) to connect to and manage the webcam.
  - Implement functions for camera initialization, setting resolution (width/height), reading frames in a continuous loop, and properly releasing the camera resource upon application exit.
- **Frame Preparation:**
  - Incorporate basic frame preprocessing, including horizontally flipping the image for a more intuitive "mirror" effect for the user.
- **Interface:**
  - Provide two capture modes:
    - **Synchronous**: `get_frame()` - Capture single frame on demand
    - **Continuous**: `start_continuous_capture()` + `get_latest_frame()` - Background thread with frame buffering
  - Return frames with precise nanosecond timestamps for temporal analysis

## Implementation

The Input Manager is implemented as a pure Python module with:

1.  **Camera Access:**
    - `cv2.VideoCapture` for camera initialization
    - Configurable camera device ID, resolution, and target FPS
2.  **Thread-Safe Continuous Capture:**
    - Background thread for frame capture
    - Frame buffering with configurable size (deque)
    - Thread-safe access with locks and Events
3.  **Module Interface:**
    - `start_capture()` / `stop_capture()` - Camera lifecycle
    - `get_frame()` - Synchronous single frame capture
    - `start_continuous_capture()` / `stop_continuous_capture()` - Background capture
    - `get_latest_frame()` - Get most recent frame with FrameData
    - `get_all_frames()` - Get buffered frame sequence
