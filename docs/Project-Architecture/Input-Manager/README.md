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
  - Expose a simple function, like `get_frame()`, that the Recognition Engine can call to retrieve the latest processed frame from the camera.

## Implementation Plan

1.  **Camera Access:**
    - Use `cv2.VideoCapture` to initialize the camera.
    - Allow configuration of camera device ID, frame width, and height.
2.  **Frame Processing Loop:**
    - Continuously read frames from the camera.
    - Perform horizontal flip (`cv2.flip`).
    - Provide a thread-safe way for other modules to access the latest frame.
3.  **Module Interface:**
    - Implement a class `InputManager` with methods like `start()`, `stop()`, and `get_frame()`.
