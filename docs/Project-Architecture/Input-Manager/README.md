# Input Manager Module

This document outlines the technical design and requirements for the Input Manager module.

## 1. Purpose

The Input Manager is responsible for connecting to the system's webcam, capturing video frames, and passing them to the Recognition Engine for processing. It must be efficient to minimize latency and handle potential camera access errors gracefully.

## 2. Requirements & Dependencies

-   **Primary Library:** `opencv-python`
    -   This library provides the necessary functions to interface with camera hardware in a platform-independent way.

## 3. Implementation Plan

The implementation will be a Python class that encapsulates the camera handling logic.

1.  **Initialization:** The class constructor will initialize a `cv2.VideoCapture` object. The camera index (e.g., 0 for the default webcam) will be a configurable parameter.
2.  **Error Handling:** Immediately after initialization, the code must check if the camera was opened successfully using the `isOpened()` method. If not, it should raise an exception or log a critical error.
3.  **Frame Capture:** A primary method, such as `get_frame()`, will be responsible for reading a single frame from the camera using the `read()` method. This method returns a success boolean and the frame data, which should be handled appropriately.
4.  **Resource Management:** The class must implement a `release()` method that calls `release()` on the `VideoCapture` object to free the camera for other applications. This is critical for clean shutdown and resource management.

## 4. Example Implementation

Below is a basic code snippet demonstrating the core functionality.

```python
import cv2

class InputManager:
    def __init__(self, camera_index=0):
        """Initializes the video capture object."""
        self.camera_index = camera_index
        self.cap = cv2.VideoCapture(self.camera_index)

        if not self.cap.isOpened():
            raise IOError(f"Cannot open webcam at index {self.camera_index}")

    def get_frame(self):
        """Reads a single frame from the webcam."""
        ret, frame = self.cap.read()
        if not ret:
            print("Error: Can't receive frame. Exiting ...")
            return None
        return frame

    def release(self):
        """Releases the camera resource."""
        self.cap.release()
        cv2.destroyAllWindows()

# Example Usage:
if __name__ == '__main__':
    try:
        input_manager = InputManager()
        while True:
            frame = input_manager.get_frame()
            if frame is None:
                break

            # The frame would be passed to the Recognition Engine here
            cv2.imshow('Webcam Feed', frame)

            # Exit on 'q' key press
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
    except IOError as e:
        print(e)
    finally:
        if 'input_manager' in locals():
            input_manager.release()

```
