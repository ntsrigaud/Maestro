# Modular Architecture for Gesture-Based Mouse Control

This document outlines a proposed high-level structure for a seamless, OS-integrated gesture control system. The system is designed to be modular, allowing for independent development and testing of each component.

## High-Level System Architecture

The system can be broken down into four main, independent modules that run as a background service:

```
+-------------------+      +----------------------+      +---------------------+      +--------------------+
| 1. Input Manager  |----->| 2. Recognition       |----->| 3. Action Mapper    |----->| 4. OS Controller   |
| (Webcam Capture)  |      |    Engine (AI/ML)    |      | (Gesture-to-Action) |      | (Virtual Mouse)    |
+-------------------+      +----------------------+      +---------------------+      +--------------------+
```

---

### 1. Input Manager (Camera Capture)
This module's sole responsibility is to connect to the designated webcam and capture video frames at a consistent rate.
-   **Function:** Access the camera, capture images, and pass them to the next module.
-   **Technology:** Libraries like **OpenCV** are standard for this task, providing a simple and efficient API for camera interactions.

### 2. Recognition Engine (The AI Core)
This is the heart of the system, where the "magic" happens. It receives an image from the Input Manager and identifies the gesture. This process, based on the HAVIT paper, would likely involve a two-step pipeline:
-   **A. Hand Detection:** First, locate the hand(s) within the image frame. This is a classic object detection task.
-   **B. Gesture Classification:** Once the hand is located, a second model classifies the gesture (e.g., open palm, fist, pointing finger). This classifier would be the custom model trained for this project.
-   **Output:** The engine outputs a simple, recognized pattern, such as `POINTER`, `FIST`, or `FIVE_FINGERS`.

### 3. Action Mapper (Stateful Logic)
This module acts as the "brain" that translates the recognized gesture into a concrete mouse command. It needs to be stateful to handle sequences of actions.
-   **Function:** It receives the pattern from the Recognition Engine (e.g., `POINTER`) and decides what to do.
-   **Example Logic:**
    -   **IF** current gesture is `POINTER` **AND** it was also `POINTER` in the last frame, **THEN** calculate the change in hand position and map it to cursor movement.
    -   **IF** gesture changes from `POINTER` to `FIST`, **THEN** issue a "left mouse button down" command.
    -   **IF** gesture changes from `FIST` to `OPEN_PALM`, **THEN** issue a "left mouse button up" command.
-   **Output:** A specific command, like `MOVE_CURSOR_TO(x, y)`, `LEFT_CLICK`, or `SCROLL_DOWN`.

### 4. OS Controller (Virtual Mouse)
This is the final layer that directly interacts with the operating system to perform the action.
-   **Function:** It takes the command from the Action Mapper and executes it.
-   **OS Integration (Linux Example):** To achieve seamless integration, this module would create a **virtual input device** using the kernel's `uinput` module. This makes the system think a real physical mouse is connected, ensuring compatibility with all applications without any special configuration. Libraries like `python-libinput` or direct interaction with `/dev/uinput` can be used.

---

### Packaging and Deployment

To make this system easily installable as a single package (`sudo apt install gesture-mouse`), the package would need to:
1.  **Bundle Dependencies:** Specify dependencies like Python, OpenCV, PyTorch/TensorFlow, and any OS-level libraries.
2.  **Include the AI Model:** The trained hand detection and gesture classification models must be packaged with the application.
3.  **Run as a Service:** The installer would set up a **systemd service** (or an equivalent for other init systems). This ensures the gesture recognition process starts automatically on boot and runs silently in the background.
4.  **Handle Permissions:** The installer would configure the necessary user permissions to access camera devices (`/dev/video*`) and create virtual input devices (`/dev/uinput`).
