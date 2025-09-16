# OS Controller Module

This document provides the technical design for the OS Controller module, which is the final link between the application's logic and the operating system.

## 1. Purpose

The OS Controller runs as a dedicated service that continuously pulls abstract mouse commands (e.g., `MoveCommand`) from the Action Queue. It is responsible for translating these commands into low-level system events to control the mouse cursor and buttons, effectively creating a virtual mouse.

## 2. Core Technology (Linux Focus)

For seamless and native-feeling control on Linux, the module will interface with the kernel's **`uinput` module**. This allows an application to create a virtual input device in userspace that is indistinguishable from a physical device to the rest of the system.

## 3. Requirements & Dependencies

-   **Primary Library:** **`python-evdev`**. This is the modern, recommended library for interacting with both `evdev` (for reading events) and `uinput` (for writing events) in Python.
-   **System-Level:**
    -   The `uinput` kernel module must be loaded.
    -   The user running the application must have write permissions to `/dev/uinput`.

## 4. Implementation Plan

### 1. Device Creation

Upon initialization, the OS Controller will define and create a new virtual input device.

-   **Capabilities:** The device will be configured with the events required for a standard mouse:
    -   `ecodes.EV_REL`: For relative motion (`REL_X`, `REL_Y`) and vertical scrolling (`REL_WHEEL`).
    -   `ecodes.EV_KEY`: For button presses (`BTN_LEFT`, `BTN_RIGHT`, `BTN_MIDDLE`).
-   **Instantiation:** The device will be created using `evdev.UInput(capabilities, name='GestureMouse')`. This should be managed within a `with` statement to ensure the device is properly destroyed on application exit.

### 2. Command Processing Loop

The controller will run in a separate thread from the rest of the application. Its main logic is a simple loop:
1.  Wait for and retrieve a command object from the thread-safe Action Queue.
2.  Translate the command into the corresponding `evdev` event(s).
3.  Write the event(s) to the virtual device.
4.  Send a synchronization event (`ui.syn()`) to notify the kernel to process the batch of events.

**Example Command Translation:**

-   **`MoveCommand(dx, dy)`** from the queue becomes:
    ```python
    ui.write(e.EV_REL, e.REL_X, dx)
    ui.write(e.EV_REL, e.REL_Y, dy)
    ui.syn()
    ```
-   **`ClickCommand('left', 'down')`** becomes:
    ```python
    ui.write(e.EV_KEY, e.BTN_LEFT, 1) # 1 for press
    ui.syn()
    ```
-   **`ClickCommand('left', 'up')`** becomes:
    ```python
    ui.write(e.EV_KEY, e.BTN_LEFT, 0) # 0 for release
    ui.syn()
    ```

## 5. Permissions and System Setup

For the application to function correctly, the following one-time setup is required on the user's system:

1.  **Load Kernel Module:** Ensure the `uinput` module is active.
    ```bash
    sudo modprobe uinput
    ```
2.  **Set Permissions:** The user needs permission to write to the `uinput` device file. The standard way to achieve this is by adding the user to the `input` group.
    ```bash
    sudo usermod -aG input $USER
    ```
    A system reboot or re-login is required for this change to take effect. These steps will be documented in the project's installation guide.

## 6. Cross-Platform Considerations

This `uinput`-based design is **Linux-specific**. To support other operating systems, this module would need to be adapted with a different backend while keeping the Action Queue interface the same.

-   **Windows:** Use the `pywin32` library to call `SendInput()` from the WinAPI.
-   **macOS:** Use the `pyobjc` library to interface with the Core Graphics framework.
-   **Alternative:** A library like **`pynput`** abstracts these OS-specific details away and could be used as a simpler, cross-platform backend, potentially with some trade-offs in performance or seamlessness compared to the direct `uinput` approach.