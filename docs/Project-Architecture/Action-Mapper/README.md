# Action Mapper Module

This document provides the technical design for the Action Mapper module, the logical core of the gesture control system.

## 1. Purpose

The Action Mapper acts as the "brain" of the application. It receives simple gesture labels (e.g., `FIST`) from the Recognition Engine and translates them into a sequence of concrete mouse commands (e.g., `LEFT_CLICK_DOWN`) that are passed to the OS Controller. Its primary challenge is to manage state effectively to interpret gestures in context.

## 2. Core Design: State Machine

To handle the context-sensitive nature of gestures (e.g., a "fist" means "click down" when the cursor is moving, but might mean something else in another context), a **Finite State Machine (FSM)** is the ideal design pattern. The module will always be in one of a few defined states, and transitions between states will be triggered by incoming gestures.

### Key States:

-   **`IDLE`**: The default state. The system is active but no mouse actions are being performed. The hand is not in a designated "control" gesture.
-   **`CURSOR_MOVE`**: The user is actively controlling the cursor position. The hand is in a specific gesture (e.g., `POINTER`).
-   **`DRAGGING`**: The user is performing a click-and-drag operation. This state is entered when a "click" gesture is made while in the `CURSOR_MOVE` state.

### State Transitions (Example):

-   `IDLE` -> `CURSOR_MOVE`
    -   **Trigger:** `POINTER` gesture is detected.
    -   **Action:** Begin mapping hand position to cursor movement.
-   `CURSOR_MOVE` -> `IDLE`
    -   **Trigger:** `OPEN_PALM` gesture is detected. This now exclusively acts as the "stop" signal.
    -   **Action:** Stop all cursor control and return to the idle state.
-   `CURSOR_MOVE` -> `DRAGGING`
    -   **Trigger:** `FIST` gesture is detected.
    -   **Action:** Issue a `LEFT_CLICK_DOWN` command to the Action Queue.
-   `DRAGGING` -> `CURSOR_MOVE`
    -   **Trigger:** The gesture changes from `FIST` back to `POINTER`. (The user "unclenches" their fist back into a pointing gesture).
    -   **Action:** Issue a `LEFT_CLICK_UP` command. This provides an intuitive way to end a drag and immediately continue positioning the cursor.

## 3. Cursor Movement Logic

Translating hand position to cursor position requires two key components:

1.  **Coordinate Mapping:**
    -   The hand's (x, y) coordinates from the camera frame must be mapped to the screen's coordinate system. This is not a direct 1:1 mapping.
    -   A **scaling factor** will be calculated based on the camera resolution and the screen resolution.
    -   An "active area" within the camera's field of view can be defined to prevent unintentional movements when the hand is near the edge of the frame.

2.  **Smoothing:**
    -   Raw coordinate data from hand detection is often jittery, which would result in a shaky cursor.
    -   To ensure smooth movement, a **smoothing algorithm** is required. An **Exponential Moving Average (EMA)** is a good choice as it is computationally efficient and provides a good balance of smoothness and responsiveness.

## 4. Decoupling with an Action Queue

To keep the logic module separate from the OS-specific control module, the Action Mapper will not execute commands directly. Instead, it will push command objects into a thread-safe **Action Queue**.

-   **Example Commands:** `MoveCommand(x, y)`, `ClickCommand('left', 'down')`, `ScrollCommand('vertical', -10)`.
-   The OS Controller will run in a separate thread, continuously pulling commands from this queue and executing them.

## 5. Configuration

To ensure flexibility, the mapping between gestures, states, and actions should not be hard-coded. This will be managed via a user-editable configuration file (e.g., `config.yaml`).

**Example `config.yaml`:**
```yaml
gesture_mappings:
  # Transition from IDLE to CURSOR_MOVE
  - gesture: POINTER
    current_state: IDLE
    next_state: CURSOR_MOVE

  # Transition from CURSOR_MOVE to IDLE (Stop)
  - gesture: OPEN_PALM
    current_state: CURSOR_MOVE
    next_state: IDLE

  # Transition from CURSOR_MOVE to DRAGGING (Start drag)
  - gesture: FIST
    current_state: CURSOR_MOVE
    next_state: DRAGGING
    action: LEFT_CLICK_DOWN

  # Transition from DRAGGING to CURSOR_MOVE (End drag)
  - gesture: POINTER
    current_state: DRAGGING
    next_state: CURSOR_MOVE
    action: LEFT_CLICK_UP
```

## 6. Requirements & Dependencies

-   **Primary:** This module is primarily logic-based and should have minimal external dependencies.
-   **Optional:** `numpy` could be beneficial for efficient numerical calculations in smoothing and coordinate mapping.