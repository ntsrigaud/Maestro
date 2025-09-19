# Action Mapper

The Action Mapper is responsible for translating the recognized gestures from the Recognition Engine into specific commands for the OS Controller.

## Technical Design

The Action Mapper acts as a bridge between the Recognition Engine and the OS Controller.

### Key Responsibilities:

*   **Command Mapping:**
    *   In the baseline implementation, the Action Mapper will use a simple dictionary or switch-case structure to map recognized gesture IDs to predefined commands.
    *   In the advanced implementation, the Action Mapper will receive the semantic meaning of a gesture (e.g., the text "Volume Up") from the Recognition Engine. It will then map this semantic meaning to the corresponding command for the OS Controller.
*   **State Management:**
    *   The Action Mapper will maintain the state of the system to avoid unintended actions. For example, it might require a specific gesture to "activate" or "deactivate" gesture control.
*   **Configuration:**
    *   The mapping from gestures to commands will be configurable, allowing users to customize the controls.

## Implementation Plan

1.  **Define Command Set:**
    *   Establish a clear set of commands that the system will support (e.g., `VOLUME_UP`, `NEXT_TRACK`, `MOUSE_MOVE_LEFT`).
2.  **Implement Mapping Logic:**
    *   Create a mapping from the output of the Recognition Engine (gesture IDs or semantic text) to the command set.
3.  **Develop State Machine:**
    *   Implement a simple state machine to manage the activation and deactivation of gesture control.
