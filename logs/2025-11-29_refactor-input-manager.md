# Work Log

## Log Information

- **Date:** 2025-11-29
- **Developer:** Neil Taison RIGAUD
- **Branch:** refactor-input-manager
- **Related Issue/Feature:** Refactor Input Manager
- **Module(s):** input-manager

---

## Objective

Refactor the input manager module to improve its portability and maintainability.

---

## Tasks Checklist

- [x] Create a unified `src/` directory to place all module implementations
- [x] Update **input-manager** current implementation to allow portability on Linux
- [x] Test installation and execution of the module

---

## Work Log

### [20:30] - Session Start

**What I'm working on:**

- **Create a unified `src/` directory to contain all module implementations**

**What I did:**

- **Changes made:**

- Files modified: Moved `InputManagerModule` under `src/input-manager`
- Files created: Created `src/` directory
- Files deleted: None

### [20:45] - Progress Update

**What I'm working on:**

- **Update input-manager implementation for Linux portability**

**What I did:**

- **Changes made:**

- Files modified:
- Files created:
- Files deleted:

**Observations/Discoveries:**

- **Issues encountered:**
  - C++ build issues on Linux due to missing dependencies.
    - CMake version compatibility problems.
    - OpenCV package installation issues.
    - Unnecessary build artifacts cluttering the source directory.
  - Frames not being captured immediately after starting the camera.

- **Solutions applied:**
  - Pure Python Implementation for Prototyping
    - Create a preliminary version of the program that works and test concepts quickly without being prevent by current technical issues.
    - Zero build requirements - works immediately
    - Cross-platform by default (Linux, macOS, Windows)
    - Fast iteration during development
    - We can always optimize later.
    - Remove C++ implementation from the repository to avoid confusion.
    - Simplify the codebase and reduce maintenance overhead.
    - Add documentation on installation and usage in `README.md`.
  - Add a small delay after starting the camera to allow it to initialize before capturing frames.

- **Observations/Discoveries:**
  - C++ build complexity outweighs benefits for current project needs.
  - Python implementation is sufficient for 30 FPS gesture recognition.
  - Focusing on core innovation (Recognition Engine) is more valuable than optimizing input capture.
  - The capture thread for the camera needs a moment (1 second) to intialize and capture the first frame.
    - **Race condition**: The `start_continuous_capture()` method was starting the background thread but returning immediately.
    - **No frames ready**: When the loop first starts, the capture thread had not captured any frames yet, so `get_latest_frame()` returned `None`

### [22:00] - Session End

**Summary of completed work:**

- **Current status:**

- [x] Refactored input-manager to use pure Python implementation
- [x] Updated documentation with installation and usage instructions
- [x] Tested module on Linux, macOS, and Windows

**Testing performed:**

- [x] Manual testing completed
- [x] Integration testing completed

**Next steps:**

- **Open questions/concerns:** None at this time.
- **Implement unit tests for input-manager module to ensure reliability.**
- **Implement recognition-engine module next.**

## Technical Notes

### Dependencies

- Added `opencv-python` for video capture and processing.
- Removed C++ build dependencies (CMake, pybind11, OpenCV C++).

### Configuration Changes

- Updated `README.md` and `QUICKSTART.md` with new installation and usage instructions.

### Performance Considerations

- The pure Python implementation achieves sufficient performance (30 FPS) for gesture recognition tasks.

---

## References

- Documentation: Check `README.md` and `QUICKSTART.md` in the `input-manager` module for installation and usage details.
- Related PRs: [Refactor input manager](https://github.com/blackbird410/CSU-Capstone-Project/pull/45)

---

## Review Notes

**Self-review checklist:**

- [x] Code follows project style guidelines
- [x] Comments added where necessary
- [x] No debugging code left in
- [x] Tests pass
- [x] Documentation updated

**Notes for reviewer:**

- Please focus on the clarity of the documentation and the correctness of the input-manager functionality.

## Lessons Learned

- Choosing the right tool for the job is crucial; sometimes simplicity and ease of use outweigh raw performance.
