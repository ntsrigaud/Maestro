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
    - Add documenttation on installation and usage in `README.md`.
  - Add a small delay after starting the camera to allow it to initialize before capturing frames.

- **Observations/Discoveries:**
  - C++ build complexity outweighs benefits for current project needs.
  - Python implementation is sufficient for 30 FPS gesture recognition.
  - Focusing on core innovation (Recognition Engine) is more valuable than optimizing input capture.
  - The capture thread for the camera needs a moment (1 second) to intialize and capture the first frame.
    - **Race condition**: The `start_continuous_capture()` method was starting the background thread but returning immediately.
    - **No frames ready**: When the loop first starts, the capture thread had not captured any frames yet, so `get_latest_frame()` returned `None`

### [HH:MM] - Session End

**Summary of completed work:**

- **Current status:**

- [ ] Feature/issue complete
- [ ] Feature/issue in progress
- [ ] Blocked (explain below)

**Testing performed:**

- [ ] Unit tests written/updated
- [ ] Manual testing completed
- [ ] Integration testing completed
- [ ] N/A

**Next steps:**

- **Open questions/concerns:**

- ***

## Technical Notes

### Architecture Changes

Document any changes to the system architecture or module interfaces.

### Dependencies

List any new dependencies added or version changes.

### Configuration Changes

Note any configuration file changes or environment variable updates.

### Performance Considerations

Document any performance impacts or optimizations made.

---

## Code Snippets

### Before

```language
// Relevant code before changes
```

### After

```language
// Code after changes
```

---

## References

- Documentation: [link]
- Related PRs: [link]
- External resources: [link]

---

## Commit History

Document your commits during this session:

- `commit-hash` - Brief commit message
- `commit-hash` - Brief commit message

---

## Review Notes

**Self-review checklist:**

- [ ] Code follows project style guidelines
- [ ] Comments added where necessary
- [ ] No debugging code left in
- [ ] Tests pass
- [ ] Documentation updated

**Notes for reviewer:**

- ***

## Lessons Learned

What did you learn during this session? Any insights for future work?

-
