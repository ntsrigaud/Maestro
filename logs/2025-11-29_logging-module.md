# Work Log

## Log Information

- **Date:** 2025-11-29 ~ 2025-11-30
- **Developer:** Neil Taison Rigaud
- **Branch:** logging-module
- **Related Issue/Feature:** Logging Module
- **Module(s):** [input-manager / recognition-engine / action-mapper / os-controller]

---

## Objective

Implement a proper logging mechanism to replace print statements scattered throughout the code.

---

## Tasks Checklist

- [x] Implement shared logging module
- [x] Use it in input manager to replace print() statements
- [x] Test logging outputs to console and file
- [x] Document logging usage in README
- [x] Update gitignore with a comprehensive Python project exclude list to keep the repository clean

---

## Work Log

### [22:55] - Session Start

**What I'm working on:**

- **Current state of the code:**
  - Implementing the shared logging module
  - Replacing `print()` statements with new logging mechanism

### [23:00] - Progress Update

**What I did:**

- **Changes made:**

- Files modified:
  - `src/input-manager/input_manager.py`
  - `.gitignore`
- Files created:
  - `src/shared/logger.py`
  - `src/shared/__init__.py`
  - `src/shared/README.md`
- Files deleted: None

### [23:41] - Progress Update

**Issues encountered:** - Using `sys.path.insert()` to manage imports was causing issues in some execution contexts.

**Solutions applied:** - Make the project installable with `pip install -e .` to properly set up the import paths.

**Changes made:**

- Files modified:
  - `src/input-manager/example_usage.py`
  - `src/input-manager/input_manager.py`
  - `src/input-manager/setup.py`
- Files created:
  - `pyproject.toml`
  - `setup.py`
  - `__init__.py` in `src/`
  - `__init__.py` in `src/input-manager/`
  - `verify_install.py`
- Files deleted: None

### [01:50] - Session End

**Summary of completed work:**

- **Current status:**

- [x] Created shared logging module
- [x] Replaced print statements in input manager with logging calls
- [x] Verified logging outputs to console
- [x] Made project installable with `pip install -e .`
- [x] Migrated to proper Python naming conventions to use `find_packages()`
- [x] Included explanatory documentation for changes
- [x] Updated .gitignore with Python project best practices
- [x] Remove deprecated setup scripts and documentation references
- [x] Implement size-based log rotation as log cleanup mechanism
- [x] Allow reinitialization of logger with different configurations if needed
- [x] Use **Event** threading primitive to ensure thread-safe camera state management

**Testing performed:**

- [ ] Unit tests written/updated
- [x] Manual testing completed
- [ ] Integration testing completed
- [ ] N/A

**Next steps:**

- Merge logging module into **refactor-input-manager** branch

### Architecture Changes

- Introduced a centralized logging module to standardize logging across all modules
- Converted the project to be installable via pip for better dependency and import management

### Dependencies

- Python's built-in `logging` module
- No external dependencies added

### Configuration Changes

- Added logging configuration options in `shared/logger.py`

### Performance Considerations

- Logging levels can be adjusted to control verbosity in different environments (development vs production)

---

## Code Snippets

### Before

```python
        if self._latest_frame is None:
            print("Warning: No frame captured yet after 1 second")
        print("Continuous capture started")
```

### After

```python
        if self._latest_frame is None:
            logger.warning("No frame captured yet after 1 second")

        logger.info("Continuous capture started")
```

---

## References

- Documentation: Chapter on Logging in README files
- Related PRs: [#46 - Logging Module Implementation](https://github.com/blackbird410/CSU-Capstone-Project/pull/46)

---

## Review Notes

**Self-review checklist:**

- [x] Code follows project style guidelines
- [x] Comments added where necessary
- [x] No debugging code left in
- [x] Tests pass
- [x] Documentation updated

**Notes for reviewer:**

- Used Python's built-in logging module for flexibility
- Log messages include timestamps and log levels for better traceability
- Use underscores in package/module names to comply with Python conventions for easier package discovery

## Lessons Learned

What did you learn during this session? Any insights for future work?

- Centralized logging improves maintainability and debugging
- Consider log rotation for production environments to manage log file sizes
- Python packages can't have hyphens in their names, now we are using a setup tool to handle that, but the module naming convention should be updated to use underscores to replace the hyphens.
- An **Event** threading primitive allow one or more thread to wait for a single specific event to be signaled by another thread. In this case, the capture loop will now stop when `stop_continuous_capture()` is called from the main thread, which eliminates the race condition that existed with the plain `_is_capturing()` boolean flag.
