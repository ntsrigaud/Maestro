# Logging System

The project uses a centralized logging system located in `src/shared/logger.py` that provides consistent logging across all modules.

## Features

- **Configurable log levels**: DEBUG, INFO, WARNING, ERROR, CRITICAL
- **Colored console output**: Different colors for different log levels
- **File logging**: Optional file output with rotation
- **Module-specific loggers**: Each module has its own logger
- **Consistent formatting**: Unified format across the project

## Usage

### Basic Usage

```python
from shared.logger import get_logger

# Get a logger for your module
logger = get_logger(__name__)

# Log messages at different levels
logger.debug("Detailed debugging information")
logger.info("General information message")
logger.warning("Warning message")
logger.error("Error message")
logger.critical("Critical error message")
```

### Configuration

#### Development Mode (Console only, verbose)

```python
from shared.logger import configure_logging
import logging

# Show all messages including DEBUG
configure_logging(level=logging.DEBUG)
```

#### Production Mode (With file logging and rotation)

```python
from shared.logger import configure_logging
import logging

# INFO level with file logging and automatic rotation
configure_logging(
    level=logging.INFO,
    enable_file_logging=True,
    log_dir="/path/to/logs",
    max_bytes=10 * 1024 * 1024,  # 10MB per file (default)
    backup_count=5  # Keep 5 backup files (default)
)
```

**Log Rotation:**

- Log file: `logs/application.log`
- When `application.log` reaches 10MB, it's renamed to `application.log.1`
- Previous backups are renamed: `.1` → `.2`, `.2` → `.3`, etc.
- Oldest backup (`.5`) is deleted
- New logs start in fresh `application.log`
- Total disk usage: ~50MB maximum (10MB × 5 backups)

#### Custom Rotation Settings

```python
# Larger files, more backups for production
configure_logging(
    level=logging.INFO,
    enable_file_logging=True,
    max_bytes=20 * 1024 * 1024,  # 20MB per file
    backup_count=10  # Keep 10 backups = 200MB total
)

# Smaller files, fewer backups for development
configure_logging(
    level=logging.DEBUG,
    enable_file_logging=True,
    max_bytes=5 * 1024 * 1024,  # 5MB per file
    backup_count=3  # Keep 3 backups = 15MB total
)
```

### Changing Log Level at Runtime

```python
from shared.logger import set_log_level
import logging

# Start with INFO level
configure_logging(level=logging.INFO)

# ... application running ...

# Switch to DEBUG for troubleshooting
set_log_level(logging.DEBUG)

# Switch back to WARNING
set_log_level(logging.WARNING)
```

## Log Levels

| Level    | Numeric Value | When to Use                                    |
| -------- | ------------- | ---------------------------------------------- |
| DEBUG    | 10            | Detailed diagnostic information                |
| INFO     | 20            | General informational messages                 |
| WARNING  | 30            | Warning messages (non-critical issues)         |
| ERROR    | 40            | Error messages (operation failures)            |
| CRITICAL | 50            | Critical errors (application may not continue) |

## Module Integration

Each module should initialize its logger at the top of the file:

```python
# At the top of your module
from shared.logger import get_logger

logger = get_logger(__name__)
```

Then use the logger throughout the module:

```python
def process_frame(frame):
    logger.debug(f"Processing frame with shape {frame.shape}")

    if frame is None:
        logger.error("Received None frame")
        return None

    try:
        result = expensive_operation(frame)
        logger.info("Frame processed successfully")
        return result
    except Exception as e:
        logger.error(f"Failed to process frame: {e}")
        return None
```

## Examples

### Input Manager Module

```python
from shared.logger import get_logger

logger = get_logger(__name__)

def start_capture(self):
    logger.info(f"Starting capture on camera {self._camera_index}")

    if not self._capture.isOpened():
        logger.error(f"Failed to open camera {self._camera_index}")
        return False

    logger.info("Camera opened successfully")
    return True
```

### Recognition Engine Module

```python
from shared.logger import get_logger

logger = get_logger(__name__)

def recognize_gesture(self, frame):
    logger.debug("Starting gesture recognition")

    features = self.extract_features(frame)
    if features is None:
        logger.warning("Failed to extract features from frame")
        return None

    prediction = self.model.predict(features)
    logger.info(f"Gesture recognized: {prediction}")

    return prediction
```

### Action Mapper Module

```python
from shared.logger import get_logger

logger = get_logger(__name__)

def map_gesture_to_action(self, gesture):
    logger.debug(f"Mapping gesture: {gesture}")

    if gesture not in self.gesture_map:
        logger.warning(f"Unknown gesture: {gesture}")
        return None

    action = self.gesture_map[gesture]
    logger.info(f"Mapped {gesture} to action: {action}")

    return action
```

## Log Output Format

Console output (with colors):

```
2025-11-29 21:30:15 | INFO     | input_manager | Camera 0 opened successfully
2025-11-29 21:30:16 | WARNING  | input_manager | No frame captured yet after 1 second
2025-11-29 21:30:16 | INFO     | input_manager | Continuous capture started
2025-11-29 21:30:17 | ERROR    | recognition_engine | Failed to extract features
```

File output (without colors):

```
2025-11-29 21:30:15 | INFO     | input_manager | Camera 0 opened successfully
2025-11-29 21:30:16 | WARNING  | input_manager | No frame captured yet after 1 second
2025-11-29 21:30:16 | INFO     | input_manager | Continuous capture started
2025-11-29 21:30:17 | ERROR    | recognition_engine | Failed to extract features
```

## Best Practices

1. **Use appropriate log levels**

   - DEBUG: Internal state, variable values
   - INFO: Normal operations, milestones
   - WARNING: Recoverable issues, degraded performance
   - ERROR: Operation failures that need attention
   - CRITICAL: System-threatening errors

2. **Include context in messages**

   ```python
   # Good
   logger.error(f"Failed to open camera {camera_index}: {error_message}")

   # Less helpful
   logger.error("Camera error")
   ```

3. **Log exceptions properly**

   ```python
   try:
       risky_operation()
   except Exception as e:
       logger.error(f"Operation failed: {e}", exc_info=True)
   ```

4. **Avoid logging in tight loops**

   ```python
   # Bad - logs every frame
   for frame in video:
       logger.debug(f"Processing frame {i}")

   # Better - log milestones
   for i, frame in enumerate(video):
       if i % 100 == 0:
           logger.debug(f"Processed {i} frames")
   ```

5. **Use module-specific loggers**

   ```python
   # Always use __name__ for the logger
   logger = get_logger(__name__)

   # This creates hierarchical loggers like:
   # input_manager
   # recognition_engine
   # action_mapper
   ```

## Configuration Examples

### Development Setup

```python
# config/development.py
from shared.logger import configure_logging
import logging

configure_logging(
    level=logging.DEBUG,
    enable_file_logging=False
)
```

### Production Setup

```python
# config/production.py
from shared.logger import configure_logging
import logging

configure_logging(
    level=logging.INFO,
    enable_file_logging=True,
    log_dir="/var/log/gesture-recognition"
)
```

### Testing Setup

```python
# tests/conftest.py
from shared.logger import configure_logging
import logging

configure_logging(
    level=logging.WARNING,  # Reduce noise during tests
    enable_file_logging=False
)
```
