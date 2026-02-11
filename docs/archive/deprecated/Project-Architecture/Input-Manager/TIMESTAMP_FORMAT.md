# Timestamp Format Specification

## Overview

The Input Manager uses **nanosecond precision integer timestamps** (`time.time_ns()`) for all frame metadata. This document explains the rationale and usage patterns.

## Format Specification

```python
timestamp: int  # nanoseconds since Unix epoch (January 1, 1970, 00:00:00 UTC)
```

### Example Values

```python
# Current time in nanoseconds
timestamp = 1732924800000000000  # ~November 30, 2025

# Time between frames at 30 FPS
frame_delta = 33333333  # ~33.33 milliseconds in nanoseconds

# Time between frames at 60 FPS
frame_delta = 16666667  # ~16.67 milliseconds in nanoseconds
```

## Rationale

### 1. **Precision Requirements**

The gesture recognition system requires precise temporal information for:

- **LSTM Temporal Modeling**: Frame sequences (8-32 frames) need accurate timing for dynamic gesture detection
- **Motion Analysis**: Computing velocity and acceleration of hand movements
- **Frame Rate Calculation**: Measuring actual capture rate vs. target FPS
- **Latency Metrics**: System performance monitoring (<100ms target)

**Nanosecond precision** provides:

- Sub-millisecond accuracy (1ns = 0.000001ms)
- More than sufficient for video frame timing (30-60 FPS)
- Headroom for future high-speed camera support (120+ FPS)

### 2. **32-bit Overflow Prevention**

**Problem**: Milliseconds since epoch will overflow 32-bit signed integers in 2038:

```python
# Milliseconds since epoch
ms_2038 = 2147483647  # Max 32-bit signed int
# System crashes on 2038-01-19

# Nanoseconds require 64-bit from the start
ns_2025 = 1732924800000000000  # Already exceeds 32-bit range
```

**Solution**: Using nanoseconds forces 64-bit integer handling:

- Python `int` type is arbitrary precision (no overflow)
- Systems must use 64-bit representation from day one
- Future-proof for decades (64-bit nanoseconds won't overflow until year 2262)

### 3. **Performance Benefits**

**Integer arithmetic advantages**:

```python
# Clean, fast integer operations
delta = frame2.timestamp - frame1.timestamp  # No floating-point errors
is_later = frame2.timestamp > frame1.timestamp  # Direct comparison
```

**Compared to float alternatives**:

- No accumulated floating-point precision loss
- Faster comparison and sorting operations
- Exact representation (no rounding errors)

### 4. **Cross-Module Compatibility**

Downstream modules (Recognition Engine, Action Mapper) benefit from:

- **Consistent ordering**: Integer comparison for sequence sorting
- **Precise deltas**: Computing inter-frame timing for LSTM input
- **Easy conversion**: Simple division to get milliseconds/seconds

## Usage Patterns

### Basic Frame Timing

```python
from input_manager import InputManager, FrameData

manager = InputManager(camera_index=0)
manager.start_continuous_capture()

# Get frames
frame1 = manager.get_latest_frame()
time.sleep(0.1)
frame2 = manager.get_latest_frame()

# Calculate time difference
delta_ms = FrameData.time_delta_ms(frame1, frame2)
print(f"Time between frames: {delta_ms:.2f}ms")

# Calculate FPS
fps = FrameData.calculate_fps(frame1, frame2)
print(f"Instantaneous FPS: {fps:.1f}")
```

### Sequence Timing for LSTM

```python
# Collect frame sequence for dynamic gesture
sequence = []
for _ in range(16):  # 16-frame window
    frame = manager.get_latest_frame()
    if frame:
        sequence.append(frame)
    time.sleep(1/30)  # ~30 FPS

# Verify temporal consistency
for i in range(len(sequence) - 1):
    delta = FrameData.time_delta_ms(sequence[i], sequence[i+1])
    print(f"Frame {i} -> {i+1}: {delta:.2f}ms")

# Pass to Recognition Engine with precise timing
recognition_engine.process_dynamic_gesture(sequence)
```

### Performance Monitoring

```python
# Track system latency
capture_start = time.time_ns()
frame = manager.get_latest_frame()
recognition_result = recognition_engine.process(frame)
action = action_mapper.translate(recognition_result)
os_controller.execute(action)
total_latency_ns = time.time_ns() - capture_start

# Convert to milliseconds for reporting
latency_ms = total_latency_ns / 1_000_000
print(f"End-to-end latency: {latency_ms:.2f}ms")

# Check if meeting <100ms target
if latency_ms < 100:
    logger.info(f"✓ Performance target met: {latency_ms:.2f}ms")
else:
    logger.warning(f"✗ Performance target missed: {latency_ms:.2f}ms")
```

## Helper Methods

The `FrameData` class provides convenient conversion methods:

```python
@dataclass
class FrameData:
    timestamp: int  # nanoseconds

    def timestamp_ms(self) -> float:
        """Get timestamp in milliseconds."""
        return self.timestamp / 1_000_000

    def timestamp_seconds(self) -> float:
        """Get timestamp in seconds."""
        return self.timestamp / 1_000_000_000

    @staticmethod
    def time_delta_ms(frame1, frame2) -> float:
        """Time difference in milliseconds."""
        return (frame2.timestamp - frame1.timestamp) / 1_000_000

    @staticmethod
    def calculate_fps(frame1, frame2) -> float:
        """Instantaneous FPS between frames."""
        delta_ns = frame2.timestamp - frame1.timestamp
        return 1_000_000_000 / delta_ns if delta_ns > 0 else 0.0
```

## Conversion Reference

```python
# Nanosecond conversions
1 second     = 1_000_000_000 nanoseconds
1 millisecond = 1_000_000 nanoseconds
1 microsecond = 1_000 nanoseconds

# Common operations
timestamp_ms = timestamp_ns / 1_000_000
timestamp_s  = timestamp_ns / 1_000_000_000
delta_ns = frame2.timestamp - frame1.timestamp
delta_ms = delta_ns / 1_000_000
fps = 1_000_000_000 / delta_ns
```

## Platform Considerations

### Python `int` Type

- **Arbitrary precision**: No fixed size limit in Python 3
- **Automatic promotion**: No overflow concerns
- **Memory**: Slightly larger than C integers, but negligible for timestamps

### System Clock Resolution

- **Linux**: `CLOCK_MONOTONIC` typically nanosecond resolution
- **macOS**: `mach_absolute_time()` nanosecond precision
- **Windows**: `QueryPerformanceCounter()` high resolution

Python's `time.time_ns()` provides best available resolution on each platform.

## Migration from Milliseconds

If upgrading from millisecond timestamps:

```python
# Old format (milliseconds)
old_timestamp_ms = 1732924800000

# Convert to nanoseconds
new_timestamp_ns = old_timestamp_ms * 1_000_000

# Or use current time
new_timestamp_ns = time.time_ns()
```

## Best Practices

✅ **DO**:

- Use `time.time_ns()` for new timestamps
- Store timestamps as `int` type
- Use helper methods for conversion
- Document timestamp format in interfaces

❌ **DON'T**:

- Convert to float unless necessary for computation
- Assume millisecond precision when nanoseconds available
- Mix timestamp formats across modules
- Use string timestamps for storage

## Future Considerations

This format is designed to be future-proof:

- ✅ High-speed cameras (120+ FPS)
- ✅ Sub-millisecond latency requirements
- ✅ Precise temporal modeling for AI
- ✅ Cross-platform compatibility
- ✅ No overflow until year 2262

## References

- Python `time.time_ns()`: [Python Docs](https://docs.python.org/3/library/time.html#time.time_ns)
- Unix Time: [Wikipedia](https://en.wikipedia.org/wiki/Unix_time)
- Year 2038 Problem: [Wikipedia](https://en.wikipedia.org/wiki/Year_2038_problem)
