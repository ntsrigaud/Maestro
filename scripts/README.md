# Scripts Directory

This directory contains utility scripts for testing, demonstration, and verification of the Hand Gesture Recognition System components.

## Test Scripts

### Thread Safety Tests

- **`test_threading_event.py`** - Tests threading.Event implementation with camera capture
- **`test_event_logic.py`** - Unit tests for Event logic without camera hardware
- **`test_event_based_startup.py`** - Demonstrates efficiency of Event-based first frame notification

### Feature Tests

- **`test_framedata_methods.py`** - Verifies FrameData helper methods are accessible (timestamp conversion, FPS calculation)

## Demonstration Scripts

- **`demo_timestamp_usage.py`** - Interactive demonstration of timestamp precision and usage patterns
  - Shows nanosecond precision at different scales
  - Frame-to-frame timing measurements
  - LSTM sequence timing analysis
  - System latency measurement

## Verification Scripts

- **`verify_install.py`** - Validates package installation and import functionality

## Running Scripts

All scripts should be run from the project root directory:

```bash
cd /Users/neil/dev/repos/CSU-Capstone-Project

# Run verification
python scripts/verify_install.py

# Run demonstrations (interactive)
python scripts/demo_timestamp_usage.py

# Run tests
python scripts/test_event_based_startup.py
python scripts/test_framedata_methods.py
```

## Requirements

Scripts require the package to be installed in editable mode:

```bash
pip install -e .
```

Some scripts require camera access and will prompt or fail gracefully if no camera is available.

## Adding New Scripts

When creating new test or utility scripts:

1. Place them in this `scripts/` directory
2. Use descriptive names with prefixes:
   - `test_*.py` - Unit or integration tests
   - `demo_*.py` - Interactive demonstrations
   - `verify_*.py` - Verification/validation scripts
   - `benchmark_*.py` - Performance benchmarks
3. Update this README with a brief description
4. Include proper error handling for missing hardware (cameras, etc.)
