# Scripts Directory

This directory contains utility scripts for testing, demonstration, and verification of the Hand Gesture Recognition System components.

## Test Scripts

### Thread Safety Tests

- **`test_threading_event.py`** - Tests threading.Event implementation with camera capture
- **`test_event_logic.py`** - Unit tests for Event logic without camera hardware
- **`test_event_based_startup.py`** - Demonstrates efficiency of Event-based first frame notification

### Feature Tests

- **`test_framedata_methods.py`** - Verifies FrameData helper methods are accessible (timestamp conversion, FPS calculation)
- **`test_hand_detection.py`** - Comprehensive tests for hand detection submodule
  - HandDetector initialization and detection
  - HandCropper with ROI extraction
  - Frame boundary handling
  - Multi-hand detection
  - Preprocessing pipeline validation

## Demonstration Scripts

- **`demo_timestamp_usage.py`** - Interactive demonstration of timestamp precision and usage patterns

  - Shows nanosecond precision at different scales
  - Frame-to-frame timing measurements
  - LSTM sequence timing analysis
  - System latency measurement

- **`demo_hand_detection.py`** - Real-time hand detection visualization
  - Live camera feed with bounding boxes
  - Hand landmarks (21 points per hand)
  - Handedness labels and confidence scores
  - FPS counter and statistics
  - Cropped hand region windows
  - Interactive controls (toggle landmarks, cropped view, reset stats)

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
python scripts/demo_hand_detection.py

# Optional arguments for hand detection demo
python scripts/demo_hand_detection.py --camera 0 --max-hands 2
python scripts/demo_hand_detection.py --no-landmarks --no-cropped

# Run tests
python scripts/test_event_based_startup.py
python scripts/test_framedata_methods.py
python scripts/test_hand_detection.py
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
