#!/usr/bin/env python3
"""
Quick test to verify FrameData helper methods are available.
"""

import sys
import importlib

# Force fresh import
for module in ["input_manager", "input_manager.input_manager"]:
    if module in sys.modules:
        del sys.modules[module]

print("Testing FrameData import and methods...")
print("-" * 60)

from input_manager import FrameData
import numpy as np
import time

# Create test frames
frame1 = FrameData(
    frame=np.zeros((224, 224, 3), dtype=np.float32),
    timestamp=time.time_ns(),
    frame_number=1,
)

time.sleep(0.033)  # ~30 FPS delay

frame2 = FrameData(
    frame=np.zeros((224, 224, 3), dtype=np.float32),
    timestamp=time.time_ns(),
    frame_number=2,
)

print(f"\nFrame 1 timestamp (ns): {frame1.timestamp:,}")
print(f"Frame 2 timestamp (ns): {frame2.timestamp:,}")

# Test helper methods
print("\n" + "-" * 60)
print("Testing helper methods:")
print("-" * 60)

methods_to_test = [
    ("timestamp_ms", lambda: frame1.timestamp_ms()),
    ("timestamp_seconds", lambda: frame1.timestamp_seconds()),
    ("time_delta_ms", lambda: FrameData.time_delta_ms(frame1, frame2)),
    ("calculate_fps", lambda: FrameData.calculate_fps(frame1, frame2)),
]

all_passed = True
for method_name, test_func in methods_to_test:
    try:
        result = test_func()
        print(f"✓ {method_name:20s} = {result}")
    except AttributeError as e:
        print(f"✗ {method_name:20s} - NOT FOUND: {e}")
        all_passed = False
    except Exception as e:
        print(f"✗ {method_name:20s} - ERROR: {e}")
        all_passed = False

print("\n" + "-" * 60)
if all_passed:
    print("✓ All helper methods are accessible!")
    print("\nIf demo script still fails, try:")
    print("  1. Restart Python interpreter")
    print("  2. pip install -e . --force-reinstall --no-deps")
else:
    print("✗ Some methods are missing!")
    print("\nPlease reinstall the package:")
    print("  pip install -e .")

print("-" * 60)
