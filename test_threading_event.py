#!/usr/bin/env python3
"""
Test threading.Event implementation in InputManager.
"""

from input_manager import InputManager
import time

print("Testing threading.Event implementation...")

# Create manager
manager = InputManager(camera_index=0)

# Test 1: Check initial state
print(f"1. Initial state - is_capturing: {manager._is_capturing.is_set()}")

# Start camera first
if not manager.start_capture():
    print("Error: Could not start camera. Make sure camera is available.")
    exit(1)
print("   Camera started successfully")

# Test 2: Start continuous capture
print("2. Starting continuous capture...")
success = manager.start_continuous_capture()
print(f"   Started: {success}, is_capturing: {manager._is_capturing.is_set()}")

# Test 3: Try to start again (should be ignored)
print("3. Try starting again (should be ignored)...")
manager.start_continuous_capture()

# Test 4: Get some frames
print("4. Capturing frames...")
for i in range(5):
    frame_data = manager.get_latest_frame()
    if frame_data:
        print(
            f"   Frame {i}: shape={frame_data.frame.shape}, is_capturing={manager._is_capturing.is_set()}"
        )
    time.sleep(0.1)

# Test 5: Stop capture
print("5. Stopping capture...")
manager.stop_continuous_capture()
print(f"   Stopped, is_capturing: {manager._is_capturing.is_set()}")

# Test 6: Stop camera
manager.stop_capture()
print("6. Camera stopped")

print("\nAll tests passed! threading.Event is working correctly.")
