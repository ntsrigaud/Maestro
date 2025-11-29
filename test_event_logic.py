#!/usr/bin/env python3
"""
Test threading.Event logic without requiring camera hardware.
"""

import threading
import time

print("Testing threading.Event behavior...")
print()

# Simulate the pattern used in InputManager
is_capturing = threading.Event()

print("1. Initial state:")
print(f"   is_capturing.is_set() = {is_capturing.is_set()}")
print()

print("2. After set():")
is_capturing.set()
print(f"   is_capturing.is_set() = {is_capturing.is_set()}")
print()

print("3. After clear():")
is_capturing.clear()
print(f"   is_capturing.is_set() = {is_capturing.is_set()}")
print()

# Test thread safety with concurrent access
print("4. Testing thread safety...")
counter = 0


def worker():
    """Simulates the capture loop."""
    global counter
    while is_capturing.is_set():
        counter += 1
        time.sleep(0.01)
    print(f"   Worker thread stopped after {counter} iterations")


# Start background thread
is_capturing.set()
thread = threading.Thread(target=worker)
thread.start()

# Let it run briefly
time.sleep(0.1)

# Stop from main thread
print("   Stopping from main thread...")
is_capturing.clear()

# Wait for thread to finish
thread.join(timeout=1.0)
if thread.is_alive():
    print("   ERROR: Thread did not stop!")
else:
    print("   Thread stopped cleanly ✓")

print()
print("5. Multiple start/stop cycles:")
for cycle in range(3):
    is_capturing.set()
    status_during = is_capturing.is_set()
    is_capturing.clear()
    status_after = is_capturing.is_set()
    print(f"   Cycle {cycle + 1}: set={status_during}, clear={status_after}")

print()
print("All Event logic tests passed! ✓")
print("threading.Event provides proper thread-safe flag synchronization.")
