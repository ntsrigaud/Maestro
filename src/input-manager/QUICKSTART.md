# InputManager Quick Start Guide

Get up and running with InputManager in 2 minutes!

## Installation

```bash
cd src/input-manager
pip install -r requirements.txt
python setup.py
```

## Quick Test

```bash
# Run examples
python example_usage.py
```

## Your First Program

Create `my_camera_app.py`:

```python
from input_manager import InputManager
import time

# Create manager
manager = InputManager(camera_index=0)

# Start capturing
manager.start_continuous_capture()

# Get 100 frames
for i in range(100):
    frame_data = manager.get_latest_frame()
    if frame_data:
        frame = frame_data.to_numpy()
        print(f"Frame {i}: {frame.shape}, timestamp={frame_data.timestamp}")
    time.sleep(0.033)  # ~30 FPS

# Cleanup
manager.stop_continuous_capture()
print("Done!")
```

Run it:

```bash
python my_camera_app.py
```

## Common Issues

### "No module named 'cv2'"

```bash
pip install opencv-python
```

### "Camera failed to open"

- Check if another app is using the camera
- Try a different camera_index (0, 1, 2, ...)
- On macOS: Grant camera permissions in System Preferences

## What's Next?

1. **Read the full API:** See `README.md`
2. **Try all examples:** Run `python example_usage.py`
3. **Integrate with your app:** Import and use!

## Need Help?

- See `README.md` for full documentation
- Check `example_usage.py` for usage patterns

Happy coding!
