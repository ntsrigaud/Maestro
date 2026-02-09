# Gesture Recognition Demo - Quick Start

## Overview

The `demo_gesture_recognition.py` script provides **end-to-end real-time gesture recognition** by integrating:

- Camera input via `InputManager`
- Hand detection via MediaPipe `HandDetector`
- Hand cropping and preprocessing via `HandCropper`
- Gesture classification via fine-tuned ViT-B/16 model

## Prerequisites

1. **Trained model checkpoint**: `best_model.pth` (from Google Colab training)
2. **Dependencies installed**:
   ```bash
   pip install torch torchvision timm opencv-python mediapipe pillow
   ```
3. **Working camera** connected to your system

## Usage

### Basic Usage

```bash
python scripts/demo_gesture_recognition.py --checkpoint notebooks/checkpoints/best_model.pth
```

### With Options

```bash
# Use specific camera (if you have multiple)
python scripts/demo_gesture_recognition.py \
    --checkpoint notebooks/checkpoints/best_model.pth \
    --camera 1

# Show cropped hand windows for debugging
python scripts/demo_gesture_recognition.py \
    --checkpoint notebooks/checkpoints/best_model.pth \
    --show-cropped

# Adjust confidence threshold (only show predictions above 70%)
python scripts/demo_gesture_recognition.py \
    --checkpoint notebooks/checkpoints/best_model.pth \
    --confidence 0.7

# Force specific device (CPU/GPU/MPS)
python scripts/demo_gesture_recognition.py \
    --checkpoint notebooks/checkpoints/best_model.pth \
    --device mps  # or cuda, cpu

# Detect up to 2 hands simultaneously
python scripts/demo_gesture_recognition.py \
    --checkpoint notebooks/checkpoints/best_model.pth \
    --max-hands 2
```

## Interactive Controls

While the demo is running:

- **C** - Toggle cropped hand view
- **R** - Reset statistics
- **Q** or **ESC** - Quit

## Features

### 1. Real-Time Gesture Classification

- Detects hands using MediaPipe (21 landmarks per hand)
- Crops hand regions with 20% padding
- Resizes to 224×224 (ViT-B/16 input size)
- Classifies 18 gesture types with confidence scores

### 2. Temporal Smoothing

- Averages predictions over last 5 frames
- Reduces flickering and improves stability
- Shows smoothed gesture and confidence

### 3. Color-Coded Confidence

- **Green**: High confidence (≥80%)
- **Orange**: Medium confidence (50-80%)
- **Red**: Low confidence (<50%)

### 4. Multi-Hand Support

- Tracks up to 2 hands simultaneously
- Independent gesture recognition per hand
- Left/Right handedness detection

### 5. Performance Monitoring

- Real-time FPS counter
- Total hands detected
- Total gestures classified
- Device type (CPU/GPU/MPS)

## Recognized Gestures

The model recognizes 18 gesture classes:

1. **call** - Phone gesture
2. **dislike** - Thumbs down
3. **fist** - Closed fist
4. **four** - Four fingers up
5. **like** - Thumbs up
6. **mute** - Finger over lips
7. **ok** - OK sign
8. **one** - One finger up
9. **palm** - Open palm
10. **peace** - Peace sign
11. **peace_inverted** - Inverted peace
12. **rock** - Rock/metal sign
13. **stop** - Stop hand
14. **stop_inverted** - Inverted stop
15. **three** - Three fingers up
16. **three2** - Alternative three
17. **two_up** - Two fingers up
18. **two_up_inverted** - Inverted two

## Architecture Flow

```
Camera Feed (InputManager)
    ↓
Hand Detection (MediaPipe HandDetector)
    ↓
Hand Cropping (HandCropper)
    ↓ (224×224 RGB, normalized)
Gesture Classification (ViT-B/16)
    ↓
Temporal Smoothing (5-frame average)
    ↓
Display with Labels & Confidence
```

## Expected Performance

- **FPS**: 20-30 fps (depends on device)
- **Latency**: 30-50ms per frame
- **Accuracy**: 75-85% (on HaGRID test set)
- **Memory**: ~500MB GPU / 1GB RAM

## Troubleshooting

### Camera Not Found

```bash
# List available cameras
ls /dev/video*

# Try different camera index
python scripts/demo_gesture_recognition.py --checkpoint ... --camera 1
```

### Low FPS

- Use GPU/MPS: `--device cuda` or `--device mps`
- Reduce max hands: `--max-hands 1`
- Close other applications

### Model Not Loading

```bash
# Verify checkpoint exists
ls -lh notebooks/checkpoints/best_model.pth

# Check file size (~330MB expected)
```

### Import Errors

```bash
# Reinstall dependencies
pip install torch torchvision timm opencv-python mediapipe pillow

# Verify imports
python -c "import torch, timm, cv2, mediapipe; print('OK')"
```

## Example Output

```
Initializing gesture recognition demo...
Device: mps
Loading model from notebooks/checkpoints/best_model.pth...
✓ Initialization complete

============================================================
Gesture Recognition Demo Running
============================================================

Controls:
  C - Toggle cropped hand view
  R - Reset statistics
  Q/ESC - Quit

[Video window shows:]
- FPS: 28.3
- Hands: 2
- Gestures: 142
- Device: mps

[Each detected hand shows:]
- Bounding box (green/orange/red based on confidence)
- Handedness (Left/Right)
- Predicted gesture
- Confidence percentage
```

## Integration with Project

This demo integrates seamlessly with the existing project architecture:

- Uses `InputManager` from `src/input_manager/`
- Uses `HandDetector` and `HandCropper` from `src/recognition_engine/hand_detection/`
- Follows the same event-driven design pattern
- Compatible with existing logging and configuration systems

## Next Steps

1. **Test with different lighting conditions**
2. **Evaluate accuracy on real-world scenarios**
3. **Integrate with action mapper** (future work)
4. **Add gesture-to-command mapping**
5. **Optimize inference speed** (model quantization, TensorRT)

## Credits

- **Hand Detection**: MediaPipe by Google
- **Model**: ViT-B/16 from `timm` library
- **Dataset**: HaGRID by Samsung AI Center
- **Framework**: PyTorch
