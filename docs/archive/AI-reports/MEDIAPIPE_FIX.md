# MediaPipe Version Fix

## Problem

MediaPipe v0.10.31 has a different API that is incompatible with the current HandDetector implementation.

## Solution

Downgrade to MediaPipe v0.9.3.0 which uses the `solutions` API:

```bash
# Uninstall current version
pip uninstall mediapipe -y

# Install compatible version
pip install mediapipe==0.9.3.0

# Verify installation
python -c "import mediapipe as mp; print(mp.__version__); print(hasattr(mp, 'solutions'))"
```

Expected output:

```
0.9.3.0
True
```

## Then Run Test

```bash
python scripts/test_reproducibility.py --test 1 \
    --checkpoint notebooks/checkpoints/best_model.pth \
    --test-dir notebooks/data/hagrid_subset/test/ \
    --max-images 50
```

## Alternative: Update Code for v0.10+

If you prefer to use MediaPipe v0.10+, the HandDetector code needs to be updated to use the new Task API. This is a more complex change that requires rewriting the detection logic.

For now, **using v0.9.3.0 is the quickest path forward**.
