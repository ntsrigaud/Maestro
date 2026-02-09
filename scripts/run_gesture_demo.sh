#!/bin/bash
# Quick start script for gesture recognition demo

echo "==================================================="
echo "Gesture Recognition Demo - Quick Start"
echo "==================================================="
echo ""

# Check if checkpoint exists
if [ ! -f "notebooks/checkpoints/best_model.pth" ]; then
    echo "Error: Model checkpoint not found!"
    echo ""
    echo "Please download or copy your trained model to:"
    echo "  notebooks/checkpoints/best_model.pth"
    echo ""
    echo "If training on Google Colab, download the checkpoint:"
    echo "  from google.colab import files"
    echo "  files.download('checkpoints/best_model.pth')"
    echo ""
    exit 1
fi

echo "✓ Model checkpoint found"
echo ""

# Check Python dependencies
echo "Checking dependencies..."
python3 -c "import torch, timm, cv2" 2>/dev/null
if [ $? -ne 0 ]; then
    echo "Missing dependencies. Installing..."
    pip install torch torchvision timm opencv-python pillow
else
    echo "✓ All dependencies installed"
fi

echo ""
echo "Starting gesture recognition demo..."
echo "Press Q or ESC to quit"
echo ""

# Run the demo
python3 scripts/demo_gesture_recognition.py \
    --checkpoint notebooks/checkpoints/best_model.pth \
    --max-hands 2 \
    --confidence 0.5

echo ""
echo "Demo finished!"
