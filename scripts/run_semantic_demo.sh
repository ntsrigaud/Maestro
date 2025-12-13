#!/bin/bash
# Quick start script for semantic gesture recognition demo

echo "==================================================="
echo "Semantic Gesture Recognition Demo - Quick Start"
echo "==================================================="
echo ""

# Check if checkpoint exists
CHECKPOINT_PATH="notebooks/checkpoints/semantic_model.pth"

if [ ! -f "$CHECKPOINT_PATH" ]; then
    echo "⚠ Semantic model not found!"
    echo ""
    echo "Downloading from Google Drive..."
    echo ""
    
    # Try to download
    python3 scripts/download_semantic_model.py
    
    if [ $? -ne 0 ]; then
        echo ""
        echo "❌ Download failed!"
        echo ""
        echo "Manual download instructions:"
        echo "  1. Visit: https://drive.google.com/file/d/1SBeHhBS30dT6XlzEL4LXlBwxCqa-QeTE/view"
        echo "  2. Download the file"
        echo "  3. Save to: $CHECKPOINT_PATH"
        echo ""
        exit 1
    fi
fi

echo "✓ Semantic model found"
echo ""

# Check Python dependencies
echo "Checking dependencies..."
python3 -c "import torch, timm, cv2, mediapipe, sentence_transformers" 2>/dev/null

if [ $? -ne 0 ]; then
    echo "⚠ Missing dependencies. Installing..."
    pip install torch torchvision timm opencv-python mediapipe sentence-transformers pillow gdown
else
    echo "✓ All dependencies installed"
fi

echo ""
echo "Starting semantic gesture recognition demo..."
echo ""
echo "Features:"
echo "  - Zero-shot gesture understanding"
echo "  - Real-time semantic similarity matching"
echo "  - Top-5 prediction display"
echo "  - Live confidence visualization"
echo ""
echo "Controls:"
echo "  - Press Q or ESC to quit"
echo ""
echo "=" * 50
echo ""

# Run the demo
python3 scripts/demo_semantic_gesture.py \
    --checkpoint "$CHECKPOINT_PATH" \
    --confidence 0.3

echo ""
echo "Demo finished!"
