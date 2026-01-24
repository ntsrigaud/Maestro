#!/bin/bash
# Setup no_gesture2 class for gesture recognition training
# This script automates the entire process of adding the no_gesture2 class

set -e  # Exit on error

PROJECT_ROOT="/home/rachel/.Neil/CSU-Capstone-Project"
DATASET_ROOT="$PROJECT_ROOT/notebooks/data/hagrid_subset"
SOURCE_DIR="$PROJECT_ROOT/notebooks/data/coco_val2017/val2017"

echo "========================================================================"
echo "SETUP NO_GESTURE2 CLASS"
echo "========================================================================"
echo ""

# Step 1: Check if COCO dataset is downloaded
echo "Step 1: Checking COCO val2017 dataset..."
if [ ! -d "$SOURCE_DIR" ]; then
    echo "  ❌ COCO val2017 not found at: $SOURCE_DIR"
    echo ""
    echo "  Please download it first:"
    echo "    cd $PROJECT_ROOT"
    echo "    python scripts/download_no_gesture2_images.py --output ./notebooks/data/coco_val2017 --auto"
    echo ""
    exit 1
fi

IMAGE_COUNT=$(find "$SOURCE_DIR" -name "*.jpg" | wc -l)
echo "  ✓ Found $IMAGE_COUNT images in $SOURCE_DIR"
echo ""

# Step 2: Check/Install MediaPipe
echo "Step 2: Checking MediaPipe installation..."
if ! /home/rachel/.Neil/CSU-Capstone-Project/.venv/bin/python -c "import mediapipe" 2>/dev/null; then
    echo "  Installing MediaPipe..."
    /home/rachel/.Neil/CSU-Capstone-Project/.venv/bin/python -m pip install -q mediapipe opencv-python
    echo "  ✓ MediaPipe installed"
else
    echo "  ✓ MediaPipe already installed"
fi
echo ""

# Step 3: Run filtering script
echo "Step 3: Running hand detection filter..."
echo "  This will filter ~5000 images from COCO where hands are NOT visible"
echo "  Processing time: ~10-15 minutes"
echo ""

cd "$PROJECT_ROOT"
/home/rachel/.Neil/CSU-Capstone-Project/.venv/bin/python scripts/build_no_gesture2_class.py \
    --source "$SOURCE_DIR" \
    --output "$DATASET_ROOT" \
    --max-samples 5000 \
    --confidence 0.3 \
    --source-name "MS COCO 2017 Validation" \
    --license "CC BY 4.0" \
    --attribution "Microsoft COCO: Common Objects in Context" \
    --url "https://cocodataset.org/"

echo ""
echo "========================================================================"
echo "✓ SETUP COMPLETE"
echo "========================================================================"
echo ""
echo "Next steps:"
echo "  1. Verify the no_gesture2 class was created:"
echo "     ls -l $DATASET_ROOT/train/no_gesture2/"
echo "     ls -l $DATASET_ROOT/val/no_gesture2/"
echo "     ls -l $DATASET_ROOT/test/no_gesture2/"
echo ""
echo "  2. Check class count:"
echo "     ls $DATASET_ROOT/train/ | wc -l"
echo "     # Should show 24 (was 23, now 24)"
echo ""
echo "  3. Run the training notebook:"
echo "     # The notebook will automatically detect 24 classes"
echo "     # No code changes needed!"
echo ""
