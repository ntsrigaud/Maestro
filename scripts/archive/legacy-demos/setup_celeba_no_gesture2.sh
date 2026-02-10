#!/bin/bash
# Download and setup CelebA dataset for no_gesture2 class

set -e

# Determine project root dynamically
PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
CELEBA_DIR="$PROJECT_ROOT/notebooks/data/celeba"
HAGRID_DIR="$PROJECT_ROOT/notebooks/data/hagrid_subset"

echo "======================================================================"
echo "CELEBA DATASET SETUP FOR NO_GESTURE2"
echo "======================================================================"
echo ""

# Create directory
mkdir -p "$CELEBA_DIR"
cd "$CELEBA_DIR"

echo "Downloading CelebA dataset..."
echo "This will take 10-20 minutes (~1.4GB download)"
echo ""

# Try multiple sources
echo "Attempting download from Kaggle mirror..."

# Method 1: Try direct Kaggle download with their API
if command -v kaggle &> /dev/null; then
    echo "Using Kaggle CLI..."
    kaggle datasets download -d jessicali9530/celeba-dataset
    unzip -q celeba-dataset.zip
    echo "✓ Downloaded via Kaggle"
else
    echo "Kaggle CLI not available, trying alternative..."
    
    # Method 2: Use wget with a mirror
    echo "Downloading from alternative source..."
    
    # CelebA official partitions (smaller, for testing)
    wget -c "https://www.dropbox.com/s/d1kjpkqklf0uw77/celeba.zip?dl=1" -O celeba.zip || {
        echo ""
        echo "======================================================================"
        echo "MANUAL DOWNLOAD REQUIRED"
        echo "======================================================================"
        echo ""
        echo "Please manually download CelebA dataset:"
        echo ""
        echo "Option 1 (Recommended): Kaggle"
        echo "  1. Visit: https://www.kaggle.com/datasets/jessicali9530/celeba-dataset"
        echo "  2. Click 'Download' (requires Kaggle login)"
        echo "  3. Move downloaded zip to: $CELEBA_DIR/"
        echo "  4. Extract: cd $CELEBA_DIR && unzip celeba-dataset.zip"
        echo ""
        echo "Option 2: Google Drive"
        echo "  1. Visit: https://drive.google.com/drive/folders/0B7EVK8r0v71pTUZsaXdaSnZBZzg"
        echo "  2. Download 'img_align_celeba.zip'"
        echo "  3. Move to: $CELEBA_DIR/"
        echo "  4. Extract: cd $CELEBA_DIR && unzip img_align_celeba.zip"
        echo ""
        echo "After download, run:"
        echo "  python $PROJECT_ROOT/scripts/build_no_gesture2_simple.py \\"
        echo "    --source_dir $CELEBA_DIR/img_align_celeba \\"
        echo "    --output_dir $HAGRID_DIR \\"
        echo "    --target_count 7000"
        echo ""
        exit 1
    }
    
    echo "Extracting..."
    unzip -q celeba.zip
    echo "✓ Extraction complete"
fi

# Find the image directory
if [ -d "img_align_celeba" ]; then
    IMG_DIR="img_align_celeba"
elif [ -d "celeba/img_align_celeba" ]; then
    IMG_DIR="celeba/img_align_celeba"
else
    echo "Looking for image directory..."
    IMG_DIR=$(find . -type d -name "*celeb*" -o -name "*img*" | head -1)
fi

if [ -z "$IMG_DIR" ]; then
    echo "❌ Could not find CelebA images"
    echo "Please check $CELEBA_DIR for extracted files"
    exit 1
fi

IMAGE_COUNT=$(find "$IMG_DIR" -name "*.jpg" -o -name "*.png" | wc -l)
echo ""
echo "✓ Found $IMAGE_COUNT images in $IMG_DIR"
echo ""

# Run the script to distribute images
echo "======================================================================"
echo "DISTRIBUTING CELEBA IMAGES TO no_gesture2 CLASS"
echo "======================================================================"
echo ""

python "$PROJECT_ROOT/scripts/build_no_gesture2_simple.py" \
    --source_dir "$CELEBA_DIR/$IMG_DIR" \
    --output_dir "$HAGRID_DIR" \
    --target_count 7000 \
    --seed 42

echo ""
echo "======================================================================"
echo "✅ SETUP COMPLETE"
echo "======================================================================"
echo ""
echo "The no_gesture2 class now contains face images from CelebA!"
echo "This better represents users sitting in front of their computer with no hands visible."
echo ""
echo "Next step: Run the training notebook with 24 classes"
