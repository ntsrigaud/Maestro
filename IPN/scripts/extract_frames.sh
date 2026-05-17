#!/bin/bash
# Extracts frames from IPN .avi videos into the folder structure expected by the IPN-hand repo.
# Frame naming convention: {video_name}_{frame_number:06d}.jpg
#
# Usage (run from the IPN-hand repo root on the server):
#   bash scripts/extract_frames.sh /path/to/IPN/videos /path/to/IPN/frames
#
# Example:
#   bash scripts/extract_frames.sh ~/.Neil/Gesture-Control-System/data/IPN/videos \
#                                  ~/.Neil/Gesture-Control-System/data/IPN/frames

VIDEO_DIR="${1:?Usage: $0 <video_dir> <output_frames_dir>}"
FRAMES_DIR="${2:?Usage: $0 <video_dir> <output_frames_dir>}"

if ! command -v ffmpeg &>/dev/null; then
    echo "ERROR: ffmpeg not found. Install with: sudo apt install ffmpeg"
    exit 1
fi

mkdir -p "$FRAMES_DIR"

total=$(ls "$VIDEO_DIR"/*.avi 2>/dev/null | wc -l)
if [ "$total" -eq 0 ]; then
    echo "ERROR: No .avi files found in $VIDEO_DIR"
    exit 1
fi

count=0
for video in "$VIDEO_DIR"/*.avi; do
    basename=$(basename "$video" .avi)
    outdir="$FRAMES_DIR/$basename"

    # Skip if already extracted (check if folder has any jpgs)
    if [ -d "$outdir" ] && [ "$(ls "$outdir"/*.jpg 2>/dev/null | wc -l)" -gt 0 ]; then
        count=$((count + 1))
        echo "[$count/$total] Skipping (already extracted): $basename"
        continue
    fi

    mkdir -p "$outdir"
    # -q:v 3 = good quality JPEG; -start_number 1 = frames start at 000001
    ffmpeg -i "$video" \
        -q:v 3 \
        -start_number 1 \
        "$outdir/${basename}_%06d.jpg" \
        -hide_banner -loglevel error

    count=$((count + 1))
    echo "[$count/$total] Extracted: $basename"
done

echo ""
echo "Done. Frames saved to: $FRAMES_DIR"
