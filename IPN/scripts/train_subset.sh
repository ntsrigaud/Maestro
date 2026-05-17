#!/bin/bash
# Train IPN-hand models on the 9-class gesture subset.
#
# Run from the IPN-hand repo root on the server.
# Edit the path variables in the CONFIG section below before running.
#
# Usage:
#   bash scripts/train_subset.sh

set -e

# ── CONFIG — edit these ───────────────────────────────────────────────────────

# Root of the IPN-hand repo on the server
REPO_DIR="$HOME/.Neil/Gesture-Control-System/IPN-hand"

# Path to the extracted frames directory (output of extract_frames.sh)
FRAMES_DIR="$HOME/.Neil/Gesture-Control-System/data/IPN/frames"

# Path to the subset JSON (output of prepare_subset_json.py)
ANNOT_JSON="$REPO_DIR/annotation_ipnGesture/ipn_subset.json"

# Where to save checkpoints and logs
RESULT_DIR="$REPO_DIR/results_subset"

# Use all available GPUs. nn.DataParallel in model.py handles multi-GPU automatically.
# To restrict to specific GPUs, set e.g. CUDA_VISIBLE_DEVICES=0,1
unset CUDA_VISIBLE_DEVICES

# Number of dataloader workers — set to number of CPU cores available
N_THREADS=8

# ── Number of classes in your subset ─────────────────────────────────────────
# D0X B0A B0B G03 G04 G05 G06 G10 G11  →  9 classes
N_CLASSES=9

# ─────────────────────────────────────────────────────────────────────────────

mkdir -p "$RESULT_DIR"
cd "$REPO_DIR"

echo "============================================================"
echo "  Training: ResNetL-10  (lightweight, fast)"
echo "  Classes:  $N_CLASSES"
echo "  Frames:   $FRAMES_DIR"
echo "  Output:   $RESULT_DIR"
echo "============================================================"

python main.py \
    --video_path        "$FRAMES_DIR" \
    --annotation_path   "$ANNOT_JSON" \
    --result_path       "$RESULT_DIR" \
    --dataset           ipn \
    --model             resnetl \
    --model_depth       10 \
    --resnet_shortcut   A \
    --modality          RGB \
    --sample_size       112 \
    --sample_duration   16 \
    --n_classes         $N_CLASSES \
    --n_finetune_classes $N_CLASSES \
    --batch_size        32 \
    --n_epochs          100 \
    --learning_rate     0.01 \
    --lr_steps          30 60 90 \
    --momentum          0.9 \
    --dampening         0.9 \
    --weight_decay      1e-3 \
    --train_crop        random \
    --n_val_samples     1 \
    --test_subset       val \
    --n_threads         $N_THREADS \
    --checkpoint        5 \
    --store_name        ipn_subset_resnetl10 \
    2>&1 | tee "$RESULT_DIR/train_resnetl10.log"

echo ""
echo "Training complete. Best model saved to:"
echo "  $RESULT_DIR/ipn_subset_resnetl10_resnetl-10_best.pth"
