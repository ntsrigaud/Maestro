#!/bin/bash
# Train all models on the 9-class gesture subset sequentially.
# Each model's best checkpoint and training log are saved to RESULT_DIR.
#
# Usage (run from the IPN-hand repo root on the server):
#   bash scripts/train_all_models.sh
#
# After all runs finish, collect results with:
#   python scripts/collect_results.py --result_dir results_subset

set -e

# ── CONFIG — edit these ───────────────────────────────────────────────────────
FRAMES_DIR="$HOME/.Neil/Gesture-Control-System/data/IPN/frames"
ANNOT_JSON="$(pwd)/annotation_ipnGesture/ipn_subset.json"
RESULT_DIR="$(pwd)/results_subset"
N_CLASSES=9
N_THREADS=8
N_EPOCHS=100
# LR_STEPS deliberately NOT set here — the paper's CLF training scripts don't
# pass --lr_steps, so the default [10, 25, 50, 80, 100] is used. We replicate
# that by passing no --lr_steps argument (see run_model below).

# Jester-pretrained weights (REQUIRED for reproducing paper accuracy on ResNeXt-101 and ResNet-50)
# Download and place in report_ipn/:
#   ResNeXt-101: https://drive.google.com/open?id=156fE3mO3YdFPY4pfreWYQn5sxQdu7Bmt  → jester_resnext_101_RGB_32.pth
#   ResNet-50:   https://drive.google.com/open?id=1X9uom_f0euHmhAgO8XNJUqUGH98saB7Z  → jester_resnet_50_RGB_16.pth
JESTER_RESNEXT="$(pwd)/report_ipn/jester_resnext_101_RGB_32.pth"
JESTER_RESNET50="$(pwd)/report_ipn/jester_resnet_50_RGB_16.pth"

# Use all available GPUs (nn.DataParallel handles it automatically)
unset CUDA_VISIBLE_DEVICES
# ─────────────────────────────────────────────────────────────────────────────

mkdir -p "$RESULT_DIR"

run_model() {
    local store_name=$1
    local model=$2
    local model_depth=$3
    local shortcut=$4
    local modality=$5
    local sample_dur=$6
    local batch=$7
    local extra_args=${8:-""}

    local log_file="$RESULT_DIR/train_${store_name}.log"

    if [ -f "$RESULT_DIR/${store_name}_${model}-${model_depth}_best.pth" ]; then
        echo "============================================================"
        echo "  SKIP (already trained): $store_name"
        echo "============================================================"
        return
    fi

    echo "============================================================"
    echo "  Training: $store_name"
    echo "  Model:    $model-$model_depth | Modality: $modality"
    echo "============================================================"

    # Note: --lr_steps is intentionally omitted to match the paper (uses default [10,25,50,80,100])
    python main.py \
        --video_path        "$FRAMES_DIR" \
        --annotation_path   "$ANNOT_JSON" \
        --result_path       "$RESULT_DIR" \
        --dataset           ipn \
        --model             "$model" \
        --model_depth       "$model_depth" \
        --resnet_shortcut   "$shortcut" \
        --modality          "$modality" \
        --sample_size       112 \
        --sample_duration   "$sample_dur" \
        --n_classes         $N_CLASSES \
        --n_finetune_classes $N_CLASSES \
        --batch_size        "$batch" \
        --n_epochs          $N_EPOCHS \
        --learning_rate     0.01 \
        --momentum          0.9 \
        --dampening         0.9 \
        --weight_decay      1e-3 \
        --train_crop        random \
        --n_val_samples     1 \
        --test_subset       val \
        --n_threads         $N_THREADS \
        --checkpoint        5 \
        --store_name        "$store_name" \
        $extra_args \
        2>&1 | tee "$log_file"

    echo "  Done: $store_name"
    echo ""
}

# ── Models to train ───────────────────────────────────────────────────────────
# All settings mirror the paper's original training scripts exactly.
# store_name                  model        depth  shortcut  modality  dur  batch  extra

# ResNetL-10 (from scratch — no Jester weights exist for this model)
run_model "resnetl10_RGB"     resnetl      10     A         RGB       16   32

# ResNet-10 (from scratch)
run_model "resnet10_RGB"      resnet       10     B         RGB       16   32

# ResNet-50 (with Jester pretraining if available, else from scratch)
if [ -f "$JESTER_RESNET50" ]; then
    run_model "resnet50_RGB"  resnet       50     B         RGB       16   32 \
        "--pretrain_path $JESTER_RESNET50 --pretrain_dataset jester"
else
    echo "WARNING: Jester ResNet-50 weights not found at $JESTER_RESNET50"
    echo "         Accuracy will be lower than paper. Download from:"
    echo "         https://drive.google.com/open?id=1X9uom_f0euHmhAgO8XNJUqUGH98saB7Z"
    run_model "resnet50_RGB"  resnet       50     B         RGB       16   32
fi

# C3D (from scratch)
run_model "c3d_RGB"           c3d          10     B         RGB       16   32

# MobileNetV2 (from scratch)
run_model "mobilenetv2_RGB"   mobilenetv2  0      B         RGB       16   32  "--width_mult 1.0"

# ShuffleNetV2 (from scratch)
run_model "shufflenetv2_RGB"  shufflenetv2 0      B         RGB       16   32  "--width_mult 1.0"

# ResNeXt-101 (with Jester pretraining — REQUIRED for paper-level accuracy; batch=32 as in paper)
if [ -f "$JESTER_RESNEXT" ]; then
    run_model "resnext101_RGB" resnext 101 B RGB 32 32 \
        "--resnext_cardinality 32 --pretrain_path $JESTER_RESNEXT --pretrain_dataset jester"
else
    echo "WARNING: Jester ResNeXt-101 weights not found at $JESTER_RESNEXT"
    echo "         Accuracy will be much lower than paper (83.59%). Download from:"
    echo "         https://drive.google.com/open?id=156fE3mO3YdFPY4pfreWYQn5sxQdu7Bmt"
    run_model "resnext101_RGB" resnext 101 B RGB 32 32 \
        "--resnext_cardinality 32"
fi

echo "============================================================"
echo "  All models trained. Run collect_results.py for the table."
echo "============================================================"
