---
language: en
license: mit
library_name: pytorch
tags:
  - gesture-recognition
  - human-computer-interaction
  - mediapipe
  - lstm
  - transfer-learning
pipeline_tag: image-classification
---

# maestro-lstm-ld-cong

Model card for the Jester→LD-CONG hybrid fine-tuned model.

## Model Details

- Architecture: `EnhancedTwoStreamLSTM` (`two_stream_attn_v1`)
- Approx parameter count: ~2.1M
- Ablation flags: `motion_temporal_conv=False`, `motion_attn_pool=False` (baseline A — confirmed optimal by ablation)
- Backbone:
  - Two-stream BiLSTM encoder (hidden_size=96, num_layers=4)
  - Multi-head self-attention per stream (8 heads, proj_dim=96)
  - Cross-stream MLP gating
- Classification head:
  - `LayerNorm(384) -> Linear(384,192) -> GELU -> Dropout -> Linear(192,C)`
- Input shape: `(batch, 16, 147)` — W=16 frames (LD-CONG) vs W=32 for Jester pretraining
- Feature schema: `feature-schema-v5`
- Class counts:
  - Pretraining: 27 (Jester)
  - Fine-tuning: 12 (LD-CONG hybrid; `fist` and `thumb_up` excluded via `training.exclude_classes`)

## Training Data

### Phase 1: Jester pretraining

- Dataset: 20BN-Jester-v1
- Labels: 27 gesture classes, W=32
- Input source: per-video JPEG sequences with MediaPipe Holistic extraction
- Pretrained checkpoint: `two_stream_attn_v1_20260513T155730Z`

### Phase 2: LD-CONG Hybrid fine-tuning

- Dataset: LD-CONG hybrid (`data/processed/ld_cong_hybrid_dataset.npz`)
- Three data sources:
  1. LD-CONG laboratory recordings (CSV-annotated RGB video)
  2. 20BN-Jester-v1 clips center-cropped to W=16
  3. IPN Hand clips center-cropped to W=16 (`point_one`, `point_two`, `unknown`)
- Labels (12 active classes; `fist` and `thumb_up` excluded):
  - `palm`, `pinch`, `click`
  - `swiping_right`, `swiping_left`, `swiping_down`, `swiping_up`
  - `zooming_in_full_hand`, `zooming_out_full_hand`
  - `point_one`, `point_two`
  - `unknown`
- Dataset size: train≈27,053 · val≈5,242 windows (W=16)
- Split: stratified 70/15/15 train/val/test

## Training Procedure

### Phase 1 config (Jester)

- Config file: `configs/model/two_stream_attn_jester.yaml`
- Epochs: 80
- Batch size: 256
- Optimizer: AdamW
- Learning rate: `1e-4`
- Weight decay: `5e-4`
- Label smoothing: `0.05`
- Class weighting: enabled

### Phase 2 config (LD-CONG Hybrid)

- Config file: `configs/model/two_stream_attn_ld_cong_12class.yaml`
- Stage A (head-only warm-up):
  - frozen backbone, new 12-class head only
  - epochs: 10
  - learning rate: `1e-3`
- Stage B (full fine-tune):
  - unfreeze all layers + EWC penalty
  - epochs: 60
  - learning rate: `3e-5`, `backbone_lr_multiplier=0.1`
  - EWC weight: `2000.0`
  - `max_samples_per_class: 3000`
- Batch size: 128
- Dropout: `0.35`

## Intended Use

- Real-time hand-gesture control in the LD-CONG lab environment driven by MediaPipe landmarks
- Research baseline for Jester→LD-CONG transfer learning
- Comparison with IPN-based and class-extended vocabulary models

## Limitations

- Depends on landmark quality; poor hand tracking degrades predictions
- Trained on LD-CONG laboratory conditions; domain shift across lighting, users, and viewpoints may reduce performance
- `fist` and `thumb_up` are excluded — the model will not recognise these classes; they map to `unknown` in practice
- Not a safety-certified model for critical decision systems

## Evaluation

Use the output of `notebooks/08_model_training_jester_ldcong_hybrid_12class.ipynb`
for exact final metrics (test accuracy, macro-F1, per-class recall, confusion matrix).

### Motion branch ablation summary

Motion-branch modifications (`motion_temporal_conv`, `motion_attn_pool`) were evaluated
in `notebooks/09_motion_branch_ablation.ipynb`. All variants (B_tcn, C_attn, D_both)
stayed within −0.36 pp of the baseline A, confirming the baseline flags are optimal
for this dataset size and task.
