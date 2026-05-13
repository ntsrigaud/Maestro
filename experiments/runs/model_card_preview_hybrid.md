---
language:
- en
license: mit
tags:
- gesture-recognition
- hand-gesture
- pytorch
- mediapipe
- temporal-model
- lstm
- attention
- bidirectional
datasets:
- IPN-Hand
metrics:
- accuracy
- f1
model-index:
- name: two_stream_attn_v1_finetune_20260513T050407Z
  results:
  - task:
      type: gesture-recognition
    dataset:
      name: IPN Hand
      type: IPN-Hand
    metrics:
    - type: accuracy
      value: 0.9551
    - type: f1
      value: 0.9481
---

# two_stream_attn_v1_finetune_20260513T050407Z

A real-time hand gesture classifier trained on
a Hybrid Jester+IPN gesture dataset (Jester dynamic classes + IPN pointing classes).

This model is part of the **Maestro** pipeline that enables touchless
control of presentation and meeting software through hand gestures captured from a
standard webcam using MediaPipe for landmark extraction.

## Model Description

- **Architecture**: EnhancedTwoStreamLSTM (BiLSTM h=96×2, MHA 8 heads, proj=96, mean+max pool, MLP gate)
- **Parameters**: 2,099,434
- **Input**: `(batch, 32, 147)`
    — 32-frame sliding window at 30 FPS ≈ 1067 ms
- **Output**: Softmax logits over 10 gesture classes
- **Inference latency**: < 1 ms per call (CPU, single sample)
- **Feature schema**: `feature-schema-v5`

## Architecture

`EnhancedTwoStreamLSTM` splits the 147-dim feature vector into two parallel streams and
processes them through a BiLSTM + self-attention + MLP-gate pipeline:

```
Input (B, T=32, 147)
    │
    ├─ Stream A — Pose/Shape (73 dims)
    │   Linear+LN+GELU → 96
    │   2-layer BiLSTM (h=96) → (B, T, 192)
    │   LayerNorm → Self-MHA (8 heads) + residual + post-LN
    │   mean+max pool → pool_LN → ctx_a (B, 192)
    │
    ├─ Stream B — Motion/Dynamics (74 dims)
    │   (identical structure) → ctx_b (B, 192)
    │
    ├─ MLP cross-stream gate
    │   gate_a = Sigmoid(
    │     Linear(96→192)(
    │       Tanh(Linear(192→96)(ctx_b))))
    │   ctx_a  = LN(ctx_a × gate_a + ctx_a)
    │   gate_b = Sigmoid(
    │     Linear(96→192)(
    │       Tanh(Linear(192→96)(ctx_a))))
    │   ctx_b  = LN(ctx_b × gate_b + ctx_b)
    │
    └─ cat(ctx_a, ctx_b) → (384,)
       LN → Linear(384→192) → GELU → Dropout → Linear(192→10)
```

**Design rationale:**
- BiLSTMs encode temporal order via their recurrent cell state — no positional encoding needed.
- Mean+Max pooling captures both sustained gesture shape (mean) and transient click events (max).
- The 2-layer MLP gate provides non-linear cross-modal recalibration at ~37 K params
  (vs ~263 K for full MHA cross-attention with a degenerate mean-pooled query).

## Gesture Classes

| Class | Description |
|-------|-------------|
| `unknown` | Background / transition / no gesture |
| `point_one` | Single-finger pointing gesture (continuous laser-pointer control) |
| `point_two` | Two-finger pointing gesture (continuous annotation-pen control) |
| `stop_sign` | Static open palm facing camera (Jester class) |
| `swiping_down` | Vertical swipe downward (Jester class) |
| `swiping_left` | Horizontal swipe from right to left (Jester class) |
| `swiping_right` | Horizontal swipe from left to right (Jester class) |
| `swiping_up` | Vertical swipe upward (Jester class) |
| `zooming_in_full_hand` | Pinch-open / spread fingers away from each other (Jester class) |
| `zooming_out_full_hand` | Pinch-close / bring fingers together (Jester class) |

## Gesture Usage In Presentation System

| Class | Mode | Command | Runtime handling |
|-------|------|---------|------------------|
| `unknown` | `discrete` | `no_action` | No-op background class |
| `point_one` | `continuous` | `—` | Continuous tracker: LaserPointerTracker (bypasses discrete dispatcher) |
| `point_two` | `continuous` | `—` | Continuous tracker: AnnotationPenTracker (bypasses discrete dispatcher) |
| `stop_sign` | `discrete` | `erase_annotations` | Discrete command via GestureActivationController → CommandDispatcher |
| `swiping_down` | `discrete` | `stop_presentation` | Discrete command via GestureActivationController → CommandDispatcher |
| `swiping_left` | `discrete` | `previous_slide` | Discrete command via GestureActivationController → CommandDispatcher |
| `swiping_right` | `discrete` | `next_slide` | Discrete command via GestureActivationController → CommandDispatcher |
| `swiping_up` | `discrete` | `start_presentation` | Discrete command via GestureActivationController → CommandDispatcher |
| `zooming_in_full_hand` | `discrete` | `zoom_in_view` | Discrete command via GestureActivationController → CommandDispatcher |
| `zooming_out_full_hand` | `discrete` | `zoom_out_view` | Discrete command via GestureActivationController → CommandDispatcher |

## Feature Schema (`feature-schema-v5`)

| Block | Dims | Description |
|-------|------|-------------|
| `position` | 0–62 | 21 wrist-relative, scale-normalised landmark positions (x, y, z) |
| `fingertip_spread` | 63–67 | 5 inter-fingertip Euclidean distances |
| `wrist_trajectory` | 68–70 | Net wrist displacement from oldest frame in the window |
| `velocity` | 71–133 | 21 per-landmark wrist-relative velocity vectors (Δposition per unit time) |
| `joint_angles` | 134–143 | 10 MCP + PIP joint angles in radians |
| `wrist_vel_raw` | 144–146 | Camera-normalised wrist velocity (x, y, z) — key directional signal |


## How to Use

```python
import torch
from huggingface_hub import hf_hub_download
from maestro.infrastructure.model.checkpoint_loader import load_inference_artifact

# Download the artifact (cached after first call)
local_path = hf_hub_download(
    repo_id="ntsrigaud/maestro-lstm-hybrid",
    filename="two_stream_attn_v1_finetune_20260513T050407Z_inference.pt",
)

# Load the artifact (includes model, class labels, and feature schema)
artifact = load_inference_artifact(
    artifact_path=local_path,
    device=torch.device("cpu"),
)
artifact.model.eval()

# Build a 147-dim feature vector using LandmarkFeatureTransformer
# and fill a 32-frame SlidingWindowSequenceBuffer, then:
with torch.no_grad():
    # tensor shape: (batch=1, T=32, F=147)
    window_tensor = torch.tensor(window_np, dtype=torch.float32).unsqueeze(0)
    logits = artifact.model(window_tensor)
    pred_class = artifact.class_labels[logits.argmax(dim=1).item()]
```

## Training Dataset

- **Source**: Hybrid merge of Jester and IPN-Hand windows: Jester provides unknown/swiping/zoom/stop_sign classes; IPN-Hand provides point_one and point_two
- **Used classes**: 10 (9 active gestures + `unknown` background)
- **Dataset split**: 70% train / 15% val / 15% test (stratified by class)
- **Augmentation**: temporal scale ±20%, spatial jitter σ=0.005

## Training Strategy

Two-phase transfer learning pipeline:
- **Phase 1 (pretraining):** backbone pretrained on external checkpoint `two_stream_attn_v1_20260513T045733Z.pt` to learn generic gesture dynamics.
- **Phase 2 (fine-tuning):** head replaced and model adapted on Hybrid Jester+IPN 10-gesture vocabulary.
- **Stage A (frozen backbone):** 10 epoch(s) head-only warmup.
- **Stage B (full model):** up to 58 epoch(s) joint fine-tuning with scheduler/early stopping.
- **Stage B retention defences:** replay_max_samples_per_class=500, distillation_weight=0.5, replay_ce_weight=0.3, backbone_lr_multiplier=0.1, ewc_weight=100.0, gpm_components=20, forgetting_penalty_weight=0.5.

## Training Configuration

| Parameter | Value |
|-----------|-------|
| Architecture | EnhancedTwoStreamLSTM (BiLSTM h=96×2, MHA 8 heads, proj=96, mean+max pool, MLP gate) |
| Input size | 147 |
| Hidden size | 96/stream (BiLSTM output: 192) |
| Projection dim | 96 |
| Num layers | 4 |
| MHA heads | 8 (head dim: 24) |
| Dropout | 0.35 |
| Learning rate | 3e-05 |
| Weight decay | 0.0005 |
| Batch size | 128 |
| Max epochs | 60 |
| Early stopping patience | 12 |
| Label smoothing | 0.05 |
| Class weighting | disabled |
| Max samples per class | 5000 |
| LR scheduler | ReduceLROnPlateau (factor=0.5, patience=8) |

## Evaluation Results (Test Set)

| Metric | Value |
|--------|-------|
| Accuracy | 95.5% |
| Macro F1 | 94.8% |

### Per-Class Recall

| Class | Recall |
|-------|--------|
| `unknown` | 82.8% |
| `point_one` | 98.1% |
| `point_two` | 97.2% |
| `stop_sign` | 98.5% |
| `swiping_down` | 92.2% |
| `swiping_left` | 93.6% |
| `swiping_right` | 88.5% |
| `swiping_up` | 92.3% |
| `zooming_in_full_hand` | 97.0% |
| `zooming_out_full_hand` | 97.1% |

## Comparison with Previous Architecture

| Feature | TwoStreamGestureLSTM | EnhancedTwoStreamLSTM |
|---------|---------------------|-----------------------|
| LSTM direction | Unidirectional | **Bidirectional** |
| Attention | Bahdanau (scalar) | **MHA Q/K/V (8 heads)** |
| Feature projection | No | **Yes (→96)** |
| Temporal pooling | Mean only | **Mean + Max** |
| Cross-stream fusion | Concat only | **2-layer MLP gate** |
| Parameters | ~182 K | ~2,099,434 |

## Limitations and Risks

- Trained on IPN Hand subjects only. Performance may degrade with unusual hand sizes,
  skin tones, or lighting conditions not represented in training data.
- The `unknown` class represents background/transition frames. At runtime, predictions
  are filtered through per-class confidence thresholds defined in `production_hybrid.yaml`.
- Requires **mediapipe>=0.10.14** for landmark extraction at inference time.
- Not intended for safety-critical or accessibility-critical applications.
- Performance was measured on a held-out test split from the same dataset; real-world
  generalisation may differ.

## Environmental Impact

Training was performed on CPU/MPS. Estimated training time: ~10 minutes.
Estimated CO₂ equivalent: negligible (<0.001 kg CO₂eq).

---

*Generated by the Maestro training pipeline on 2026-05-13.*
