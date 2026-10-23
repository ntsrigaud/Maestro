---
language:
- en
license: other
license_name: maestro-weights-research-only
license_link: https://github.com/ntsrigaud/Maestro/blob/main/MODEL_LICENSE.md
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
- Jester
- IPN-Hand
- LD-ConGR
metrics:
- accuracy
- f1
model-index:
- name: two_stream_attn_v1_2layer_ld_cong_finetune_20260515T134706Z
  results:
  - task:
      type: gesture-recognition
    dataset:
      name: LDJI (LD-ConGR + Jester + IPN Hand)
      type: LDJI
    metrics:
    - type: accuracy
      value: 0.9606
    - type: f1
      value: 0.9587
---

# two_stream_attn_v1_2layer_ld_cong_finetune_20260515T134706Z

A real-time hand gesture classifier trained on
the LDJI hybrid gesture dataset (LD-ConGR swipes and fist, Jester zooms, IPN Hand pointing and background).

This model is part of the **Maestro** pipeline that enables touchless
control of presentation and meeting software through hand gestures captured from a
standard webcam using MediaPipe for landmark extraction.

## Model Description

- **Architecture**: EnhancedTwoStreamLSTM (BiLSTM h=96×2, MHA 8 heads, proj=96, mean+max pool, MLP gate)
- **Parameters**: 1,208,554
- **Input**: `(batch, 16, 147)`
    — 16-frame sliding window at 30 FPS ≈ 533 ms
- **Output**: Softmax logits over 10 gesture classes
- **Inference latency**: < 1 ms per call (CPU, single sample)
- **Feature schema**: `feature-schema-v5`

## Architecture

`EnhancedTwoStreamLSTM` splits the 147-dim feature vector into two parallel streams and
processes them through a BiLSTM + self-attention + MLP-gate pipeline:

```
Input (B, T=16, 147)
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
| `fist` | Closed fist (LD-ConGR class) |
| `swiping_down` | Vertical swipe downward (LD-ConGR class) |
| `swiping_left` | Horizontal swipe from right to left (LD-ConGR class) |
| `swiping_right` | Horizontal swipe from left to right (LD-ConGR class) |
| `swiping_up` | Vertical swipe upward (LD-ConGR class) |
| `zooming_in_full_hand` | Pinch-open / spread fingers away from each other (Jester class) |
| `zooming_out_full_hand` | Pinch-close / bring fingers together (Jester class) |

## Gesture Usage In Presentation System

| Class | Mode | Command | Runtime handling |
|-------|------|---------|------------------|
| `unknown` | `discrete` | `no_action` | No-op background class |
| `point_one` | `continuous` | `—` | Continuous tracker: LaserPointerTracker (bypasses discrete dispatcher) |
| `point_two` | `continuous` | `—` | Continuous tracker: AnnotationPenTracker (bypasses discrete dispatcher) |
| `fist` | `discrete` | `erase_annotations` | Discrete command via GestureActivationController → CommandDispatcher |
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
    filename="two_stream_attn_v1_2layer_ld_cong_finetune_20260515T134706Z_inference.pt",
)

# Load the artifact (includes model, class labels, and feature schema)
artifact = load_inference_artifact(
    artifact_path=local_path,
    device=torch.device("cpu"),
)
artifact.model.eval()

# Build a 147-dim feature vector using LandmarkFeatureTransformer
# and fill a 16-frame SlidingWindowSequenceBuffer, then:
with torch.no_grad():
    # tensor shape: (batch=1, T=16, F=147)
    window_tensor = torch.tensor(window_np, dtype=torch.float32).unsqueeze(0)
    logits = artifact.model(window_tensor)
    pred_class = artifact.class_labels[logits.argmax(dim=1).item()]
```

## Training Dataset

- **Source**: LDJI corpus of 28,882 windows pooled from LD-ConGR (fist and the four swipes), Jester (two zoom classes, center-cropped to 16 frames) and IPN Hand (`point_one`, `point_two`, `unknown`)
- **Used classes**: 10 (9 active gestures + `unknown` background)
- **Dataset split**: 70% train / 15% val / 15% test (stratified by class)
- **Augmentation**: temporal scale ±20%, spatial jitter σ=0.005

## Training Strategy

Two-phase transfer learning pipeline:
- **Phase 1 (pretraining):** backbone pretrained on Jester.
- **Phase 2 (fine-tuning):** classification head replaced and model fine-tuned on the 10-class LDJI vocabulary.
- **Stage A (frozen backbone):** 10 epochs of head-only warmup.
- **Stage B (full model):** up to 80 epochs of joint fine-tuning (early stopping selected epoch 46) with a 5-epoch backbone warmup.

## Training Configuration

| Parameter | Value |
|-----------|-------|
| Architecture | EnhancedTwoStreamLSTM (BiLSTM h=96×2, MHA 8 heads, proj=96, mean+max pool, MLP gate) |
| Input size | 147 |
| Hidden size | 96/stream (BiLSTM output: 192) |
| Projection dim | 96 |
| Num layers | 2 per stream |
| MHA heads | 8 (head dim: 24) |
| Learning rate | 3e-05 |
| Weight decay | 0.001 |
| Batch size | 128 |
| Max epochs | 80 |
| Early stopping patience | 20 |
| Label smoothing | 0.05 |
| Class weighting | disabled |
| Max samples per class | 3000 |
| LR scheduler | ReduceLROnPlateau (factor=0.5, patience=10) |

## Evaluation Results (Test Set)

| Metric | Value |
|--------|-------|
| Accuracy | 96.1% |
| Macro F1 | 95.9% |

### Per-Class Recall

| Class | Recall |
|-------|--------|
| `fist` | 97.3% |
| `swiping_left` | 98.3% |
| `swiping_right` | 97.1% |
| `swiping_up` | 98.2% |
| `swiping_down` | 98.0% |
| `zooming_in_full_hand` | 97.0% |
| `zooming_out_full_hand` | 95.1% |
| `point_one` | 97.4% |
| `point_two` | 95.1% |
| `unknown` | 85.7% |

## Comparison with Previous Architecture

| Feature | TwoStreamGestureLSTM | EnhancedTwoStreamLSTM |
|---------|---------------------|-----------------------|
| LSTM direction | Unidirectional | **Bidirectional** |
| Attention | Bahdanau (scalar) | **MHA Q/K/V (8 heads)** |
| Feature projection | No | **Yes (→96)** |
| Temporal pooling | Mean only | **Mean + Max** |
| Cross-stream fusion | Concat only | **2-layer MLP gate** |
| Parameters | ~182 K | ~1,208,554 |

## License

These weights are for academic and non-commercial research use only (see
[MODEL_LICENSE.md](https://github.com/ntsrigaud/Maestro/blob/main/MODEL_LICENSE.md)), because
they were trained on LD-ConGR (academic research only), Jester (research-use license) and
IPN Hand (CC BY 4.0). The Maestro source code is Apache-2.0.

## Limitations and Risks

- Trained on a hybrid of Jester, IPN Hand and LD-ConGR, Jester and IPN Hand recordings. Performance may degrade with unusual hand sizes,
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
