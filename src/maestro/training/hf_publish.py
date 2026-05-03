"""Hugging Face Hub publishing for trained gesture recognition models.

Usage
-----
    from maestro.training.hf_publish import publish_to_huggingface

    url = publish_to_huggingface(
        result=train_result,
        config=pipeline_config,
        test_report=test_evaluation_report,
        inference_artifact_path="models/exported/lstm_v1_20260417T052358Z_inference.pt",
        repo_id="your-username/maestro-lstm",
        token="hf_...",        # or set HF_TOKEN env var
        private=True,
    )
    print(f"Published: {url}")
"""

from __future__ import annotations

import json
import os
import tempfile
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

try:
    from huggingface_hub import HfApi, login

    _HF_AVAILABLE = True
except ImportError:  # pragma: no cover
    _HF_AVAILABLE = False

from maestro.training.evaluate import EvaluationReport
from maestro.training.train_lstm import TrainingPipelineConfig, TrainingResult

# ---------------------------------------------------------------------------
# Feature schema description embedded in the model card
# ---------------------------------------------------------------------------
_FEATURE_SCHEMA_DESCRIPTION = """\
| Block | Dims | Description |
|-------|------|-------------|
| `position` | 0–62 | 21 wrist-relative, scale-normalised landmark positions (x, y, z) |
| `fingertip_spread` | 63–67 | 5 inter-fingertip Euclidean distances |
| `wrist_trajectory` | 68–70 | Net wrist displacement from oldest frame in the window |
| `velocity` | 71–133 | 21 per-landmark wrist-relative velocity vectors (Δposition per unit time) |
| `joint_angles` | 134–143 | 10 MCP + PIP joint angles in radians |
| `wrist_vel_raw` | 144–146 | Camera-normalised wrist velocity (x, y, z) — key directional signal |
"""

_CLASS_DESCRIPTIONS: dict[str, str] = {
    "open_palm_hold": "Static open palm facing camera",
    "swipe_right": "Horizontal swipe from left to right",
    "swipe_left": "Horizontal swipe from right to left",
    "swipe_up": "Vertical swipe upward",
    "swipe_down": "Vertical swipe downward",
    "zoom_in": "Pinch-open (spread fingers away from each other)",
    "zoom_out": "Pinch-close (bring fingers together)",
    "unknown": "Background / transition / no gesture",
}

_GRAPH_SCHEMA_DESCRIPTION = """\
| Block | Nodes | Coords | Description |
|-------|-------|--------|-------------|
| `right_hand` | 0–20 | x, y, z | MediaPipe right-hand landmarks (21 nodes) |
| `left_hand` | 21–41 | x, y, z | MediaPipe left-hand landmarks (21 nodes) |
| `pose` | 42–52 | x, y, z | 11 MediaPipe Pose landmarks (shoulders, elbows, wrists, ears, nose) |
| `face` | 53–66 | x, y, z | 14 selected MediaPipe Face Mesh contour points |

Coordinates are wrist-anchor normalised: translated by the mean of both wrist positions
and scaled by the inter-wrist distance.
"""

_PHONOLOGICAL_FEATURE_DESCRIPTION = """\
Per frame, Branch 1 computes the λ(k) feature vector (56-dim, parameter-free):

| Component | Symbol | Dims | Description |
|-----------|--------|------|-------------|
| Finger joint angles | H(k) — β_ij | 15×2=30 | Angle at each non-endpoint joint (PIP/DIP/IP) via
arccos of adjacent bone vectors; both hands |
| Inter-finger angles | H(k) — ω_i | 4×2=8 | Angle between adjacent MCP vectors from wrist
(thumb–index, index–middle, middle–ring, ring–pinky); both hands |
| Palm direction | n(k) | 3×2=6 | Unit normal of wrist→index-MCP × wrist→pinky-MCP; both hands |
| Hand position | l(k) | 3×2=6 | Middle-finger MCP coordinates; both hands |
| Motion | μ(k) | 3×2=6 | Velocity, acceleration, curvature of middle-MCP trajectory; both hands |

**Total**: 56-dim per frame. No learnable parameters in feature extraction.

Note: The original paper uses SVM top-3 hand-shape labels (s ∈ [0,60]) for H(k).
This implementation uses the raw geometric angles directly for end-to-end differentiability.
"""

_DUAL_STGCN_MODEL_CARD_TEMPLATE = """\
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
- graph-neural-network
- stgcn
- dual-branch
- phonological-features
datasets:
- IPN-Hand
metrics:
- accuracy
- f1
model-index:
- name: {model_version}
  results:
  - task:
      type: gesture-recognition
    dataset:
      name: IPN Hand
      type: IPN-Hand
    metrics:
    - type: accuracy
      value: {test_accuracy:.4f}
    - type: f1
      value: {test_f1:.4f}
---

# {model_version}

A real-time hand gesture classifier trained on the
[IPN Hand dataset](https://gibranbenitez.github.io/IPN_Hand/).

This model is part of the **Maestro** pipeline that enables touchless
control of presentation and meeting software through hand gestures captured from a
standard webcam using MediaPipe Holistic for landmark extraction.

## Model Description

- **Architecture**: {architecture}
- **Parameters**: {num_params:,}
- **Input**: `(batch, 24, 67, 3)` — 24-frame sliding window, 67-node graph, 3 coordinates
- **Output**: Softmax logits over {num_classes} gesture classes
- **Graph schema**: `{graph_schema_version}` — 67 nodes (hands + pose + face)
- **Reference**: Implemented from [IEEE 10965662](https://ieeexplore.ieee.org/document/10965662)

## Architecture

This model implements a dual-branch fusion architecture from:

> *"A Study on the STGCN-LSTM Sign Language Recognition Model Based on Phonological
> Features of Sign Language"*, IEEE 2025 (doi: 10.1109/ACCESS.2025.10965662).

Adapted for the IPN Hand skeleton dataset (67-node graph-schema-v1) and
hand gesture classification rather than sign language recognition.

### Branch 1 — Phonological SA-ConvLSTM

Extracts structured geometric phonological features **λ(k)** per frame, then
processes them through a Spatial-Attention ConvLSTM pipeline (Eq. 1–8 of the paper).

**λ(k) feature extraction** (parameter-free, deterministic):

{phono_feat_description}

**SA-ConvLSTM pipeline**:

```
λ(k): (B, T, 56)
  → Conv1d({phono_conv_channels}, k=3) + BN + ReLU  →  (B, T, {phono_conv_channels})
  → LSTM(hidden={lstm_hidden}, layers={lstm_layers}) →  (B, T, {lstm_hidden})
  → Self-attention Q/K/V (Eq. 6–8) + residual gate  →  (B, T, {lstm_hidden})
  → Bahdanau temporal attention                      →  (B, {lstm_hidden})
```

**Self-attention (Eq. 6–8)**:
$$Q = W_q H_t, \quad K = W_k H_t, \quad V = W_v H_t$$
$$\alpha_{{i,j}} = \text{{softmax}}_j\!\left(
    \frac{{Q_i K_j^T}}{{\sqrt{{D}}}}\right),\quad Z_i = \sum_j \alpha_{{i,j}} V_j$$
$$\hat{{H}}_t = w_f \cdot Z + H_t \quad (\text{{learnable residual gate}})$$

### Branch 2 — STGCN (3 blocks, fully-connected graph)

Three STGCNBlocks (Yan et al., AAAI 2018) on a K=3 partitioned fully-connected
adjacency over all 67 nodes. Partitions derived from anatomical BFS distances:
self (I₆₇), centripetal, centrifugal.

```
(B, T, 67, 3)
  → STGCNBlock(3   →  64, K=3, residual=True)
  → STGCNBlock(64  → 128, K=3, residual=True)
  → STGCNBlock(128 → 256, K=3, residual=True)
  → mean-pool over N and T                    →  (B, 256)
```

### Fusion

```
cat(Branch1={lstm_hidden}, Branch2=256) → (B, {fusion_dim})
→ Dropout({dropout}) → Linear({fusion_dim} → {num_classes}) → (B, {num_classes})
```

## Gesture Classes

| Class | Description |
|-------|-------------|
{class_table}

## Graph Schema (`{graph_schema_version}`)

{graph_schema_description}

## How to Use

```python
import torch
from huggingface_hub import hf_hub_download
from maestro.infrastructure.model.checkpoint_loader import load_inference_artifact

# Download the artifact (cached after first call)
local_path = hf_hub_download(
    repo_id="{repo_id}",
    filename="{model_version}_inference.pt",
)

# Load the artifact (includes model, class labels, and graph schema)
artifact = load_inference_artifact(
    artifact_path=local_path,
    device=torch.device("cpu"),
)
artifact.model.eval()

# Extract 67-node holistic keypoints via HolisticLandmarkExtractor,
# fill a 24-frame GraphSequenceBuffer, then:
with torch.no_grad():
    # tensor shape: (batch=1, T=24, N=67, C=3)
    window_tensor = torch.tensor(window_np, dtype=torch.float32).unsqueeze(0)
    logits = artifact.model(window_tensor)
    pred_class = artifact.class_labels[logits.argmax(dim=1).item()]
```

## Training Dataset

- **Source**: IPN Hand database — multi-subject, multi-session gesture recordings
- **Used classes**: {num_classes} ({num_classes_active} active gestures + `unknown` background)
- **Dataset split**: 70% train / 15% val / 15% test (group-aware split by recording session)
- **Graph**: 67 nodes — 42 hand + 11 pose + 14 face contour landmarks
- **Augmentation**: temporal scale ±20%, spatial jitter σ=0.003;
  label-aware offline horizontal mirror (swipe_left ↔ swipe_right, node-block swap)

## Training Configuration

| Parameter | Value |
|-----------|-------|
| Architecture | {architecture} |
| Phono feature dim | 56 (28 per hand, parameter-free) |
| Phono conv channels | {phono_conv_channels} |
| LSTM hidden | {lstm_hidden} |
| LSTM layers | {lstm_layers} |
| STGCN channels | {stgcn_channels} |
| Dropout | {dropout} |
| Learning rate | {learning_rate} |
| Batch size | {batch_size} |
| Max epochs | {epochs} |
| Early stopping patience | {patience} |
| Class weighting | ✓ inverse-frequency |
| LR scheduler | ReduceLROnPlateau (factor=0.5, patience=5) |

## Evaluation Results (Test Set)

| Metric | Value |
|--------|-------|
| Accuracy | {test_accuracy:.1%} |
| Macro F1 | {test_f1:.1%} |

### Per-Class Recall

| Class | Recall |
|-------|--------|
{per_class_table}

## Limitations and Risks

- Trained on IPN Hand subjects only. Performance may degrade with unusual hand sizes,
  skin tones, or lighting conditions not represented in training data.
- The `unknown` class represents background/transition frames. At runtime, predictions
  are filtered through per-class confidence thresholds defined in `production_ipn.yaml`.
- Requires **mediapipe>=0.10.14** for landmark extraction at inference time.
- Not intended for safety-critical or accessibility-critical applications.

## Citation

If you use this model, please cite the original paper this architecture is adapted from:

```bibtex
@article{{dual_stgcn_sign_2025,
  title   = {{A Study on the {{STGCN-LSTM}} Sign Language Recognition Model Based on
             Phonological Features of Sign Language}},
  journal = {{IEEE Access}},
  year    = {{2025}},
  doi     = {{10.1109/ACCESS.2025.10965662}},
  url     = {{https://ieeexplore.ieee.org/document/10965662}}
}}
```

## Environmental Impact

Training was performed on CPU. Estimated training time: ~20 minutes.
Estimated CO₂ equivalent: negligible (<0.01 kg CO₂eq).

---

*Generated by the Maestro training pipeline on {created_at}.*
"""

_ENHANCED_TWO_STREAM_MODEL_CARD_TEMPLATE = """\
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
- name: {model_version}
  results:
  - task:
      type: gesture-recognition
    dataset:
      name: IPN Hand
      type: IPN-Hand
    metrics:
    - type: accuracy
      value: {test_accuracy:.4f}
    - type: f1
      value: {test_f1:.4f}
---

# {model_version}

A real-time hand gesture classifier trained on the
[IPN Hand dataset](https://gibranbenitez.github.io/IPN_Hand/).

This model is part of the **Maestro** pipeline that enables touchless
control of presentation and meeting software through hand gestures captured from a
standard webcam using MediaPipe for landmark extraction.

## Model Description

- **Architecture**: {architecture}
- **Parameters**: {num_params:,}
- **Input**: `(batch, 24, {feature_dim})` — 24-frame sliding window at 30 FPS ≈ 800 ms
- **Output**: Softmax logits over {num_classes} gesture classes
- **Inference latency**: < 1 ms per call (CPU, single sample)
- **Feature schema**: `{feature_schema_version}`

## Architecture

`EnhancedTwoStreamLSTM` splits the 147-dim feature vector into two parallel streams and
processes them through a BiLSTM + self-attention + MLP-gate pipeline:

```
Input (B, T=24, 147)
    │
    ├─ Stream A — Pose/Shape (73 dims)
    │   Linear+LN+GELU → {proj_dim}
    │   2-layer BiLSTM (h={hidden_size}) → (B, T, {bilstm_out})
    │   LayerNorm → Self-MHA ({num_heads} heads) + residual + post-LN
    │   mean+max pool → pool_LN → ctx_a (B, {bilstm_out})
    │
    ├─ Stream B — Motion/Dynamics (74 dims)
    │   (identical structure) → ctx_b (B, {bilstm_out})
    │
    ├─ MLP cross-stream gate
    │   gate_a = Sigmoid(
    │     Linear({gate_hidden}→{bilstm_out})(
    │       Tanh(Linear({bilstm_out}→{gate_hidden})(ctx_b))))
    │   ctx_a  = LN(ctx_a × gate_a + ctx_a)
    │   gate_b = Sigmoid(
    │     Linear({gate_hidden}→{bilstm_out})(
    │       Tanh(Linear({bilstm_out}→{gate_hidden})(ctx_a))))
    │   ctx_b  = LN(ctx_b × gate_b + ctx_b)
    │
    └─ cat(ctx_a, ctx_b) → ({fused_dim},)
       LN → Linear({fused_dim}→{bilstm_out}) → GELU → Dropout → Linear({bilstm_out}→{num_classes})
```

**Design rationale:**
- BiLSTMs encode temporal order via their recurrent cell state — no positional encoding needed.
- Mean+Max pooling captures both sustained gesture shape (mean) and transient click events (max).
- The 2-layer MLP gate provides non-linear cross-modal recalibration at ~37 K params
  (vs ~263 K for full MHA cross-attention with a degenerate mean-pooled query).

## Gesture Classes

| Class | Description |
|-------|-------------|
{class_table}

## Feature Schema (`{feature_schema_version}`)

{feature_schema_description}

## How to Use

```python
import torch
from huggingface_hub import hf_hub_download
from maestro.infrastructure.model.checkpoint_loader import load_inference_artifact

# Download the artifact (cached after first call)
local_path = hf_hub_download(
    repo_id="{repo_id}",
    filename="{model_version}_inference.pt",
)

# Load the artifact (includes model, class labels, and feature schema)
artifact = load_inference_artifact(
    artifact_path=local_path,
    device=torch.device("cpu"),
)
artifact.model.eval()

# Build a {feature_dim}-dim feature vector using LandmarkFeatureTransformer
# and fill a 24-frame SlidingWindowSequenceBuffer, then:
with torch.no_grad():
    # tensor shape: (batch=1, T=24, F={feature_dim})
    window_tensor = torch.tensor(window_np, dtype=torch.float32).unsqueeze(0)
    logits = artifact.model(window_tensor)
    pred_class = artifact.class_labels[logits.argmax(dim=1).item()]
```

## Training Dataset

- **Source**: IPN Hand database — multi-subject, multi-session gesture recordings
- **Used classes**: {num_classes} ({num_classes_active} active gestures + `unknown` background)
- **Dataset split**: 70% train / 15% val / 15% test (stratified by class)
- **Augmentation**: temporal scale ±20%, spatial jitter σ=0.005;
  label-aware horizontal mirror (swipe_left ↔ swipe_right)

## Training Configuration

| Parameter | Value |
|-----------|-------|
| Architecture | {architecture} |
| Input size | {feature_dim} |
| Hidden size | {hidden_size}/stream (BiLSTM output: {bilstm_out}) |
| Projection dim | {proj_dim} |
| Num layers | {num_layers} |
| MHA heads | {num_heads} (head dim: {head_dim}) |
| Dropout | {dropout} |
| Learning rate | {learning_rate} |
| Weight decay | {weight_decay} |
| Batch size | {batch_size} |
| Max epochs | {epochs} |
| Early stopping patience | {patience} |
| Label smoothing | {label_smoothing} |
| Class weighting | ✓ inverse-frequency |
| LR scheduler | ReduceLROnPlateau (factor=0.5, patience=8) |

## Evaluation Results (Test Set)

| Metric | Value |
|--------|-------|
| Accuracy | {test_accuracy:.1%} |
| Macro F1 | {test_f1:.1%} |

### Per-Class Recall

| Class | Recall |
|-------|--------|
{per_class_table}

## Comparison with Previous Architecture

| Feature | TwoStreamGestureLSTM | EnhancedTwoStreamLSTM |
|---------|---------------------|-----------------------|
| LSTM direction | Unidirectional | **Bidirectional** |
| Attention | Bahdanau (scalar) | **MHA Q/K/V ({num_heads} heads)** |
| Feature projection | No | **Yes (→{proj_dim})** |
| Temporal pooling | Mean only | **Mean + Max** |
| Cross-stream fusion | Concat only | **2-layer MLP gate** |
| Parameters | ~182 K | ~{num_params:,} |

## Limitations and Risks

- Trained on IPN Hand subjects only. Performance may degrade with unusual hand sizes,
  skin tones, or lighting conditions not represented in training data.
- The `unknown` class represents background/transition frames. At runtime, predictions
  are filtered through per-class confidence thresholds defined in `production_ipn.yaml`.
- Requires **mediapipe>=0.10.14** for landmark extraction at inference time.
- Not intended for safety-critical or accessibility-critical applications.
- Performance was measured on a held-out test split from the same dataset; real-world
  generalisation may differ.

## Environmental Impact

Training was performed on CPU/MPS. Estimated training time: ~10 minutes.
Estimated CO₂ equivalent: negligible (<0.001 kg CO₂eq).

---

*Generated by the Maestro training pipeline on {created_at}.*
"""

_TWO_STREAM_MODEL_CARD_TEMPLATE = """\
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
datasets:
- IPN-Hand
metrics:
- accuracy
- f1
model-index:
- name: {model_version}
  results:
  - task:
      type: gesture-recognition
    dataset:
      name: IPN Hand
      type: IPN-Hand
    metrics:
    - type: accuracy
      value: {test_accuracy:.4f}
    - type: f1
      value: {test_f1:.4f}
---

# {model_version}

A real-time hand gesture classifier trained on the
[IPN Hand dataset](https://gibranbenitez.github.io/IPN_Hand/).

This model is part of the **Maestro** pipeline that enables touchless
control of presentation and meeting software through hand gestures captured from a
standard webcam using MediaPipe for landmark extraction.

## Model Description

- **Architecture**: {architecture}
- **Parameters**: {num_params:,}
- **Input**: `(batch, 24, {feature_dim})` — 24-frame sliding window at 30 FPS ≈ 800 ms
- **Output**: Softmax logits over {num_classes} gesture classes
- **Inference latency**: ~0.5 ms per call (CPU, single sample)
- **Feature schema**: `{feature_schema_version}`

## Gesture Classes

| Class | Description |
|-------|-------------|
{class_table}

## Feature Schema (`{feature_schema_version}`)

{feature_schema_description}

## How to Use

```python
import torch
from huggingface_hub import hf_hub_download
from maestro.infrastructure.model.checkpoint_loader import load_inference_artifact

# Download the artifact (cached after first call)
local_path = hf_hub_download(
    repo_id="{repo_id}",
    filename="{model_version}_inference.pt",
)

# Load the artifact (includes model, class labels, and feature schema)
artifact = load_inference_artifact(
    artifact_path=local_path,
    device=torch.device("cpu"),
)
artifact.model.eval()

# Build a {feature_dim}-dim feature vector using LandmarkFeatureTransformer
# and fill a 24-frame SlidingWindowSequenceBuffer, then:
with torch.no_grad():
    # tensor shape: (batch=1, T=24, F={feature_dim})
    window_tensor = torch.tensor(window_np, dtype=torch.float32).unsqueeze(0)
    logits = artifact.model(window_tensor)
    pred_class = artifact.class_labels[logits.argmax(dim=1).item()]
```

## Training Dataset

- **Source**: IPN Hand database — multi-subject, multi-session gesture recordings
- **Used classes**: {num_classes} ({num_classes_active} active gestures + `unknown` background)
- **Dataset split**: 70% train / 15% val / 15% test (group-aware split by recording session)
- **Sequences**: ~22,973 sliding windows × 24 frames × {feature_dim} features
- **Augmentation**: temporal scale ±20%, spatial jitter σ=0.005;
  label-aware offline horizontal mirror (swipe_left ↔ swipe_right)

## Training Configuration

| Parameter | Value |
|-----------|-------|
| Architecture | {architecture} |
| Input size | {feature_dim} |
| Hidden size | {hidden_size} |
| Num layers | {num_layers} |
| Dropout | {dropout} |
| Learning rate | {learning_rate} |
| Batch size | {batch_size} |
| Max epochs | {epochs} |
| Early stopping patience | {patience} |
| Label smoothing | 0.1 |
| Class weighting | ✓ inverse-frequency |
| LR scheduler | ReduceLROnPlateau (factor=0.5, patience=3) |

## Evaluation Results (Test Set)

| Metric | Value |
|--------|-------|
| Accuracy | {test_accuracy:.1%} |
| Macro F1 | {test_f1:.1%} |

### Per-Class Recall

| Class | Recall |
|-------|--------|
{per_class_table}

## Limitations and Risks

- Trained on IPN Hand subjects only. Performance may degrade with unusual hand sizes,
  skin tones, or lighting conditions not represented in training data.
- The `unknown` class represents background/transition frames. At runtime, predictions
  are filtered through per-class confidence thresholds defined in `production_ipn.yaml`.
- Requires **mediapipe==0.10.14** for landmark extraction at inference time.
- Not intended for safety-critical or accessibility-critical applications.
- Performance was measured on a held-out test split from the same dataset; real-world
  generalisation may differ.

## Environmental Impact

Training was performed on CPU. Estimated training time: ~5 minutes.
Estimated CO₂ equivalent: negligible (<0.001 kg CO₂eq).

---

*Generated by the Maestro training pipeline on {created_at}.*
"""


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _count_parameters(model: Any) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def _build_class_table(class_labels: list[str]) -> str:
    rows = []
    for cls in class_labels:
        desc = _CLASS_DESCRIPTIONS.get(cls, "—")
        rows.append(f"| `{cls}` | {desc} |")
    return "\n".join(rows)


def _build_per_class_table(per_class_recall: dict[str, float]) -> str:
    rows = [f"| `{cls}` | {recall:.1%} |" for cls, recall in per_class_recall.items()]
    return "\n".join(rows)


def _build_model_card(
    *,
    result: TrainingResult,
    config: TrainingPipelineConfig,
    test_report: EvaluationReport,
    repo_id: str = "ntsrigaud/maestro-lstm",
) -> str:
    is_dual_stgcn = config.model.model_name.startswith("dual_stgcn")
    num_params = _count_parameters(result.model)
    num_classes_active = len(result.class_labels) - 1  # subtract unknown
    created_at = datetime.now(UTC).strftime("%Y-%m-%d")

    if is_dual_stgcn:
        lstm_hidden = getattr(config.model, "lstm_hidden", 256)
        phono_conv_channels = getattr(config.model, "phono_conv_channels", 256)
        lstm_layers = getattr(config.model, "lstm_layers", 2)
        stgcn_channels = getattr(config.model, "stgcn_channels", (64, 128, 256))
        fusion_dim = lstm_hidden + stgcn_channels[-1]
        architecture = (
            f"DualBranchSTGCN_LSTM "
            f"(phono_feat=56, conv={phono_conv_channels}, "
            f"lstm={lstm_hidden}×{lstm_layers}, "
            f"stgcn={list(stgcn_channels)}, fusion={fusion_dim})"
        )
        return _DUAL_STGCN_MODEL_CARD_TEMPLATE.format(
            model_version=result.model_version,
            repo_id=repo_id,
            architecture=architecture,
            num_params=num_params,
            num_classes=config.model.num_classes,
            num_classes_active=num_classes_active,
            graph_schema_version=getattr(config.export, "graph_schema_version", "graph-schema-v1"),
            graph_schema_description=_GRAPH_SCHEMA_DESCRIPTION,
            phono_feat_description=_PHONOLOGICAL_FEATURE_DESCRIPTION,
            class_table=_build_class_table(result.class_labels),
            test_accuracy=test_report.accuracy,
            test_f1=test_report.macro_f1,
            phono_conv_channels=phono_conv_channels,
            lstm_hidden=lstm_hidden,
            lstm_layers=lstm_layers,
            stgcn_channels=list(stgcn_channels),
            fusion_dim=fusion_dim,
            dropout=config.model.dropout,
            learning_rate=config.training.learning_rate,
            batch_size=config.training.batch_size,
            epochs=config.training.epochs,
            patience=config.training.early_stopping_patience,
            per_class_table=_build_per_class_table(test_report.per_class_recall),
            created_at=created_at,
        )

    is_two_stream_attn = config.model.model_name.startswith("two_stream_attn")
    if is_two_stream_attn:
        hidden_size = getattr(config.model, "hidden_size", 96)
        num_layers = getattr(config.model, "num_layers", 2)
        num_heads = getattr(config.model, "num_heads", 8)
        proj_dim = getattr(config.model, "proj_dim", 96)
        dropout = config.model.dropout
        bilstm_out = hidden_size * 2
        gate_hidden = bilstm_out // 2
        fused_dim = bilstm_out * 2
        head_dim = bilstm_out // num_heads
        architecture = (
            f"EnhancedTwoStreamLSTM (BiLSTM h={hidden_size}×2, "
            f"MHA {num_heads} heads, proj={proj_dim}, mean+max pool, MLP gate)"
        )
        return _ENHANCED_TWO_STREAM_MODEL_CARD_TEMPLATE.format(
            model_version=result.model_version,
            repo_id=repo_id,
            architecture=architecture,
            num_params=num_params,
            feature_dim=config.model.input_size,
            num_classes=config.model.num_classes,
            num_classes_active=num_classes_active,
            feature_schema_version=config.export.feature_schema_version,
            feature_schema_description=_FEATURE_SCHEMA_DESCRIPTION,
            class_table=_build_class_table(result.class_labels),
            test_accuracy=test_report.accuracy,
            test_f1=test_report.macro_f1,
            hidden_size=hidden_size,
            num_layers=num_layers,
            num_heads=num_heads,
            proj_dim=proj_dim,
            bilstm_out=bilstm_out,
            gate_hidden=gate_hidden,
            fused_dim=fused_dim,
            head_dim=head_dim,
            dropout=dropout,
            weight_decay=config.training.weight_decay,
            learning_rate=config.training.learning_rate,
            batch_size=config.training.batch_size,
            epochs=config.training.epochs,
            patience=config.training.early_stopping_patience,
            label_smoothing=getattr(config.training, "label_smoothing", 0.05),
            per_class_table=_build_per_class_table(test_report.per_class_recall),
            created_at=created_at,
        )

    # Legacy two-stream LSTM path
    architecture = (
        f"Two-stream LSTM with temporal attention "
        f"(hidden_size={config.model.hidden_size}/stream, layers={config.model.num_layers})"
    )
    return _TWO_STREAM_MODEL_CARD_TEMPLATE.format(
        model_version=result.model_version,
        repo_id=repo_id,
        architecture=architecture,
        num_params=num_params,
        feature_dim=config.model.input_size,
        num_classes=config.model.num_classes,
        num_classes_active=num_classes_active,
        feature_schema_version=config.export.feature_schema_version,
        feature_schema_description=_FEATURE_SCHEMA_DESCRIPTION,
        class_table=_build_class_table(result.class_labels),
        test_accuracy=test_report.accuracy,
        test_f1=test_report.macro_f1,
        hidden_size=getattr(config.model, "hidden_size", "N/A"),
        num_layers=getattr(config.model, "num_layers", "N/A"),
        dropout=config.model.dropout,
        learning_rate=config.training.learning_rate,
        batch_size=config.training.batch_size,
        epochs=config.training.epochs,
        patience=config.training.early_stopping_patience,
        per_class_table=_build_per_class_table(test_report.per_class_recall),
        created_at=created_at,
    )


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def build_model_card(
    *,
    result: TrainingResult,
    config: TrainingPipelineConfig,
    test_report: EvaluationReport,
    repo_id: str = "ntsrigaud/maestro-lstm",
) -> str:
    """Return the rendered Markdown model card string without uploading."""
    return _build_model_card(result=result, config=config, test_report=test_report, repo_id=repo_id)


def publish_to_huggingface(
    *,
    result: TrainingResult,
    config: TrainingPipelineConfig,
    test_report: EvaluationReport,
    inference_artifact_path: str | Path,
    repo_id: str,
    token: str | None = None,
    private: bool = True,
    commit_message: str | None = None,
) -> str:
    """Publish a trained model to the Hugging Face Hub.

    Uploads three artefacts in two commits:
      1. ``README.md``     — rendered model card (YAML front-matter + Markdown body)
      2. ``config.json``   — structured metadata (model config, feature schema, metrics)
      3. ``<version>_inference.pt`` — the exported PyTorch inference artefact

    Parameters
    ----------
    result:
        Training result returned by ``train_lstm()``.
    config:
        Training pipeline configuration used to produce *result*.
    test_report:
        Evaluation report computed on the held-out test split.
    inference_artifact_path:
        Local path to the exported ``.pt`` inference artefact
        (produced by ``export_inference_artifact()``).
    repo_id:
        Hugging Face repository identifier in ``"username/repo-name"`` form.
    token:
        Hugging Face API token.  Falls back to the ``HF_TOKEN`` environment
        variable when *token* is ``None``.
    private:
        Whether to create the repository as private.  Ignored if the repository
        already exists.
    commit_message:
        Optional git commit message.  Defaults to the model version string.

    Returns
    -------
    str
        Public (or private) URL of the Hugging Face repository.

    Raises
    ------
    ImportError
        If ``huggingface_hub`` is not installed.
    ValueError
        If *repo_id* does not contain a ``/`` separator.
    """
    if not _HF_AVAILABLE:
        raise ImportError(
            "huggingface_hub is not installed.\nInstall it with: pip install huggingface_hub"
        )

    if "/" not in repo_id:
        raise ValueError(f"repo_id must be in the form 'username/repo-name', got: {repo_id!r}")

    resolved_token = token or os.environ.get("HF_TOKEN")
    if resolved_token:
        login(token=resolved_token, add_to_git_credential=False)

    api = HfApi()
    api.create_repo(repo_id=repo_id, repo_type="model", private=private, exist_ok=True)

    model_card_text = _build_model_card(
        result=result, config=config, test_report=test_report, repo_id=repo_id
    )

    is_dual_stgcn = config.model.model_name.startswith("dual_stgcn")
    is_stgcn = not is_dual_stgcn and config.model.model_name.startswith("stgcn")

    if is_dual_stgcn:
        model_config_snapshot: dict[str, Any] = {
            "model_name": config.model.model_name,
            "num_nodes": getattr(config.model, "num_nodes", 67),
            "num_coords": getattr(config.model, "num_coords", 3),
            "phono_proj_dim": getattr(config.model, "phono_proj_dim", 128),
            "phono_conv_channels": getattr(config.model, "phono_conv_channels", 256),
            "lstm_hidden": getattr(config.model, "lstm_hidden", 256),
            "lstm_layers": getattr(config.model, "lstm_layers", 2),
            "stgcn_channels": list(getattr(config.model, "stgcn_channels", [64, 128, 256])),
            "dropout": config.model.dropout,
            "num_classes": config.model.num_classes,
        }
        schema_snapshot: dict[str, Any] = {
            "graph_schema": {
                "graph_schema_version": getattr(
                    config.export, "graph_schema_version", "graph-schema-v1"
                ),
                "num_nodes": getattr(config.model, "num_nodes", 67),
                "num_coords": getattr(config.model, "num_coords", 3),
                "window_length": result.dataset_metadata.get("window_length", 24),
                "window_step": result.dataset_metadata.get("window_step", 3),
            }
        }
    elif is_stgcn:
        model_config_snapshot = {
            "model_name": config.model.model_name,
            "num_nodes": getattr(config.model, "num_nodes", 67),
            "num_coords": getattr(config.model, "num_coords", 3),
            "stgcn_channels": getattr(config.model, "stgcn_channels", [64, 128]),
            "lstm_hidden": getattr(config.model, "lstm_hidden", 128),
            "lstm_layers": getattr(config.model, "lstm_layers", 2),
            "dropout": config.model.dropout,
            "num_classes": config.model.num_classes,
        }
        schema_snapshot = {
            "graph_schema": {
                "graph_schema_version": getattr(
                    config.export, "graph_schema_version", "graph-schema-v1"
                ),
                "num_nodes": getattr(config.model, "num_nodes", 67),
                "num_coords": getattr(config.model, "num_coords", 3),
                "window_length": result.dataset_metadata.get("window_length", 24),
                "window_step": result.dataset_metadata.get("window_step", 3),
            }
        }
    else:
        model_config_snapshot = {
            "model_name": config.model.model_name,
            "input_size": config.model.input_size,
            "hidden_size": config.model.hidden_size,
            "num_layers": config.model.num_layers,
            "dropout": config.model.dropout,
            "num_classes": config.model.num_classes,
        }
        schema_snapshot = {
            "feature_schema": {
                "feature_schema_version": config.export.feature_schema_version,
                "feature_dim": config.model.input_size,
                "orientation_normalization": False,
                "window_length": result.dataset_metadata.get("window_length", 24),
                "window_step": result.dataset_metadata.get("window_step", 3),
            }
        }

    config_snapshot: dict[str, Any] = {
        "model_version": result.model_version,
        "model_config": model_config_snapshot,
        **schema_snapshot,
        "training_config": {
            "epochs": config.training.epochs,
            "batch_size": config.training.batch_size,
            "learning_rate": config.training.learning_rate,
            "weight_decay": config.training.weight_decay,
            "grad_clip_norm": config.training.grad_clip_norm,
            "seed": config.training.seed,
            "label_smoothing": 0.1,
        },
        "evaluation": {
            "test_accuracy": test_report.accuracy,
            "test_macro_f1": test_report.macro_f1,
            "test_loss": test_report.loss,
            "calibration_ece": test_report.calibration_ece,
            "per_class_recall": test_report.per_class_recall,
            "per_class_precision": test_report.per_class_precision,
        },
        "class_labels": result.class_labels,
        "created_at": datetime.now(UTC).isoformat(),
    }

    commit_msg = commit_message or f"Upload {result.model_version}"

    # Commit 1: README + config.json
    with tempfile.TemporaryDirectory() as tmp_dir:
        tmp = Path(tmp_dir)

        readme_path = tmp / "README.md"
        readme_path.write_text(model_card_text, encoding="utf-8")

        config_path = tmp / "config.json"
        config_path.write_text(json.dumps(config_snapshot, indent=2), encoding="utf-8")

        api.upload_folder(
            folder_path=str(tmp),
            repo_id=repo_id,
            repo_type="model",
            commit_message=commit_msg,
        )

    # Commit 2: inference artifact (may be large — streamed separately)
    api.upload_file(
        path_or_fileobj=str(Path(inference_artifact_path)),
        path_in_repo=f"{result.model_version}_inference.pt",
        repo_id=repo_id,
        repo_type="model",
        commit_message=f"Add inference artifact: {result.model_version}",
    )

    return f"https://huggingface.co/{repo_id}"
