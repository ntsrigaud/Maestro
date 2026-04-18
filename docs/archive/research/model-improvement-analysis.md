# Gesture Recognition Model — Technical Review and Improvement Analysis

## 1. System Architecture Review

### 1.1 Temporal Modelling Capacity

The pipeline uses a 24-frame fixed window at 30 FPS (≈800 ms of motion). Key limitations:

- `output[:, -1, :]` retains only the final hidden state, compressing 24 frames of motion history into 128 dims. Gestures whose discriminative motion peaks mid-sequence (e.g., swipes) are not well served by this readout.
- No attention over the temporal axis — all time steps receive equal structural weight regardless of motion salience.
- Window step = 3 (10 Hz emission) means consecutive windows share 21/24 frames. Redundant inference amplifies consistent errors in the EMA stabilisation stage.

### 1.2 Spatial Feature Representation Gaps

| Missing Feature                           | Impacted Classes                                                  |
| ----------------------------------------- | ----------------------------------------------------------------- |
| Joint angles (MCP, PIP, DIP)              | `open_palm_hold` static discrimination                            |
| Relative fingertip spread/pinch distances | `zoom_in` / `zoom_out` separation                                 |
| Wrist trajectory (frame 0 → frame T)      | Swipe direction encoding                                          |
| Temporal acceleration                     | Gesture initiation vs. hold separation                            |
| Orientation normalisation (disabled)      | All classes — forces model to learn rotation invariance from data |

### 1.3 Dataset Structure

| Class            | Sequence count |
| ---------------- | -------------- |
| `unknown`        | 4,052          |
| `open_palm_hold` | 3,467          |
| `swipe_down`     | 2,503          |
| `swipe_up`       | 2,487          |
| `swipe_left`     | 2,720          |
| `swipe_right`    | 2,406          |
| `zoom_in`        | 2,669          |
| `zoom_out`       | 2,669          |

`unknown` is 68% larger than the smallest dynamic class. Class weighting partially compensates, but the minority classes lack diversity across recording sessions.

### 1.4 Overfitting Signature

Training accuracy at best validation epoch reaches **≥95%** while validation accuracy plateaus at **≈79%** — a ±16 pp gap. This is a data diversity problem, not solely a regularisation problem. Increasing model capacity without addressing diversity will worsen it.

---

## 2. Confusion Matrix Analysis

### 2.1 Dominant Confusion Pairs (Test Set)

| Source           | Predicted As  | Count | Error Rate |
| ---------------- | ------------- | ----- | ---------- |
| `zoom_in`        | `zoom_out`    | 53    | 14.3%      |
| `zoom_out`       | `unknown`     | 47    | 10.9%      |
| `swipe_left`     | `swipe_down`  | 41    | 10.8%      |
| `open_palm_hold` | `swipe_right` | 38    | 7.9%       |
| `zoom_in`        | `unknown`     | 37    | 10.0%      |
| `zoom_out`       | `zoom_in`     | 36    | 8.3%       |

### 2.2 Root Cause by Confusion Cluster

**Cluster A — Symmetric zoom confusion (`zoom_in` ↔ `zoom_out`, recall 70%)**
Velocity features encode per-landmark differences but do not cleanly encode whether the hand is opening or closing. The directional signature of spread vs. pinch requires relative inter-fingertip distance, which is absent from the current feature vector.

**Cluster B — Vertical swipe contamination (`swipe_left` → `swipe_down`, 10.8%)**
Wrist global displacement is not captured as a feature. Swipe direction is inferred from 21-landmark velocity fields; diagonal motions ambiguously match both left and down class signatures. An explicit wrist trajectory feature would resolve this.

**Cluster C — `open_palm_hold` decay (recall 72.1%)**
Transition frames at gesture onset/offset temporarily display a low-velocity open-palm configuration. With majority-vote labelling, windows spanning gesture boundaries receive the predominant label but contain mixed structure. The model learns these boundary frames as positive open-palm examples.

**Cluster D — `zoom_out` → `unknown` leak (10.9%)**
Low-confidence `zoom_out` predictions fail the 0.85 per-class threshold and are silently downgraded to `unknown`. Model capacity for this class is insufficient, not calibration noise (ECE = 0.033).

---

## 3. Evaluation of the Current LSTM Model

```
Architecture : 2-layer LSTM, hidden_size=128, dropout=0.3
Parameters   : 264,200
Latency (CPU): 0.485 ms per inference call (measured, single sample)
Test accuracy: 78.8%
Test macro F1: 78.0%
```

| Limitation                           | Detail                                                                         |
| ------------------------------------ | ------------------------------------------------------------------------------ |
| Final-step bottleneck                | `output[:, -1, :]` discards intermediate states that encode peak-motion frames |
| No temporal attention                | Cannot weight time steps by gesture-discriminative relevance                   |
| Directional recurrence only          | No bidirectionality; early frames cannot condition on later context            |
| Model overfits to recording sessions | 95% train vs. 79% val accuracy signals session-level memorisation              |
| Sufficient latency headroom          | At 0.5 ms, there is ≥10 ms available for a more capable model                  |

---

## 4. Survey of Modern Gesture Recognition Models

| Architecture                 | Params (est.) | CPU Latency (est.) | Notes                                        |
| ---------------------------- | ------------- | ------------------ | -------------------------------------------- |
| 2-layer LSTM (current)       | 264 K         | 0.5 ms             | Final-step bottleneck                        |
| LSTM + temporal attention    | 280 K         | 0.7 ms             | Drop-in replacement, best Phase 1 ROI        |
| Residual dilated TCN         | 180 K         | 0.4 ms             | Parallelisable, receptive field ≥ 24 frames  |
| TCN + self-attention         | 200 K         | 0.5 ms             | **Recommended Phase 2**                      |
| Transformer encoder (2L, 4H) | 380 K         | 1.5 ms             | Data-hungry; risky at current dataset size   |
| ST-GCN (lightweight)         | 600 K         | 3–5 ms             | Best accuracy ceiling; high integration cost |

**Key finding**: Attention-augmented LSTM or TCN + attention architectures consistently outperform plain LSTM by 6–10 pp on gesture benchmark datasets (Huang et al., 2021; Esteve et al., 2023) while remaining within a 2 ms CPU budget.

---

## 5. Proposed Model Architectures

### 5.1 Architecture A — LSTM with Temporal Attention (Phase 1)

Add a `nn.Linear(128, 1)` attention head over the full hidden state sequence:

```python
# Replace final-step extraction with attention-weighted pooling
output, _ = self.lstm(x)                               # (B, 24, 128)
attn_w = torch.softmax(self.attn(output).squeeze(-1), dim=1)  # (B, 24)
context = (attn_w.unsqueeze(-1) * output).sum(dim=1)   # (B, 128)
return self.head(self.dropout(context))
```

- **New parameters**: 129 (one Linear layer).
- **Expected gain**: +3–6 pp.
- **Integration**: Single method change in `lstm_model.py`; config unchanged.

### 5.2 Architecture B — Residual Dilated TCN (Phase 2)

Replace the LSTM with dilated causal convolutions:

```
Input (B, 24, F) → transpose → (B, F, 24)
TCN Block 1: Conv1d(F, 128, kernel=3, dilation=1)   receptive field: 3
TCN Block 2: Conv1d(128, 128, kernel=3, dilation=2)  receptive field: 7
TCN Block 3: Conv1d(128, 128, kernel=3, dilation=4)  receptive field: 15
TCN Block 4: Conv1d(128, 128, kernel=3, dilation=8)  receptive field: 31
Adaptive average pool → (B, 128) → Dropout → Linear(128, C)
```

Each block uses a residual connection and weight normalisation. Receptive field 31 > window length 24, so every output position attends to the full sequence. Parallelisable across the time axis; faster than LSTM on CPU.

### 5.3 Architecture C — TCN + Self-Attention (Phase 2, Recommended)

Extend Architecture B with a 2-head self-attention layer before pooling:

```
TCN output: (B, 128, 24) → transpose → (B, 24, 128)
MultiheadAttention(embed_dim=128, num_heads=2)
Mean pool → (B, 128) → Dropout → Linear(128, C)
```

- **Parameters**: ~200 K.
- **Latency**: ~0.6 ms on CPU.
- **Expected gain vs. current**: +6–10 pp.
- **Key benefit for zoom gestures**: Attention can learn to emphasise peak-spread or peak-pinch frames, directly addressing the symmetric zoom confusion.
- **Integration**: Same `forward(x) → logits` interface; zero changes to `runtime.py` or `checkpoint_loader.py`.

### 5.4 Architecture D — ST-GCN (Phase 3)

Model the hand as a 21-node graph with anatomical adjacency edges. Graph convolution captures spatial hierarchy (wrist → palm → finger segments). Best accuracy ceiling, highest integration complexity. Reserved for after Phase 1–2 benefits are validated.

---

## 6. Feature Engineering Improvements

Current: 126-d vector = 63 position dims + 63 velocity dims.

### 6.1 Wrist Trajectory Vector (High Priority)

Displacement of landmark 0 from the first frame to frame $t$, normalised by hand scale. Directly encodes swipe direction. Adds 3 dims per frame.

**Targets**: `swipe_left ↔ swipe_down` confusion (cluster B).

### 6.2 Fingertip Spread Distances (High Priority)

Five inter-fingertip Euclidean distances per frame:

| Pair                                     | Rationale        |
| ---------------------------------------- | ---------------- |
| Thumb tip (4) ↔ Index tip (8)            | Pinch state      |
| Thumb tip (4) ↔ Middle tip (12)          | Spread state     |
| Mean distance across tips (4,8,12,16,20) | Overall openness |

Adds 5 dims per frame. **Targets**: `zoom_in ↔ zoom_out` confusion (cluster A).

### 6.3 Joint Angle Features (Medium Priority)

Angle at MCP and PIP joints for all five fingers (10 angles per frame) using the three-point formula:

$$\theta_{ijk} = \arccos\frac{(\vec{j}-\vec{i}) \cdot (\vec{k}-\vec{i})}{||\vec{j}-\vec{i}||\ ||\vec{k}-\vec{i}||}$$

Invariant to global rotation and scale. Adds 10 dims per frame. **Targets**: `open_palm_hold` static discrimination (cluster C).

### 6.4 Enable Orientation Normalisation

The `_apply_orientation_normalization` method in `LandmarkFeatureTransformer` is complete but gated by `enable_orientation_normalization: false` in config. Enabling it removes the model's burden to learn rotation invariance from data. Expected: +2–3 pp at no engineering cost.

### 6.5 Cumulative Feature Budget

| Addition             | Dims per frame | Running total |
| -------------------- | -------------- | ------------- |
| Current baseline     | 126            | 126           |
| + Wrist trajectory   | +3             | 129           |
| + Fingertip spread   | +5             | 134           |
| + Joint angles       | +10            | 144           |
| + Wrist acceleration | +3             | 147           |

All additions are in `LandmarkFeatureTransformer`, which is shared between training and inference by design. Feature schema versioning and `validate_runtime_compatibility` enforce consistency between checkpoint and runtime automatically.

---

## 7. Dataset and Training Enhancements

### 7.1 Label Purity Filtering

Majority-vote windowing over gesture boundaries generates mislabelled sequences containing mixed gesture signatures. Apply a minimum label purity threshold: discard windows where the majority-class label fraction is below 90%. This eliminates the most confusable transition frames from training.

### 7.2 Augmentation Improvements

Current augmentation: temporal scale ±20%, spatial jitter σ=0.005.

| Addition                                                     | Implementation                  | Benefit                                                                 |
| ------------------------------------------------------------ | ------------------------------- | ----------------------------------------------------------------------- |
| **Hand mirror flip** (p=0.5)                                 | Negate all x-coords             | Doubles dataset effective size; handedness invariance                   |
| **Gaussian noise on velocity dims** (σ=0.002)                | Independent of position jitter  | Prevents over-reliance on sharp velocity peaks from specific recordings |
| **Random frame dropout** (1–3 frames replaced by neighbours) | In `GestureAugmenter.augment()` | Simulates MediaPipe landmark drop-out                                   |
| **Temporal reversal** (class-specific hard negatives)        | Reverse time axis within class  | Sharpens directional gesture boundaries                                 |

### 7.3 Label Smoothing

Replace hard one-hot cross-entropy targets with ε=0.1 label smoothing:

```python
criterion = nn.CrossEntropyLoss(weight=class_weights, label_smoothing=0.1)
```

Prevents logits from saturating at gesture boundaries, improving calibration and reducing overconfident misclassifications.

### 7.4 Curriculum Fine-Tuning for Zoom Classes

After initial training: collect all misclassified `zoom_in`, `zoom_out`, and `open_palm_hold` samples from the validation set and perform a single additional fine-tuning epoch with 3× oversampling of these examples. Targets the weakest classes directly without retraining from scratch.

---

## 8. Real-Time Performance Considerations

At 30 FPS, the full frame budget is 33 ms. Measured stage contributions:

| Stage                         | Typical cost |
| ----------------------------- | ------------ |
| OpenCV frame capture          | ~1–2 ms      |
| MediaPipe landmark extraction | ~5–15 ms     |
| Feature engineering           | ~0.1 ms      |
| LSTM inference (current)      | 0.485 ms     |
| Stabilisation + dispatch      | <0.1 ms      |

The LSTM contributes less than 2% of pipeline latency. Architecture C (TCN + attention, ~0.6 ms) and even a 147-d feature vector (adds <0.05 ms) operate comfortably within budget. All proposed architectures expose the same `forward(x: Tensor[B, T, F]) -> Tensor[B, C]` interface — no runtime or checkpoint loader changes required.

---

## 9. Implementation Roadmap

### Phase 1 — Low-Risk Improvements (est. 1–2 weeks)

Expected combined improvement: **+8–12 pp → val accuracy ≥ 85%, macro F1 ≥ 84%**

| Item                                             | Change location                       | Complexity |
| ------------------------------------------------ | ------------------------------------- | ---------- |
| 1a. Temporal attention readout                   | `lstm_model.py` (+3 lines)            | Low        |
| 1b. Enable orientation normalisation             | `configs/model/feature_pipeline.yaml` | Low        |
| 1c. Fingertip spread + wrist trajectory features | `feature_engineering.py`              | Medium     |
| 1d. Label smoothing (ε=0.1)                      | `train_lstm.py` (1 line)              | Low        |
| 1e. Hand mirror flip augmentation                | `augmentation.py`                     | Medium     |
| 1f. Label purity filtering                       | `dataset_builder.py`                  | Medium     |

### Phase 2 — Architecture Upgrade (est. 2–3 weeks)

Expected further improvement: **+4–6 pp → val accuracy ≥ 88–90%**

| Item                                                          | Change location          | Complexity |
| ------------------------------------------------------------- | ------------------------ | ---------- |
| 2a. `GestureTCNClassifier` with residual dilated convolutions | new `tcn_model.py`       | Medium     |
| 2b. Self-attention layer over TCN temporal features           | `tcn_model.py`           | Medium     |
| 2c. Joint angle features (10 angles)                          | `feature_engineering.py` | Medium     |
| 2d. Hard-mining fine-tune pass for zoom/palm classes          | `train_lstm.py`          | Medium     |

### Phase 3 — Research Models (est. 3–4 weeks)

Only warranted after Phase 1–2 benefits are validated.

| Item                                        | Notes                                                  |
| ------------------------------------------- | ------------------------------------------------------ |
| ST-GCN with hand graph adjacency            | Requires graph topology definition + `torch_geometric` |
| Additional recording sessions (≥3 subjects) | Most reliable long-term improvement path               |
| Compact Transformer encoder                 | Data-hungry; requires dataset expansion first          |

---

## 10. Final Architecture Recommendation

**Model: TCN with Temporal Self-Attention (Architecture C)**
**Features: 147-d (position + velocity + fingertip spread + wrist trajectory + joint angles)**

### Rationale

| Criterion              | Assessment                                                                                                              |
| ---------------------- | ----------------------------------------------------------------------------------------------------------------------- |
| Addresses root causes  | Fingertip spread targets zoom confusion; wrist trajectory targets swipe confusion; attention locates peak-motion frames |
| Latency                | ~0.6 ms on CPU — 2% of frame budget                                                                                     |
| Pipeline compatibility | Same `forward` interface; zero changes to `runtime.py`, `checkpoint_loader.py`, or any test                             |
| Feature parity         | All additions are in `LandmarkFeatureTransformer`, enforced by `validate_runtime_compatibility`                         |
| Engineering risk       | TCN + attention is well-understood; no external graph library required                                                  |

### Execution Order

Implement **Phase 1a, 1c, 1d, 1e** together in a single PR and validate against the current accuracy baseline in one training run. Feature additions (1c) and attention readout (1a) target the top confusion clusters with the least risk. Once Phase 1 baseline confirms ≥85% validation accuracy, proceed with Architecture C in Phase 2.

The primary bottleneck is **feature discriminability**, not model capacity. Fixing the features first yields the highest return per engineering hour.

---

**Document**: Model Improvement Analysis  
**Suite**: Architecture Documentation (`docs/architecture/`)  
**Last Updated**: April 2026
