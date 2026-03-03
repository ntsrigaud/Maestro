# Architecture A — Dual-Stream Supervised Model: Implementation Blueprint

**Date**: March 3, 2026  
**Status**: Execution Plan — Active  
**Depends on**: [SYSTEM_AUDIT_AND_REDESIGN.md](SYSTEM_AUDIT_AND_REDESIGN.md)  
**Research Question**: Does explicit geometric inductive bias combined with RGB fusion outperform RGB-only models for fine-grained static gesture recognition?

---

## 1. Research Objective Formalization

### 1.1 Model Definitions

**Baseline — $M_\text{RGB}$**: ViT-B/16 pretrained on ImageNet, fine-tuned end-to-end on the 8-class HaGRID subset.  
Input: 224×224×3 RGB crop.  
Output: $P \in \mathbb{R}^8$.

**Proposed — $M_\text{Dual}$**: Dual-stream model consisting of:

- GCN over 21 MediaPipe keypoints → $f_\text{geo} \in \mathbb{R}^{256}$
- ViT-B/16 fine-tuned on RGB → $f_\text{rgb} \in \mathbb{R}^{768}$
- Cross-attention fusion: $f_\text{fused} = \text{CrossAttn}(Q{=}f_\text{rgb},\, K{=}V{=}f_\text{geo}) \in \mathbb{R}^{512}$
- MLP classification head: $512 \to 256 \to 8$

**Ablation variants** (defined in Section 7):

- $M_\text{Geo}$: GCN-only, no RGB
- $M_\text{RGB+Cat}$: RGB + geometry via concatenation + MLP (no attention)
- $M_\text{NoAngle}$: $M_\text{Dual}$ without the engineered angle features
- $M_\text{NoPartTok}$: $M_\text{Dual}$ without part token supervision

---

### 1.2 Hypotheses

**Null hypothesis $H_0$**:

$$\text{Acc@1}(M_\text{Dual}) - \text{Acc@1}(M_\text{RGB}) < 2\%$$

Geometric inductive bias does not meaningfully improve over fine-tuned RGB-only classification.

**Alternative hypothesis $H_1$**:

$$\text{Acc@1}(M_\text{Dual}) - \text{Acc@1}(M_\text{RGB}) \geq 2\%$$

Explicit geometric inductive bias provides a measurable, statistically significant improvement over RGB-only fine-tuning, particularly on geometrically confusable classes (thumbs_up/thumbs_down).

---

### 1.3 Quantitative Success Criteria

| Criterion                          | Threshold                                                        | Measurement                                        |
| ---------------------------------- | ---------------------------------------------------------------- | -------------------------------------------------- |
| Top-1 accuracy improvement         | ≥ +2% absolute over $M_\text{RGB}$                               | Test set, mean over 3 random seeds                 |
| thumbs_up/down confusion reduction | ≥ 30% relative reduction in off-diagonal confusion count         | Confusion matrix, test set                         |
| ECE improvement                    | ≤ 0.10 (vs baseline ECE; see Section 8 for calibration pipeline) | 15-bin reliability diagram                         |
| Statistical significance           | $p < 0.05$, paired McNemar's test on per-sample predictions      | $M_\text{RGB}$ vs $M_\text{Dual}$, same test split |
| Inference latency                  | < 35ms per frame, CPU (Apple M1 or equivalent)                   | Median over 500 forward passes                     |

---

### 1.4 Evaluation Protocol

**Dataset splits**:

| Split      | Fraction | Usage                                                   |
| ---------- | -------- | ------------------------------------------------------- |
| Train      | 70%      | Model training                                          |
| Validation | 15%      | Hyperparameter selection, early stopping                |
| Test       | 15%      | Final evaluation only — accessed once per model variant |

The test set is fixed at preprocessing time and never used for any tuning decision.

**Reproducibility**:

- 3 random seeds: `{42, 123, 7}` for weight initialization and data shuffling
- Report mean ± std across seeds for all primary metrics
- 95% confidence intervals via bootstrap resampling (2000 samples) on test accuracy

**Required outputs per model**:

- Top-1 and Top-3 accuracy (mean ± std)
- Per-class F1 for all 8 classes
- Full 8×8 confusion matrix (absolute counts + normalized)
- ECE, MCE, Brier score
- Reliability diagram (15 bins)
- 2D t-SNE of penultimate embeddings (UMAP optional)
- McNemar's test statistic vs baseline

---

## 2. Data Pipeline Implementation Plan

### 2.1 HaGRID Subset Preparation

**Target classes and HaGRID name mapping**:

| System label   | HaGRID directory name | Notes                             |
| -------------- | --------------------- | --------------------------------- |
| `thumbs_up`    | `like`                | Direct mapping                    |
| `thumbs_down`  | `dislike`             | Direct mapping                    |
| `open_palm`    | `palm` or `stop`      | Use `palm`; verify directory name |
| `closed_fist`  | `fist`                | Direct mapping                    |
| `peace_sign`   | `peace`               | 2-finger V-shape                  |
| `ok_sign`      | `ok`                  | Circle thumb+index                |
| `pointing`     | `one`                 | Index extended only               |
| `call_gesture` | `call`                | Thumb + pinky                     |

**Per-class sample target**: 1,500 training + 320 validation + 320 test = **2,140 images per class**, **17,120 total**.

If a class has fewer images available, apply class-weighted sampling to compensate (do not oversample with synthetic data at this stage).

**Image format**: Resize to 256×256 first (preserving aspect ratio with padding), then center-crop to 224×224 at training time (random crop during training augmentation). Store originals; apply transforms in the dataloader.

---

### 2.2 Balanced Sampling Strategy

Use stratified splitting: sort images per class alphabetically, then apply a deterministic 70/15/15 split per class before any shuffle. This guarantees class balance in all three splits regardless of per-class dataset size variation.

```
for each class c:
    paths_c = sorted(glob(f"{hagrid_root}/{c}/*.jpg"))
    n = len(paths_c)
    train_c = paths_c[:int(0.70 * n)]
    val_c   = paths_c[int(0.70 * n):int(0.85 * n)]
    test_c  = paths_c[int(0.85 * n):]
```

Serialize the split index to `data/splits/split_seed42.json` before any training. All experiments use the same split file.

---

### 2.3 MediaPipe Keypoint Extraction

**Processing strategy**: Batch process offline; do not run MediaPipe inside the training loop.

**Algorithm per image**:

```
1. Load image as BGR (OpenCV), convert to RGB.
2. Run MediaPipe Hands (static_image_mode=True, max_num_hands=1,
   min_detection_confidence=0.5).
3. If no hand detected:
   a. Log the image path to data/keypoints/failed_detection.txt.
   b. Store a sentinel value: np.full((21, 3), np.nan, dtype=np.float32).
   c. Do NOT discard the image — the RGB stream can still use it;
      flag it so the keypoint stream falls back to zero-vector.
4. If hand detected:
   a. Extract NormalizedLandmark list (21 points: x, y, z).
   b. Convert normalized x, y to pixel coordinates:
      kpt[i] = [lm.x * W, lm.y * H, lm.z * W]  # z scaled by image width
   c. Store as float32 array shape (21, 3).
```

**Failure rate expectation**: < 5% on HaGRID (controlled indoor captures). If failure rate > 10% for any class, inspect images manually before proceeding.

**Caching format**:

Store per-split as a single compressed NumPy archive alongside a path index:

```
data/keypoints/
    train_keypoints.npz     # keys: "kpts" shape (N, 21, 3), "paths" shape (N,)
    val_keypoints.npz
    test_keypoints.npz
    failed_detection.txt
```

Using `.npz` (not `.pt`) keeps the cache framework-agnostic. Load into tensors in the dataset `__getitem__` method.

**Keypoint extraction time estimate**: ~1.5s per 100 images on CPU. For 17,120 images: ~4.3 minutes.

---

### 2.4 Keypoint Normalization

Apply normalization **inside the dataset `__getitem__`**, not at cache time. This allows experimenting with different normalization strategies without re-extracting.

**Step 1 — Handle NaN sentinel** (failed detections):

```python
if np.any(np.isnan(kpts)):
    kpts_normalized = np.zeros((21, 3), dtype=np.float32)
    kpt_valid_flag = 0  # passed to model as auxiliary input
else:
    kpt_valid_flag = 1
    # proceed to steps 2–3
```

**Step 2 — Translation invariance**: Subtract wrist keypoint $k_0$ (WRIST):

$$k'_i = k_i - k_0 \quad \forall i \in \{0, \ldots, 20\}$$

After this step, $k'_0 = (0,0,0)$ by construction.

**Step 3 — Scale invariance**: Divide by the distance from wrist to middle finger MCP ($k_9$):

$$\hat{k}_i = \frac{k'_i}{\|k'_9\|_2 + \epsilon}, \quad \epsilon = 10^{-6}$$

$\epsilon$ prevents division by zero on degenerate detections.

**After normalization, expected coordinate ranges**:

- $\hat{k}_i \in [-3.0, 3.0]$ (approximately; fingertips at ~1.5–2.0 wrist-to-MCP units)
- $\hat{k}_0 = (0,0,0)$ exactly

**Rotation normalization** (optional, implement only if ablation indicates unstabilized rotation hurts GCN):

- Align the palm normal vector to a canonical upward direction using Rodriguez rotation formula.
- Gate this behind a `--normalize_rotation` flag; do not apply by default.

**Output tensor shape**: `(21, 3)` float32.

---

### 2.5 Engineered Geometric Features

Append per-sample scalar features to form an augmented feature vector concatenated to the GCN output (not to the graph input — keep the graph input as pure keypoints).

**Feature 1 — Thumb vertical angle** (primary thumbs_up/down discriminator):

$$\theta_\text{thumb} = \arctan2\!\left((\hat{k}_4 - \hat{k}_0)_y,\, (\hat{k}_4 - \hat{k}_0)_x\right) \in [-\pi, \pi]$$

Normalize to $[-1, 1]$ by dividing by $\pi$.

**Feature 2 — Per-finger extension flags** (5 binary values):

For each finger $f \in \{\text{thumb, index, middle, ring, pinky}\}$:

$$\text{ext}_f = \mathbf{1}\!\left[\|k_\text{tip}^f - k_0\|_2 > \alpha_f \cdot \|k_\text{mcp}^f - k_0\|_2\right]$$

Set $\alpha_f = 1.1$ for all fingers (tip is further from wrist than MCP means finger is extended). Keypoint indices: thumb `(1,4)`, index `(5,8)`, middle `(9,12)`, ring `(13,16)`, pinky `(17,20)` — `(mcp, tip)`.

**Feature 3 — Thumb-index pinch distance** (ok_sign discriminator):

$$d_\text{pinch} = \|\hat{k}_4 - \hat{k}_8\|_2$$

Normalize by a running mean/std computed on the training set.

**Feature 4 — Thumb-pinky span** (call_gesture discriminator):

$$d_\text{span} = \|\hat{k}_4 - \hat{k}_{20}\|_2$$

**Concatenated engineered feature vector**: $f_\text{eng} \in \mathbb{R}^{8}$ (1 angle + 5 extension flags + 1 pinch + 1 span).

This is concatenated to the GCN's global pooled output before the classification head:

$$f_\text{geo} = [\text{GCN-pool}(\hat{K}) \,\|\, f_\text{eng}] \in \mathbb{R}^{256 + 8} = \mathbb{R}^{264}$$

A linear projection layer then maps $264 \to 256$ to maintain a consistent embedding dimension contract.

---

### 2.6 Pre-Training Validation Checklist

Before any model training begins, run `scripts/validate_keypoints.py` and verify:

- [ ] `kpts_normalized` coordinate range: all values within $[-4.0, 4.0]$ (flag anything outside)
- [ ] `kpt_valid_flag == 0` rate < 5% per class (log counts per class)
- [ ] $\|\hat{k}_0\|_2 = 0.0$ for all valid samples (translation invariance sanity check)
- [ ] $\|\hat{k}'_9\|_2 \approx 1.0 \pm 0.01$ for all valid samples (scale invariance sanity check)
- [ ] $\theta_\text{thumb}$ histogram: thumbs_up distribution should be centered near $+\pi/2$; thumbs_down near $-\pi/2$. **If these distributions overlap significantly (mean difference < 1.0 rad), halt and re-examine coordinate extraction.**
- [ ] t-SNE of $f_\text{eng}$ (8-dim engineered features only): thumbs_up and thumbs_down must show visible separation before any training. Save visualization to `results/keypoint_validation/tsne_engineered_features.png`.
- [ ] Extension flag mean per class (printed table): verify peace_sign shows index+middle = 1, others = 0; open_palm shows all 5 = 1; closed_fist shows all 5 = 0.

---

## 3. Geometric Stream Implementation

### 3.1 Graph Construction

**Node set**: $V = \{k_0, k_1, \ldots, k_{20}\}$, 21 nodes.

**Bone edge list** (directed; add both directions for undirected GCN):

```python
MEDIAPIPE_BONE_EDGES = [
    # Thumb
    (0, 1), (1, 2), (2, 3), (3, 4),
    # Index
    (0, 5), (5, 6), (6, 7), (7, 8),
    # Middle
    (0, 9), (9, 10), (10, 11), (11, 12),
    # Ring
    (0, 13), (13, 14), (14, 15), (15, 16),
    # Pinky
    (0, 17), (17, 18), (18, 19), (19, 20),
    # Palm cross-connections
    (5, 9), (9, 13), (13, 17),
]
```

Total directed edges (both directions): $(20 + 4) \times 2 = 48$ edges.

**Adjacency matrix**: $A \in \{0,1\}^{21 \times 21}$, symmetric, with self-loops added ($\tilde{A} = A + I_{21}$) per spectral GCN convention (Kipf & Welling, 2017).

**Degree normalization**: $\tilde{D}^{-1/2}\tilde{A}\tilde{D}^{-1/2}$ where $\tilde{D}_{ii} = \sum_j \tilde{A}_{ij}$.

Precompute the normalized adjacency once and register as a buffer (not a parameter) on the GCN module.

**Input feature per node**: $\hat{k}_i \in \mathbb{R}^3$ (normalized x, y, z). Node feature matrix: $X \in \mathbb{R}^{21 \times 3}$.

---

### 3.2 GCN Architecture

```
Input: X ∈ ℝ^{21×3}

GCN Layer 1:  Linear(3 → 64),  ReLU,  Dropout(0.2)   → H1 ∈ ℝ^{21×64}
GCN Layer 2:  Linear(64 → 128), ReLU, Dropout(0.2)   → H2 ∈ ℝ^{21×128}
GCN Layer 3:  Linear(128 → 256), ReLU                 → H3 ∈ ℝ^{21×256}

Global Mean Pool:  mean over node dim                 → h_pool ∈ ℝ^{256}
Concat engineered features:  [h_pool || f_eng]        → ℝ^{264}
Linear projection: 264 → 256, ReLU                   → f_geo ∈ ℝ^{256}
```

**Residual connection**: Between layer 2 → 3 only (dimensions match after projection):

$$H_3 = \text{GCN}_3(H_2) + W_\text{res} H_2$$

where $W_\text{res} \in \mathbb{R}^{128 \times 256}$ is a learned linear projection. This stabilizes training without adding significant parameters.

**Total GCN parameter count estimate**: ~120K parameters.

**Implementation note**: Use `torch_geometric.nn.GCNConv` for each layer. Register the precomputed edge index `(2, 48)` as a module buffer.

---

### 3.3 GCN Pretraining Protocol

Train the GCN with a classification head ($256 \to 8$) as a **standalone model $M_\text{Geo}$** before any joint training. This serves dual purposes: (1) establishes the geometry-only baseline, (2) provides a warm-started backbone for Phase 3 joint training.

**Configuration**:

```
Optimizer: AdamW, lr=1e-3, weight_decay=1e-4
LR schedule: CosineAnnealingLR, T_max=40, eta_min=1e-5
Epochs: 40
Batch size: 128
Loss: CrossEntropyLoss(label_smoothing=0.05)
Dropout: 0.2 in layers 1 and 2
Mixed precision: fp16 if CUDA/MPS available
```

**Handling failed detections ($\text{kpt\_valid\_flag} = 0$)**:

Pass the zero-vector through GCN normally. The model will learn that zero-vectors correspond to an unknown state. In the dataloader, randomly zero-out valid keypoints with probability 0.05 during training as a perturbation (keypoint dropout augmentation), improving robustness. Do not apply this during eval.

**Early stopping**: If validation accuracy does not improve by > 0.1% over 8 consecutive epochs, stop training and restore the best checkpoint.

**Expected performance of $M_\text{Geo}$**: 72–82% Top-1. The geometry stream alone lacks texture and lighting cues; this level is expected and acceptable. If $M_\text{Geo}$ exceeds 90% Top-1 by itself, reconsider whether RGB fusion is necessary (Decision Gate, Section 10).

---

### 3.4 GCN Diagnostic Requirements

After GCN pretraining, **before proceeding to joint training**, verify:

1. **t-SNE of $f_\text{geo}$** (256-dim GCN embeddings, test set): Save to `results/geometry_stream/tsne_gcn_embeddings.png`. Class clusters must be visually separable. thumbs_up and thumbs_down must occupy distinct regions.

2. **Class centroids and pairwise cosine distance**: Compute mean embedding per class. thumbs_up/thumbs_down centroid cosine distance must be ≥ 0.30. If < 0.20, the GCN is failing to use the geometric signal — re-examine normalization and angle feature correctness.

3. **Confusion matrix of $M_\text{Geo}$**: The thumbs*up/down confusion rate should already be low (< 15%) given the deterministic angle feature $\theta*\text{thumb}$.

**Failure criteria and fallback**:

| Observation                                | Action                                                                                                       |
| ------------------------------------------ | ------------------------------------------------------------------------------------------------------------ |
| $M_\text{Geo}$ Top-1 < 50%                 | Debug keypoint extraction and normalization pipeline before proceeding                                       |
| thumbs_up/down confusion > 30%             | Verify $\theta_\text{thumb}$ computation; check coordinate frame (HaGRID images have consistent orientation) |
| t-SNE shows complete mixing of all classes | GCN is not learning; check adjacency matrix correctness and optimizer settings                               |

---

## 4. RGB Stream Implementation

### 4.1 Model and Initialization

**Model**: `vit_base_patch16_224` from `timm`, pretrained weights `vit_base_patch16_224.orig_in21k_ft_in1k` (ImageNet-21K pretrained, fine-tuned on ImageNet-1K). This is strictly superior to using the CLIP ViT-B/16 weights for downstream supervised classification.

Do **not** use CLIP's ViT visual encoder as the RGB stream initialization. CLIP's ViT weights are optimized for contrastive alignment to text, not for discriminative image classification.

Replace the final classification head:

```python
model.head = nn.Linear(768, 8)  # 768 = ViT-B/16 output dim
nn.init.trunc_normal_(model.head.weight, std=0.02)
nn.init.zeros_(model.head.bias)
```

**RGB-only baseline $M_\text{RGB}$**: output of `model.head` is the final logit. This model is trained to completion independently and recorded as the official baseline before any fusion work.

---

### 4.2 Data Augmentation (RGB Stream)

Training augmentations (applied in dataloader, not cached):

```
RandomResizedCrop(224, scale=(0.7, 1.0))
RandomHorizontalFlip(p=0.5)          # OK for gestures; handedness labeled separately
ColorJitter(brightness=0.3, contrast=0.3, saturation=0.2, hue=0.05)
RandomGrayscale(p=0.1)
Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])  # ImageNet stats
```

Validation/test: `Resize(256) → CenterCrop(224) → Normalize(...)`.

**Note on horizontal flip**: HaGRID does not distinguish left/right hand as separate classes for the 8 chosen gestures. Horizontal flip is a valid augmentation. If extending to handedness-sensitive classes, revisit.

---

### 4.3 Fine-Tuning Schedule (Three Stages)

**Stage RGB-1: Classifier Head Only** (feature extractor frozen)

```
Freeze: all blocks, patch embedding, positional embedding, norm layers
Train: model.head only
Optimizer: AdamW, lr=1e-3, weight_decay=1e-4
Epochs: 5
Batch size: 64
Purpose: Warm-start head; avoid corrupting pretrained features on first pass
Expected val accuracy: 60–75%
```

**Stage RGB-2: Unfreeze Last 4 Transformer Blocks**

```
Unfreeze: blocks[-4:], final norm layer, head
Optimizer: AdamW with layer-wise LR decay:
    head: lr=1e-3
    last 4 blocks: lr=2e-4
    all other layers: frozen (lr=0)
LR schedule: CosineAnnealingLR, T_max=15, eta_min=1e-6
Epochs: 15
Batch size: 64
Regularization: label_smoothing=0.1, weight_decay=1e-4
Expected val accuracy: 85–92%
```

**Stage RGB-3: Full Fine-Tune**

```
Unfreeze: all layers
Optimizer: AdamW with layer-wise LR decay coefficient 0.65:
    head: lr=5e-5
    blocks[8:11]: lr ~ 3e-5
    blocks[4:7]:  lr ~ 2e-5
    blocks[0:3]:  lr ~ 1e-5
    patch embed:  lr ~ 5e-6
LR schedule: CosineAnnealingLR, T_max=10, eta_min=1e-7
Epochs: 10
Batch size: 32 (reduced for memory; gradient accumulation × 2 if needed)
Regularization: label_smoothing=0.1, weight_decay=1e-4, gradient clipping max_norm=1.0
Expected val accuracy: 90–94%
```

**Baseline $M_\text{RGB}$ final checkpoint**: best validation accuracy across all three stages. Record test accuracy **once** at the end.

---

### 4.4 Overfitting Detection Criteria

At each epoch, record `train_loss`, `val_loss`, `val_acc`. Trigger overfitting alert if:

- `(val_loss_current - val_loss_min) / val_loss_min > 0.08` AND
- `val_acc_current < val_acc_max - 1.0%` for 3 consecutive epochs

If triggered: restore best checkpoint and reduce LR by factor of 5 before proceeding to next stage, or terminate if already at minimum LR.

---

## 5. Fusion Module Implementation

### 5.1 Primary — Cross-Attention Fusion

**Module definition**:

```
Inputs:
    f_rgb ∈ ℝ^{B × 768}   (batch of RGB embeddings from ViT CLS token)
    f_geo ∈ ℝ^{B × 256}   (batch of GCN embeddings)

Step 1: Project to common dimension d_model = 256
    Q = Linear(768, 256)(f_rgb)   → ℝ^{B × 256}
    K = Linear(256, 256)(f_geo)   → ℝ^{B × 256}  (identity if d already 256)
    V = Linear(256, 256)(f_geo)   → ℝ^{B × 256}

Step 2: Multi-Head Attention (num_heads=4, head_dim=64)
    attn_out = MHA(Q, K, V, num_heads=4)  → ℝ^{B × 256}
    [scaled dot-product: Q·Kᵀ / √64, softmax, ×V]

Step 3: Residual + LayerNorm
    f_cross = LayerNorm(attn_out + Q)     → ℝ^{B × 256}

Step 4: Project to final embedding
    f_fused = Linear(256, 512)(f_cross)   → ℝ^{B × 512}

Step 5: Classification
    logits = MLP(f_fused): Linear(512, 256) → GeLU → Dropout(0.1) → Linear(256, 8)
```

**Rationale for $Q{=}f_\text{rgb}$, $K{=}V{=}f_\text{geo}$**: The RGB query attends over geometric key-value pairs. This means the fusion module learns "which geometric features are relevant, given this visual appearance". This is preferable to the reverse ($Q{=}f_\text{geo}$) because the RGB stream carries higher-dimensional, richer information and benefits more from targeted geometric context injection.

**Total fusion module parameters**: ~1.5M. Well within memory budget.

---

### 5.2 Fallback — Concatenation + MLP

Trigger fallback ($M_\text{RGB+Cat}$) if cross-attention fusion produces **no improvement** over $M_\text{RGB}$ after Phase 3 training (Section 6). Fallback decision criterion: if $M_\text{Dual}$ val accuracy after Phase 3 is within 0.5% of $M_\text{RGB}$, switch to concatenation to reduce complexity.

```
f_cat = [f_rgb_proj || f_geo]  where f_rgb_proj = Linear(768, 256)(f_rgb)
      → ℝ^{B × 512}

logits = MLP: Linear(512, 256) → GeLU → Dropout(0.1) → Linear(256, 8)
```

This is a simpler module with no inter-stream attention. It provides a strict upper bound check: if even concatenation does not help over RGB-only, the geometry stream has no joint benefit regardless of fusion architecture.

---

### 5.3 Expected Fusion Behavior

Fusion is expected to improve specifically:

| Gesture Pair            | Confusion Mechanism                                                             | Fusion Benefit                                                          |
| ----------------------- | ------------------------------------------------------------------------------- | ----------------------------------------------------------------------- |
| thumbs_up / thumbs_down | RGB appearance similar (both show prominent thumb); CLIP previously failed here | $\theta_\text{thumb}$ in $f_\text{geo}$ directly disambiguates          |
| peace_sign / pointing   | Both show extended fingers; RGB unsure about count                              | Extension flag vector in $f_\text{geo}$ encodes finger count explicitly |
| ok_sign / closed_fist   | Both partially closed; pinch distance subtle                                    | $d_\text{pinch}$ in $f_\text{geo}$ directly encodes ring formation      |

If fusion improves all three pairs, it validates that geometric features carry complementary discriminative information.

---

## 6. Training Strategy — Full Model

### Phase 1: GCN Pretraining (Standalone)

_See Section 3.3 for full specification._

| Parameter            | Value                              |
| -------------------- | ---------------------------------- |
| Epochs               | 40                                 |
| Optimizer            | AdamW, lr=1e-3                     |
| LR schedule          | CosineAnnealingLR, T_max=40        |
| Batch size           | 128                                |
| Loss                 | CrossEntropy, label_smoothing=0.05 |
| Dropout              | 0.2 (layers 1 and 2)               |
| Mixed precision      | fp16 if available                  |
| Early stopping       | Patience = 8 epochs                |
| **Checkpoint saved** | Best val accuracy                  |

**Phase 1 exit criterion**: GCN achieves ≥ 65% val Top-1. If not met after 40 epochs, debug pipeline (do not proceed to Phase 3).

---

### Phase 2: RGB-Only Baseline (Standalone)

_See Section 4.3 for full per-stage specification._

| Sub-phase | Epochs | LR                   | Unfrozen             |
| --------- | ------ | -------------------- | -------------------- |
| RGB-1     | 5      | 1e-3 (head only)     | head                 |
| RGB-2     | 15     | 2e-4 (last 4 blocks) | last 4 blocks + head |
| RGB-3     | 10     | 5e-5 (all)           | all                  |
| **Total** | **30** | —                    | —                    |

**Phase 2 exit criterion**: $M_\text{RGB}$ achieves ≥ 88% val Top-1 after Phase 2. If not met, verify data augmentation and split correctness. (Based on HaGRID benchmarks, ViT-B/16 routinely exceeds this.)

---

### Phase 3: Joint Training — Frozen ViT Backbone

Initialize:

- GCN from Phase 1 best checkpoint
- ViT-B/16 from Phase 2 best checkpoint
- Fusion module: random initialization (Xavier uniform)
- Classification head: random initialization

Freeze ViT and GCN backbones entirely. Train fusion module + head only.

```
Optimizer: AdamW, lr=5e-4
LR schedule: CosineAnnealingLR, T_max=10, eta_min=1e-5
Epochs: 10
Batch size: 32
Loss: CrossEntropy, label_smoothing=0.1
Gradient clipping: max_norm=1.0
Mixed precision: fp16 if available
```

**Purpose**: Allows the fusion module to learn meaningful geometry-RGB routing before the backbone weights are disturbed. Prevents the fusion module from receiving inconsistent gradient signals while the ViT is also adapting.

**Phase 3 exit criterion**: Fusion module val accuracy must exceed $M_\text{RGB}$ val accuracy by ≥ 0.5%. If the frozen-backbone fusion already fails to improve over RGB-only, this is a strong signal for fallback to concatenation.

---

### Phase 4: End-to-End Fine-Tuning

Unfreeze all components with layer-wise LR decay:

```
Fusion head + classification head: lr=1e-4
GCN all layers: lr=5e-5
ViT last 4 blocks: lr=2e-5
ViT blocks 4–7: lr=1e-5
ViT blocks 0–3: lr=5e-6
ViT patch + pos embed: lr=2e-6

Optimizer: AdamW, weight_decay=1e-4
LR schedule: CosineAnnealingLR with warm restarts (T_0=5, T_mult=2)
Epochs: 20 (max)
Batch size: 32
Gradient clipping: max_norm=0.5
Mixed precision: fp16 if available
Early stopping: patience = 5 epochs on val accuracy
```

**Final checkpoint**: Best val accuracy during Phase 4. Test set evaluated exactly once.

---

### Resource Expectations

| Phase                 | Estimated GPU time (A100 40GB) | Estimated time (Apple M1 CPU) |
| --------------------- | ------------------------------ | ----------------------------- |
| Phase 1 (GCN)         | ~8 min                         | ~45 min                       |
| Phase 2 (ViT)         | ~25 min                        | ~3 hrs                        |
| Phase 3 (Fusion head) | ~5 min                         | ~30 min                       |
| Phase 4 (E2E)         | ~20 min                        | ~2.5 hrs                      |
| **Total**             | **~1 hr**                      | **~7 hrs**                    |

Keypoint preprocessing: ~5 min (CPU, all splits).

---

## 7. Ablation Study Design

All ablations use the identical train/val/test split and are evaluated over 3 seeds. Results reported as mean ± std.

---

### Ablation 1: $M_\text{Geo}$ vs $M_\text{RGB}$

|              | $M_\text{Geo}$                         | $M_\text{RGB}$      |
| ------------ | -------------------------------------- | ------------------- |
| Input        | Keypoints (21×3) + engineered features | RGB (224×224×3)     |
| Architecture | GCN 3-layer + head                     | ViT-B/16 fine-tuned |

**Expected outcome**: $M_\text{RGB}$ outperforms $M_\text{Geo}$ by 10–20% on most classes (texture and color cues matter). However, $M_\text{Geo}$ outperforms $M_\text{RGB}$ specifically on thumbs_up/down confusion.

**Interpretation if $M_\text{Geo}$ > $M_\text{RGB}$**: Keypoint geometry alone is sufficient; RGB stream may be redundant. Strongly reconsider fusion necessity.

**Interpretation if $M_\text{Geo}$ < 60%** Top-1: Keypoint extraction or normalization has a systematic bug. Halt all downstream experiments.

---

### Ablation 2: $M_\text{RGB}$ vs $M_\text{Dual}$

This is the primary ablation answering $H_0$ vs $H_1$.

**Expected outcome**: $M_\text{Dual}$ outperforms $M_\text{RGB}$ by ≥ 2% Top-1. The improvement should be localized to the geometrically confusable classes (thumbs_up/down, peace/pointing, ok/fist).

**If $M_\text{Dual}$ ≤ $M_\text{RGB}$** (no improvement): Null hypothesis $H_0$ not rejected. The geometric stream is not providing complementary information. Diagnose: are the attention weights in the fusion module near-uniform? If so, the geometry is being ignored — increase $\mathcal{L}_\text{geo}$ auxiliary loss weight.

**Statistical significance**: Run McNemar's test on per-sample test-set predictions (correct/incorrect binary). Report $\chi^2$ statistic and $p$-value.

---

### Ablation 3: $M_\text{Dual}$ vs $M_\text{NoAngle}$

Remove the thumb angle feature $\theta_\text{thumb}$ and finger extension flags from $f_\text{eng}$. Retrain geometry stream from scratch. All other settings identical.

**Expected outcome**: 2–5% drop in thumbs_up/down class F1 scores; < 1% change in other classes.

**Interpretation if no change**: Engineered features are not contributing beyond what the GCN learns from raw keypoints. Simplify $f_\text{eng}$ for production.

**Interpretation if drop > 8%**: Engineered features dominate the geometry signal. Verify that the GCN is actually learning; check activations.

---

### Ablation 4: Cross-Attention vs Concatenation ($M_\text{Dual}$ vs $M_\text{RGB+Cat}$)

Replace cross-attention fusion with concatenation + MLP. Same training schedule, same number of epochs.

**Expected outcome**: Cross-attention outperforms concatenation by 0.5–1.5%. If the gap is < 0.5%, use concatenation in production (lower complexity, same performance).

**Interpretation if concatenation outperforms cross-attention**: Cross-attention is overfitting or not training stably. Increase dropout in attention, reduce learning rate in Phase 3.

---

### Ablation 5: $M_\text{Dual}$ vs $M_\text{NoPartTok}$

This ablation is **conditional**: only run if the part token supervision component is implemented. If not implemented by Month 3, defer to Month 5.

Remove part token supervision loss ($\mathcal{L}_\text{part}$) from the training objective; all other settings identical.

**Expected outcome**: 1–2% drop in fine-grained classes (peace_sign, pointing, ok_sign). No expected effect on thumbs_up/down (handled by GCN).

---

### Summary Ablation Table (target format for paper Table 2)

| Model                                  | Top-1 (%) | thumbs↑↓ Confusion | ECE |
| -------------------------------------- | --------- | ------------------ | --- |
| $M_\text{Geo}$ (keypoints only)        | —         | —                  | —   |
| $M_\text{RGB}$ (ViT-B/16 baseline)     | —         | —                  | —   |
| $M_\text{RGB+Cat}$ (concat fusion)     | —         | —                  | —   |
| $M_\text{NoAngle}$ (no angle features) | —         | —                  | —   |
| $M_\text{Dual}$ (full model, proposed) | —         | —                  | —   |

Fill in values after experiments. The paper's central claim is the $M_\text{RGB}$ → $M_\text{Dual}$ row.

---

## 8. Evaluation and Analysis Protocol

### 8.1 Primary Metrics

Computed on held-out test set, reported per seed and as mean ± std across 3 seeds.

```python
# Collect for each model:
all_probs  : np.ndarray  # shape (N, 8), softmax probabilities
all_preds  : np.ndarray  # shape (N,), argmax predictions
all_labels : np.ndarray  # shape (N,), ground truth integer labels
```

**Top-1 Accuracy**:

$$\text{Acc@1} = \frac{1}{N}\sum_{i=1}^N \mathbf{1}[\hat{y}_i = y_i]$$

**Top-3 Accuracy**:

$$\text{Acc@3} = \frac{1}{N}\sum_{i=1}^N \mathbf{1}[y_i \in \text{top-3}(\hat{p}_i)]$$

**Per-class F1**: Compute macro and per-class F1 via `sklearn.metrics.classification_report`. Report all 8 classes individually.

---

### 8.2 Confusion Matrix

Produce two forms:

1. **Absolute counts**: `confusion_matrix(all_labels, all_preds)` — shape (8, 8)
2. **Row-normalized** (recall-based): divide each row by its sum

Save as CSV and PNG heatmap. Annotate the thumbs_up/thumbs_down cell explicitly in the paper figure.

**thumbs_up/down confusion rate**:

$$r_\text{thumb} = \frac{C[\text{thumbs\_up}, \text{thumbs\_down}] + C[\text{thumbs\_down}, \text{thumbs\_up}]}{N_\text{thumbs\_up} + N_\text{thumbs\_down}}$$

Target: $r_\text{thumb}(M_\text{Dual}) \leq 0.70 \times r_\text{thumb}(M_\text{RGB})$ (30% relative reduction).

---

### 8.3 Calibration Metrics

Use 15 equal-width confidence bins over $[0, 1]$.

**Expected Calibration Error (ECE)**:

$$\text{ECE} = \sum_{m=1}^{15} \frac{|B_m|}{N} \left|\text{acc}(B_m) - \text{conf}(B_m)\right|$$

**Maximum Calibration Error (MCE)**:

$$\text{MCE} = \max_m \left|\text{acc}(B_m) - \text{conf}(B_m)\right|$$

**Brier Score**:

$$\text{BS} = \frac{1}{N}\sum_{i=1}^N \sum_{c=1}^8 (p_{ic} - y_{ic})^2$$

Produce reliability diagram (confidence on x-axis, accuracy on y-axis, bar chart per bin, diagonal = perfect calibration). Save to `results/calibration/reliability_{model_name}.png`.

**ECE success threshold**: The supervised fine-tuned $M_\text{RGB}$ baseline should achieve ECE ≤ 0.08 (vs. the observed CLIP ECE of 0.857). If $M_\text{RGB}$ ECE > 0.15, apply post-hoc temperature scaling before comparison with $M_\text{Dual}$.

---

### 8.4 Embedding Separability Analysis

**t-SNE visualization**:

Extract penultimate embeddings (before the final linear head) for all test samples. Run t-SNE with `perplexity=40, n_iter=1000`. Color by class label. Save to `results/embeddings/tsne_{model_name}.png`.

**Cosine margin statistics**:

For each test sample $i$, compute:

$$\text{margin}_i = p_{i,\hat{y}_i} - \max_{j \neq \hat{y}_i} p_{i,j}$$

where $p$ are softmax probabilities. Report:

- Mean margin, correctly classified samples
- Mean margin, misclassified samples
- Percentage of samples with margin > 0.3 (high-confidence correct)

**Thumbs_up/down embedding separability** — inter-class centroid cosine distance:

$$d_\text{TU/TD} = 1 - \frac{\mu_\text{thumbs\_up} \cdot \mu_\text{thumbs\_down}}{||\mu_\text{thumbs\_up}||\, ||\mu_\text{thumbs\_down}||}$$

Report for $M_\text{RGB}$, $M_\text{Geo}$, and $M_\text{Dual}$. Target: $d_\text{TU/TD}(M_\text{Dual}) > d_\text{TU/TD}(M_\text{RGB})$.

---

### 8.5 Statistical Significance — McNemar's Test

For two models $A$ and $B$ evaluated on the same test set:

Let:

- $n_{10}$ = samples $A$ correct, $B$ wrong
- $n_{01}$ = samples $A$ wrong, $B$ correct

$$\chi^2 = \frac{(|n_{10} - n_{01}| - 1)^2}{n_{10} + n_{01}}$$

$p$-value from $\chi^2$ distribution with 1 degree of freedom. Report for $M_\text{RGB}$ vs $M_\text{Dual}$.

---

## 9. Computational Budget and Feasibility

### 9.1 Memory Estimates

| Component                                 | Memory                               |
| ----------------------------------------- | ------------------------------------ |
| ViT-B/16 weights (fp32)                   | ~330 MB                              |
| ViT-B/16 activations (batch 32, fp16)     | ~850 MB                              |
| GCN weights                               | ~2 MB                                |
| GCN activations (batch 128)               | ~10 MB                               |
| Fusion module                             | ~6 MB                                |
| Dataset (17,120 images @ 224×224×3 uint8) | ~3.1 GB uncompressed / ~400 MB as H5 |
| Keypoint cache (17,120 × 21 × 3 float32)  | ~4.3 MB                              |

**Total peak GPU memory (Phase 4, batch=32)**: ~2.5 GB. Fits on a 4GB GPU with fp16. For 8GB+ GPU, batch size can be increased to 64.

### 9.2 Training Time Estimates

_See Section 6 for per-phase breakdown._

| Platform                 | Total training time |
| ------------------------ | ------------------- |
| NVIDIA A100 40GB         | ~1 hour             |
| NVIDIA RTX 3090          | ~2.5 hours          |
| Apple M1 Pro (CPU + MPS) | ~5–7 hours          |

Preprocessing (keypoint extraction): ~5 minutes CPU on full 17,120-image dataset.

### 9.3 Fallback Strategies for Limited Compute

| Constraint             | Fallback                                                                       |
| ---------------------- | ------------------------------------------------------------------------------ |
| GPU memory < 4GB       | Reduce batch to 16, enable gradient accumulation ×4                            |
| No GPU available       | Reduce ViT to ViT-S/16 (50% of parameters); accept ~3% lower baseline accuracy |
| Training time > 10 hrs | Skip Phase 4 (end-to-end fine-tune); report Phase 3 fusion result only         |
| HaGRID download fails  | Use 500 images/class (4,000 total); reduce val/test to 100/100 per class       |

---

## 10. Decision Gate Criteria

The following gates determine whether to proceed, pivot, or halt.

### Gate 1 — After Phase 1 (GCN pretraining)

| Result                              | Decision                                                              |
| ----------------------------------- | --------------------------------------------------------------------- |
| $M_\text{Geo}$ val Top-1 ≥ 65%      | ✅ Proceed to Phase 2                                                 |
| $M_\text{Geo}$ val Top-1 50–65%     | ⚠️ Debug; check normalization. Proceed with caution                   |
| $M_\text{Geo}$ val Top-1 < 50%      | 🛑 Halt. Systematic data pipeline error. Do not proceed               |
| thumbs_up/down confusion rate > 30% | ⚠️ Verify $\theta_\text{thumb}$ feature correctness before proceeding |

### Gate 2 — After Phase 2 (RGB baseline)

| Result                          | Decision                                                         |
| ------------------------------- | ---------------------------------------------------------------- |
| $M_\text{RGB}$ val Top-1 ≥ 88%  | ✅ Proceed to Phase 3                                            |
| $M_\text{RGB}$ val Top-1 80–88% | ⚠️ Extend Phase 2 by 5 epochs; check data split and augmentation |
| $M_\text{RGB}$ val Top-1 < 80%  | 🛑 Halt. ViT fine-tuning not converging. Verify ImageNet init.   |

### Gate 3 — After Phase 3 (Frozen-backbone fusion)

| Result                                                   | Decision                                                                 |
| -------------------------------------------------------- | ------------------------------------------------------------------------ |
| $M_\text{Dual}$ val Top-1 > $M_\text{RGB}$ + 0.5%        | ✅ Proceed to Phase 4 — fusion is learning                               |
| $M_\text{Dual}$ val Top-1 within ±0.5% of $M_\text{RGB}$ | ⚠️ Switch to concatenation fallback ($M_\text{RGB+Cat}$); re-run Phase 3 |
| $M_\text{Dual}$ val Top-1 < $M_\text{RGB}$ - 1%          | 🛑 Cross-attention fusion is interfering. Switch to $M_\text{RGB+Cat}$   |

### Gate 4 — After Phase 4 (End-to-end fine-tune, test set evaluation)

| Result                                                         | Decision                                                                                                                                       |
| -------------------------------------------------------------- | ---------------------------------------------------------------------------------------------------------------------------------------------- |
| $M_\text{Dual}$ test Top-1 ≥ $M_\text{RGB}$ + 2%, $p$ < 0.05   | ✅ **$H_1$ accepted**. Proceed to Architecture B (semantic alignment)                                                                          |
| $M_\text{Dual}$ test Top-1 = $M_\text{RGB}$ + 1–2%, $p$ < 0.05 | ⚠️ Marginal improvement. Report but do not claim strong $H_1$ support. Investigate per-class breakdown                                         |
| $M_\text{Dual}$ test Top-1 < $M_\text{RGB}$ + 1%, $p$ ≥ 0.05   | 🛑 **$H_0$ not rejected.** Geometric fusion does not significantly improve RGB-only. Publish negative result. Do not proceed to Architecture B |

### Gate 5 — Special Case: Geometry-Only Dominance

| Result                                                                         | Decision                                                                                                                                                                                                  |
| ------------------------------------------------------------------------------ | --------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| $M_\text{Geo}$ test Top-1 > 90%                                                | 🔄 Reconsider: geometry alone may suffice. RGB stream adds complexity without proportional benefit. Investigate whether $M_\text{Dual}$ over $M_\text{Geo}$ improvement justifies the added ViT component |
| $M_\text{Geo}$ test Top-1 > $M_\text{RGB}$ on thumbs_up/down but lower overall | ✅ Expected. Validates geometric stream's targeted benefit; fusion justified                                                                                                                              |

---

## 11. Publication Readiness Criteria

The following outputs are required before declaring Architecture A complete for manuscript preparation:

- [ ] 3-seed results for $M_\text{RGB}$, $M_\text{Geo}$, $M_\text{RGB+Cat}$, $M_\text{Dual}$
- [ ] Full ablation table (Section 7)
- [ ] Confusion matrices for $M_\text{RGB}$ and $M_\text{Dual}$ (side-by-side figure)
- [ ] t-SNE embeddings for $M_\text{RGB}$ and $M_\text{Dual}$ (side-by-side figure)
- [ ] Reliability diagrams (calibration) for both models
- [ ] McNemar's test $p$-value for primary comparison
- [ ] thumbs*up/down confusion reduction quantified ($r*\text{thumb}$ ratio)
- [ ] Inference latency measurement (median ± std, 500 forward passes, CPU)
- [ ] All checkpoints saved to `checkpoints/arch_a/` with seed-tagged filenames

---

## References

1. Kipf, T.N., Welling, M. "Semi-Supervised Classification with Graph Convolutional Networks." _ICLR_, 2017.
2. Dosovitskiy, A. et al. "An Image is Worth 16x16 Words: Transformers for Image Recognition at Scale." _ICLR_, 2021.
3. Wah, C. et al. "The Caltech-UCSD Birds-200-2011 Dataset." Technical Report, 2011. _(Part-token supervision precedent)_
4. Kapitanov, A. et al. "HaGRID." arXiv:2412.01508, 2024.
5. Khosla, P. et al. "Supervised Contrastive Learning." _NeurIPS_, 2020. _(SupCon loss)_
6. Zhang, F. et al. "MediaPipe Hands." arXiv:2006.10214, 2020.
7. Touvron, H. et al. "Training data-efficient image transformers & distillation through attention." _ICML_, 2021. _(Layer-wise LR decay for ViT fine-tuning)_
