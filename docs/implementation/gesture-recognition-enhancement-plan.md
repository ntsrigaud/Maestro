# Gesture Recognition Enhancement — Implementation Plan

## Baseline (April 2026)

| Metric        | Value                                 |
| ------------- | ------------------------------------- |
| Test accuracy | 78.8%                                 |
| Test macro F1 | 78.0%                                 |
| Feature dims  | 126                                   |
| Model readout | final hidden state `output[:, -1, :]` |
| Augmentation  | temporal scale + spatial jitter       |
| Quality gates | ✅ mypy / ruff / 204 tests            |

---

## Phase 1 — Feature & Training Hardening

**Target**: val accuracy ≥ 85%, macro F1 ≥ 84%

### Step 1.1 — Extend `LandmarkFeatureTransformer` (feature dims 126 → 134)

**File**: `src/gesture_control/pipeline/feature_engineering.py`

**1.1.a** Add fingertip index constant at module level, below the existing imports:

```python
_FINGERTIP_INDICES: list[int] = [4, 8, 12, 16, 20]  # thumb, index, middle, ring, pinky
```

**1.1.b** Add `_first_normalized` state field to `__init__`:

```python
self._first_normalized: list[list[float]] | None = None
```

**1.1.c** Reset it in `_clear_velocity_state()`:

```python
def _clear_velocity_state(self) -> None:
    self._prev_normalized = None
    self._prev_timestamp_ms = None
    self._first_normalized = None  # add this line
```

**1.1.d** In `transform()`, after the `velocity = ...` line and before building `vector`,
capture the first normalized pose and compute the two new feature blocks:

```python
# Wrist trajectory: displacement of landmark 0 from the first valid frame.
if self._first_normalized is None:
    self._first_normalized = normalized
wrist_origin = self._first_normalized[0]
wrist_traj = [
    normalized[0][0] - wrist_origin[0],
    normalized[0][1] - wrist_origin[1],
    normalized[0][2] - wrist_origin[2],
]

# Fingertip spread: 5 Euclidean distances between selected tip pairs.
tips = [normalized[i] for i in _FINGERTIP_INDICES]
spread = [
    self._distance(tips[0], tips[1]),  # thumb ↔ index
    self._distance(tips[0], tips[2]),  # thumb ↔ middle
    self._distance(tips[0], tips[3]),  # thumb ↔ ring
    self._distance(tips[0], tips[4]),  # thumb ↔ pinky
    float(sum(self._distance(tips[j], tips[k])
               for j in range(len(tips)) for k in range(j + 1, len(tips)))
          / 10.0),                     # mean pairwise spread
]

vector = self._flatten(normalized) + self._flatten(velocity) + spread + wrist_traj
```

> Feature layout after this change:
>
> - dims 0–62: position (21 × 3)
> - dims 63–125: velocity (21 × 3)
> - dims 126–130: fingertip spread (5 scalars)
> - dims 131–133: wrist trajectory (3 scalars)
> - **Total: 134 dims**

**1.1.e** Update `_zero_vector()` to return 134 zeros:

```python
@staticmethod
def _zero_vector() -> list[float]:
    return [0.0] * 134
```

**Validation**: `python -c "from gesture_control.pipeline.feature_engineering import LandmarkFeatureTransformer; print('OK')"` must succeed without error.

---

### Step 1.2 — Update Feature Pipeline Config

**File**: `configs/model/feature_pipeline.yaml`

Apply both changes atomically:

```yaml
orientation_normalization: true # was: false


# Feature vector structure after Phase 1 changes:
# [position(63) | velocity(63) | spread(5) | wrist_traj(3)] = 134 dims
```

Also update the schema version comment and add the new dim total to the bottom comment block. The `feature_schema_version` is embedded per-dataset at build time; updating the comment here serves as the ground-truth reference.

---

### Step 1.3 — Update Training Config

**File**: `configs/model/lstm_default.yaml`

```yaml
model:
  input_size: 134 # was: 126
  feature_schema_version: feature-schema-v2 # was: feature-schema-v1

export:
  feature_schema_version: feature-schema-v2 # was: feature-schema-v1
```

---

### Step 1.4 — Add Label Purity Filter to `SequenceDatasetBuilder`

**File**: `src/gesture_control/training/dataset_builder.py`

**1.4.a** Add `min_label_purity` parameter to `__init__` (after `min_valid_ratio`):

```python
min_label_purity: float = 0.9,
```

Store it: `self._min_label_purity = min_label_purity`

**1.4.b** In `add_frame()`, replace the unconditional `self._samples.append(sample)` block with a purity check. After `selected_label = self._select_label(labels)`, compute and check purity before constructing `SequenceSample`:

```python
majority_count = labels.count(selected_label)
purity = majority_count / len(labels)
if purity < self._min_label_purity:
    self._sample_index += 1
    return emitted          # discard this boundary window
```

Place this block immediately before the `sample = SequenceSample(...)` construction.

**1.4.c** Expose the parameter in `build_dataset.py` / `build_ipn_dataset.py` if those scripts instantiate `SequenceDatasetBuilder` directly. Set `min_label_purity=0.9` at all call sites.

**Validation**: Run `python -m pytest tests/unit/ -q` — all unit tests must pass.

---

### Step 1.5 — Regenerate Dataset

> **Required because**: feature dims changed (126 → 134) and orientation normalization toggled.

```bash
conda activate gesture-control
python scripts/build_dataset.py        # or build_ipn_dataset.py for IPN data
```

Verify the output `.npz` metadata:

```bash
python - <<'EOF'
import numpy as np, json
d = np.load("data/processed/gesture_dataset_v2.npz", allow_pickle=True)
meta = json.loads(str(d["metadata_json"]))
assert meta["feature_dim"] == 134, f"Expected 134, got {meta['feature_dim']}"
assert meta["enable_orientation_normalization"] is True
print("Dataset OK:", meta)
EOF
```

**Gate**: This verification must pass before proceeding to training.

---

### Step 1.6 — Add Temporal Attention to `GestureLSTMClassifier`

**File**: `src/gesture_control/infrastructure/model/lstm_model.py`

Add the attention head in `__init__` (after `self.head`):

```python
self.attn = nn.Linear(hidden_size, 1)
```

Replace the `forward` body:

```python
def forward(self, x: torch.Tensor) -> torch.Tensor:
    output, _ = self.lstm(x)                                         # (B, T, H)
    attn_w = torch.softmax(self.attn(output).squeeze(-1), dim=1)     # (B, T)
    context = (attn_w.unsqueeze(-1) * output).sum(dim=1)             # (B, H)
    return self.head(self.dropout(context))
```

**Validation**: `python -m pytest tests/unit/ -q` — existing LSTM unit tests must pass with the new readout.

---

### Step 1.7 — Add Mirror Flip Augmentation

**File**: `src/gesture_control/training/augmentation.py`

**1.7.a** Add field to `AugmentationConfig`:

```python
mirror_flip_prob: float = 0.5
```

**1.7.b** Pre-compute the x-coordinate indices at module level (update when feature dim changes):

```python
# Indices of x-coordinates across position, velocity, and wrist-trajectory blocks.
# Position block:     dims  0–62,  x at 0, 3, 6, ..., 60
# Velocity block:     dims 63–125, x at 63, 66, ..., 123
# Wrist trajectory:   dims 131–133, x at 131
# Spread (126–130): scalar distances, sign-invariant — no flip required.
_X_COORD_DIMS: list[int] = list(range(0, 63, 3)) + list(range(63, 126, 3)) + [131]
```

**1.7.c** Add `_mirror_flip` method to `GestureAugmenter`:

```python
def _mirror_flip(self, seq: np.ndarray) -> np.ndarray:
    """Negate all x-coordinates to simulate a mirrored hand."""
    result = seq.copy()
    result[:, _X_COORD_DIMS] *= -1
    return result
```

**1.7.d** Add the flip call in `augment()` after the existing spatial jitter block:

```python
if self._rng.random() < self._config.mirror_flip_prob:
    result = self._mirror_flip(result)
```

**1.7.e** Update `_POSITION_DIMS` to reflect the new layout:

```python
_POSITION_DIMS: int = 63          # unchanged; spread and wrist_traj handled separately
```

**1.7.f** Expose `mirror_flip_prob` in `_parse_augmentation_config()` in `train_lstm.py`:

```python
mirror_flip_prob=float(aug_data.get("mirror_flip_prob", 0.5)),
```

**1.7.g** Add to `configs/model/lstm_default.yaml` under `augmentation`:

```yaml
augmentation:
  enabled: true
  temporal_scale_range: [0.8, 1.2]
  spatial_jitter_sigma: 0.005
  apply_prob: 0.5
  mirror_flip_prob: 0.5 # add this line
```

**Validation**: `python -m pytest tests/unit/ -q`

---

### Step 1.8 — Add Label Smoothing

**File**: `src/gesture_control/training/train_lstm.py`

Locate the criterion construction block (around line 280 in `train_lstm()`):

```python
# Before:
criterion: nn.Module = nn.CrossEntropyLoss(weight=class_weights)
# ...
criterion = nn.CrossEntropyLoss()

# After:
criterion: nn.Module = nn.CrossEntropyLoss(weight=class_weights, label_smoothing=0.1)
# ...
criterion = nn.CrossEntropyLoss(label_smoothing=0.1)
```

Both occurrences (weighted and unweighted branch) must be updated.

---

### Step 1.9 — Full Quality Gate

Run before training:

```bash
conda activate gesture-control
python -m ruff check .
python -m mypy src
python -m pytest tests/ -m "not training" -q
```

All must report zero errors/failures.

---

### Step 1.10 — Train Phase 1 Model

```bash
python scripts/train_model.py \
  --config configs/model/lstm_default.yaml \
  --dataset data/processed/gesture_dataset_v2.npz
```

**Success criteria** (read from the run's `training_metrics.json`):

| Metric            | Required |
| ----------------- | -------- |
| Val accuracy      | ≥ 85%    |
| Val macro F1      | ≥ 84%    |
| zoom_in recall    | ≥ 80%    |
| zoom_out recall   | ≥ 80%    |
| swipe_left recall | ≥ 82%    |

If **any criterion is not met**, do not proceed to Phase 2. Diagnose via the per-class confusion matrix in `training_metrics.json`.

---

## Phase 2 — Architecture Upgrade

**Prerequisite**: Phase 1 success criteria confirmed.  
**Target**: val accuracy ≥ 88–90%

### Step 2.1 — Implement `GestureTCNClassifier`

**New file**: `src/gesture_control/infrastructure/model/tcn_model.py`

Implement a residual dilated TCN with self-attention:

```python
from __future__ import annotations

import torch
from torch import nn


class _ResidualTCNBlock(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, dilation: int) -> None:
        super().__init__()
        pad = dilation * (3 - 1)
        self.conv = nn.utils.weight_norm(
            nn.Conv1d(in_channels, out_channels, kernel_size=3,
                      padding=pad, dilation=dilation)
        )
        self.chomp = _Chomp1d(pad)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(p=0.1)
        self.residual = (
            nn.Conv1d(in_channels, out_channels, 1)
            if in_channels != out_channels else nn.Identity()
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.dropout(self.relu(self.chomp(self.conv(x))))
        return self.relu(out + self.residual(x))


class _Chomp1d(nn.Module):
    def __init__(self, chomp_size: int) -> None:
        super().__init__()
        self.chomp_size = chomp_size

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x[:, :, : -self.chomp_size].contiguous()


class GestureTCNClassifier(nn.Module):
    """Residual dilated TCN with 2-head self-attention for gesture classification."""

    def __init__(
        self,
        *,
        input_size: int,
        hidden_size: int,
        num_classes: int,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        dilations = [1, 2, 4, 8]  # receptive field = 31 > window 24
        channels = [input_size] + [hidden_size] * len(dilations)
        self.tcn = nn.Sequential(
            *[
                _ResidualTCNBlock(channels[i], channels[i + 1], d)
                for i, d in enumerate(dilations)
            ]
        )
        self.attention = nn.MultiheadAttention(
            embed_dim=hidden_size, num_heads=2, dropout=dropout, batch_first=True
        )
        self.dropout = nn.Dropout(p=dropout)
        self.head = nn.Linear(hidden_size, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, T, F) → (B, F, T) for Conv1d
        tcn_out = self.tcn(x.transpose(1, 2)).transpose(1, 2)  # (B, T, H)
        attn_out, _ = self.attention(tcn_out, tcn_out, tcn_out)
        context = attn_out.mean(dim=1)                          # (B, H)
        return self.head(self.dropout(context))
```

**Validation**: `python -m pytest tests/unit/ -q` and verify no import errors.

---

### Step 2.2 — Add Joint Angle Features (feature dims 134 → 144)

**File**: `src/gesture_control/pipeline/feature_engineering.py`

**2.2.a** Add finger joint index triplets at module level:

```python
# (landmark_a, vertex, landmark_b) for MCP and PIP of each finger.
_JOINT_TRIPLETS: list[tuple[int, int, int]] = [
    (0, 1, 2),   # thumb CMC
    (1, 2, 3),   # thumb MCP
    (0, 5, 6),   # index MCP
    (5, 6, 7),   # index PIP
    (0, 9, 10),  # middle MCP
    (9, 10, 11), # middle PIP
    (0, 13, 14), # ring MCP
    (13, 14, 15),# ring PIP
    (0, 17, 18), # pinky MCP
    (17, 18, 19),# pinky PIP
]
```

**2.2.b** Add `_joint_angles` method to `LandmarkFeatureTransformer`:

```python
def _joint_angles(self, normalized: list[list[float]]) -> list[float]:
    angles: list[float] = []
    for a_idx, v_idx, b_idx in _JOINT_TRIPLETS:
        a = self._subtract(normalized[a_idx], normalized[v_idx])
        b = self._subtract(normalized[b_idx], normalized[v_idx])
        norm_a = math.sqrt(sum(c * c for c in a))
        norm_b = math.sqrt(sum(c * c for c in b))
        if norm_a < 1e-9 or norm_b < 1e-9:
            angles.append(0.0)
        else:
            cos_theta = max(-1.0, min(1.0, self._dot(a, b) / (norm_a * norm_b)))
            angles.append(math.acos(cos_theta))
    return angles
```

**2.2.c** Append joint angles to `vector` in `transform()`:

```python
angles = self._joint_angles(normalized)
vector = (
    self._flatten(normalized)
    + self._flatten(velocity)
    + spread
    + wrist_traj
    + angles
)
```

**2.2.d** Update `_zero_vector()`:

```python
return [0.0] * 144
```

> Feature layout after Step 2.2:
>
> - dims 0–62: position (63)
> - dims 63–125: velocity (63)
> - dims 126–130: fingertip spread (5)
> - dims 131–133: wrist trajectory (3)
> - dims 134–143: joint angles (10)
> - **Total: 144 dims**

**2.2.e** Update `_X_COORD_DIMS` in `augmentation.py` — joint angles are rotation-invariant scalars, no x-flip needed; existing list remains valid.

---

### Step 2.3 — Create TCN Training Config

**New file**: `configs/model/tcn_default.yaml`

```yaml
model:
  model_name: tcn_v1
  input_size: 144
  hidden_size: 128
  num_layers: 4 # number of TCN dilation blocks
  dropout: 0.1
  num_classes: 8
  feature_schema_version: feature-schema-v3

training:
  epochs: 40
  batch_size: 64
  learning_rate: 0.001
  weight_decay: 0.0001
  grad_clip_norm: 1.0
  early_stopping_patience: 8
  early_stopping_min_delta: 0.0001
  seed: 42
  num_workers: 0
  class_weighting: true

splits:
  train_ratio: 0.7
  val_ratio: 0.15
  test_ratio: 0.15
  stratified: true

scheduler:
  enabled: true
  factor: 0.5
  patience: 4
  min_lr: 0.000001

export:
  checkpoint_dir: models/checkpoints
  exported_dir: models/exported
  registry_path: models/registry.json
  feature_schema_version: feature-schema-v3

augmentation:
  enabled: true
  temporal_scale_range: [0.8, 1.2]
  spatial_jitter_sigma: 0.005
  apply_prob: 0.5
  mirror_flip_prob: 0.5
```

---

### Step 2.4 — Update `train_lstm.py` to Dispatch on Model Name

**File**: `src/gesture_control/training/train_lstm.py`

Add a model factory before the `model = GestureLSTMClassifier(...)` line:

```python
from gesture_control.infrastructure.model.tcn_model import GestureTCNClassifier

def _build_model(config: ModelConfig) -> nn.Module:
    if config.model_name.startswith("tcn"):
        return GestureTCNClassifier(
            input_size=config.input_size,
            hidden_size=config.hidden_size,
            num_classes=config.num_classes,
            dropout=config.dropout,
        )
    return GestureLSTMClassifier(
        input_size=config.input_size,
        hidden_size=config.hidden_size,
        num_layers=config.num_layers,
        dropout=config.dropout,
        num_classes=config.num_classes,
    )
```

Replace `model = GestureLSTMClassifier(...).to(device)` with:

```python
model = _build_model(config.model).to(device)
```

---

### Step 2.5 — Regenerate Dataset for Phase 2

```bash
python scripts/build_dataset.py
```

Verify:

```bash
python - <<'EOF'
import numpy as np, json
d = np.load("data/processed/gesture_dataset_v2.npz", allow_pickle=True)
meta = json.loads(str(d["metadata_json"]))
assert meta["feature_dim"] == 144, f"Expected 144, got {meta['feature_dim']}"
print("Dataset OK:", meta)
EOF
```

---

### Step 2.6 — Full Quality Gate

```bash
python -m ruff check .
python -m mypy src
python -m pytest tests/ -m "not training" -q
```

---

### Step 2.7 — Train Phase 2 Model

```bash
python scripts/train_model.py \
  --config configs/model/tcn_default.yaml \
  --dataset data/processed/gesture_dataset_v2.npz
```

**Success criteria**:

| Metric            | Required |
| ----------------- | -------- |
| Val accuracy      | ≥ 88%    |
| Val macro F1      | ≥ 87%    |
| zoom_in recall    | ≥ 85%    |
| zoom_out recall   | ≥ 85%    |
| swipe_left recall | ≥ 87%    |

---

## Dependency Graph

```
Step 1.1  feature_engineering.py  ─┐
Step 1.2  feature_pipeline.yaml    ├──► Step 1.5  build_dataset.py (regen, dim=134)
Step 1.3  lstm_default.yaml       ─┘

Step 1.4  dataset_builder.py ──────────► Step 1.5  (purity applied at build time)

Steps 1.6, 1.7, 1.8 are independent of each other
(lstm_model.py, augmentation.py, train_lstm.py)
All must complete before Step 1.9 quality gate.

                    ▼
             Step 1.10  train (dim=134, LSTM+attn)
                    ▼
           Phase 1 gate: val ≥ 85%
                    ▼
Step 2.2  +joint angles → dim=144
Step 2.3  tcn_default.yaml
Step 2.4  train_lstm.py model dispatch
                    ▼
             Step 2.5  build_dataset.py (regen, dim=144)
                    ▼
             Step 2.7  train (dim=144, TCN+attn)
                    ▼
           Phase 2 gate: val ≥ 88%
```

---

## Change Summary per File

| File                                                     | Phase | Change                                             |
| -------------------------------------------------------- | ----- | -------------------------------------------------- |
| `src/gesture_control/pipeline/feature_engineering.py`    | 1 + 2 | Add spread, wrist traj (→134), joint angles (→144) |
| `src/gesture_control/training/augmentation.py`           | 1     | Add mirror flip; update `_X_COORD_DIMS`            |
| `src/gesture_control/training/dataset_builder.py`        | 1     | Add `min_label_purity=0.9` filter                  |
| `src/gesture_control/training/train_lstm.py`             | 1 + 2 | `label_smoothing=0.1`; model factory dispatch      |
| `src/gesture_control/infrastructure/model/lstm_model.py` | 1     | Replace final-step with attention-weighted readout |
| `src/gesture_control/infrastructure/model/tcn_model.py`  | 2     | New — `GestureTCNClassifier`                       |
| `configs/model/feature_pipeline.yaml`                    | 1     | `orientation_normalization: true`                  |
| `configs/model/lstm_default.yaml`                        | 1     | `input_size: 134`, schema v2                       |
| `configs/model/tcn_default.yaml`                         | 2     | New — TCN training config, `input_size: 144`       |

---

**Document**: Gesture Recognition Enhancement — Implementation Plan  
**Suite**: Implementation Documentation (`docs/implementation/`)  
**Last Updated**: April 2026
